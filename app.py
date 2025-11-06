#!/usr/bin/env python3
"""
CLAMS app for Spoken Language Identification using SpeechBrain VoxLingua107 (ECAPA).

Mirrors the structure of Whisper-LID CLAMS app
"""

import os
import numpy as np
import torch
import librosa
from clams.app import ClamsApp
from mmif import Mmif, AnnotationTypes, DocumentTypes
from speechbrain.inference.classifiers import EncoderClassifier
import ffmpeg


def load_audio_16k(path):
    wave, sr = librosa.load(path, sr=16000, mono=True)
    return wave, sr


def chunk_audio(wave, sr, chunk_sec=30.0):
    window = int(chunk_sec * sr)
    return [wave[i:i + window] for i in range(0, len(wave), window)]


def _topk_from_logprobs(logprobs, k, ind2lab):
    # top-k language predictions and their scores
    probs = logprobs.exp()
    k = min(k, probs.numel())
    vals, idxs = torch.topk(probs, k)
    return [(ind2lab[int(i)], float(v)) for i, v in zip(idxs, vals)]


class VoxLinguaModel:
    # SpeechBrain ECAPA-based VoxLingua107 model wrapper
    def __init__(self, device="auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir="voxlingua_cache",
            run_opts={"device": self.device},
        )
        self.ind2lab = self.classifier.hparams.label_encoder.ind2lab

    def infer_chunk(self, chunk_wav, top=3):
        wav = torch.from_numpy(chunk_wav.astype(np.float32)).unsqueeze(0).to(self.device)
        wav_lens = torch.tensor([1.0], dtype=torch.float32, device=self.device)
        logp, _, _, _ = self.classifier.classify_batch(wav, wav_lens)
        logp = logp.squeeze(0)
        topN = _topk_from_logprobs(logp, top, self.ind2lab)
        pred = topN[0][0] if topN else ""
        return pred, topN


class VoxLinguaLID(ClamsApp):
    # same interface as WhisperLID."""
    def __init__(self):
        super().__init__()
        from metadata import appmetadata
        self.__metadata = appmetadata()

    def _appmetadata(self):
        return self.__metadata

    def _annotate(self, mmif, **params):
        """Annotate audio documents with chunk-level language predictions."""
        # normalize params like Whisper app
        def _get_param(name, default, caster):
            v = params.get(name, default)
            if isinstance(v, list):
                v = v[0] if v else default
            return caster(v)

        chunk = _get_param("chunk", 30.0, float)
        top = _get_param("top", 3, int)
        device = _get_param("device", "auto", str)

        mmif_obj = Mmif(mmif) if not isinstance(mmif, Mmif) else mmif
        new_view = mmif_obj.new_view()
        self.sign_view(new_view, params)
        new_view.new_contain(AnnotationTypes.TimeFrame, timeUnit="milliseconds")

        model = VoxLinguaModel(device=device)

        for doc in mmif_obj.documents:
            if doc.at_type != DocumentTypes.AudioDocument:
                continue
            loc = doc.location
            path = loc.replace("file://", "") if loc and loc.startswith("file://") else loc
            if not path or not os.path.exists(path):
                continue

            wave, sr = load_audio_16k(path)
            chunks = chunk_audio(wave, sr, chunk)

            for idx, chunk_wav in enumerate(chunks):
                if len(chunk_wav) == 0:
                    continue
                start_s = idx * chunk
                end_s = (idx + 1) * chunk
                pred, topN = model.infer_chunk(chunk_wav, top=top)

                tf = new_view.new_annotation(AnnotationTypes.TimeFrame)
                tf.add_property("start", int(start_s * 1000))
                tf.add_property("end", int(end_s * 1000))
                tf.add_property("label", pred)
                tf.add_property("scores", [{"label": l, "score": float(s)} for l, s in topN])
                tf.add_property("document", doc.id)

        return mmif


def get_app():
    return VoxLinguaLID()
