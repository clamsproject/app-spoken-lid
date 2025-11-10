import argparse
import logging
import pathlib
from itertools import chain
from typing import Dict, Generator, Tuple, List, Union

import librosa
import torch
import whisper
from clams import ClamsApp, Restifier
from mmif import Mmif, View, AnnotationTypes, DocumentTypes
from whisper.tokenizer import LANGUAGES, get_tokenizer

from lid_util import detect_language_by_chunk


def load_audio_mono16(path: Union[str, pathlib.Path], sr: int = 16_000):
    """Return waveform (mono, 16 kHz) and sample-rate."""
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav, sr


def chunk_audio(
        wave, sr: int, window_sec: float = 30.0
) -> Generator[Tuple[torch.Tensor, int, int], None, None]:
    """
    Yield (chunk, start_ms, end_ms).  
    `whisper.pad_or_trim` will pad/trim to 30 s anyway, so we don’t overlap.
    """
    window = int(window_sec * sr)
    for i in range(0, len(wave), window):
        chunk = wave[i: i + window]
        start_ms = int(1_000 * i / sr)
        end_ms = int(1_000 * (i + len(chunk)) / sr)
        yield chunk, start_ms, end_ms


def _get_tokenizer_cached(cache: Dict[str, "whisper.tokenizer.Tokenizer"], key: str, model):
    """Cache tokenizers per *model size* string (tiny, small, …)."""
    if key not in cache:
        try:
            cache[key] = get_tokenizer(model.is_multilingual)
        except TypeError:
            cache[key] = get_tokenizer(multilingual=model.is_multilingual, task="lang_id")
    return cache[key]


class SpokenLIDWrapper(ClamsApp):
    """Whisper-based Spoken-Language-ID wrapped as a CLAMS app."""

    def __init__(self):
        super().__init__()
        self._models: Dict[str, whisper.Whisper] = {}
        self._tokenizers: Dict[str, "whisper.tokenizer.Tokenizer"] = {}
        self.labelset: List[str] = list(LANGUAGES)

    def _appmetadata(self):
        from metadata import appmetadata
        return appmetadata()

    def _get_model(self, size: str):
        if size not in self._models:
            self.logger.debug(f"Loading Whisper model: {size}")
            self._models[size] = whisper.load_model(size)
        return self._models[size]

    def _annotate(self, mmif_input: Union[str, dict, Mmif], **params) -> Mmif:
        """
        Attach TimeFrame annotations with ISO 639-3 language labels and a
        `classification` dict of log-probabilities.
        """
        if isinstance(mmif_input, Mmif):
            mmif = mmif_input
        else:
            mmif = Mmif(mmif_input)

        model_size = params.get("model", "tiny")
        window_sec = float(params.get("chunk", 30))
        top_k = int(params.get("top", 3))

        model = self._get_model(model_size)
        # tokenizer = _get_tokenizer_cached(self._tokenizers, model)
        tokenizer = _get_tokenizer_cached(self._tokenizers, model_size, model)

        for doc in chain(mmif.get_documents_by_type(DocumentTypes.AudioDocument),
                         mmif.get_documents_by_type(DocumentTypes.VideoDocument)):
            audio_path = doc.location_path(nonexist_ok=False)
            wave, sr = load_audio_mono16(audio_path)

            # create view
            view: View = mmif.new_view()
            self.sign_view(view, params)

            view.new_contain(
                AnnotationTypes.TimeFrame,
                document=doc.id,
                timeUnit="milliseconds",
                # labelset=self.labelset,
            )

            for chunk, start_ms, end_ms in chunk_audio(wave, sr, window_sec):
                if len(chunk) == 0:
                    continue

                mel = whisper.log_mel_spectrogram(
                    whisper.pad_or_trim(torch.tensor(chunk, dtype=torch.float32).to(model.device)),
                    n_mels=model.dims.n_mels,
                ).unsqueeze(0)
                probs_sorted, iso_code = detect_language_by_chunk(model, mel, tokenizer)
                tf = view.new_annotation(AnnotationTypes.TimeFrame)
                tf.add_property("start", start_ms)
                tf.add_property("end", end_ms)
                # strip LLM affixes if present, keeping alphanumerics only
                iso_code = ''.join(c for c in iso_code if c.isalnum())
                tf.add_property("label", iso_code)
                tf.add_property("classification",
                                dict(list(probs_sorted.items())[:top_k]))

        return mmif


def get_app():
    """Required by CLAMS runner/cli – returns a ready-to-use app."""
    return SpokenLIDWrapper()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    parsed_args = parser.parse_args()

    # create the app instance
    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
