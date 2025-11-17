import argparse
import logging
import os
import pathlib
from itertools import chain
from typing import Dict, Generator, Tuple, List, Union

import librosa
import torch
from clams import ClamsApp, Restifier
from mmif import Mmif, View, AnnotationTypes, DocumentTypes

from lid_util_ambernet import detect_language_by_chunk_ambernet, AMBERNET_LANGUAGES


# Force CPU mode for broader GPU compatibility --> TODO: need to check cuda version 
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def load_audio_mono16(path: Union[str, pathlib.Path], sr: int = 16_000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav, sr


def chunk_audio(wave, sr: int, window_sec: float = 30.0):
    # yield numpy arrays.
    window = int(window_sec * sr)
    for i in range(0, len(wave), window):
        chunk = wave[i: i + window]
        start_ms = int(1_000 * i / sr)
        end_ms = int(1_000 * (i + len(chunk)) / sr)
        yield chunk, start_ms, end_ms


class SpokenLIDAmberNetWrapper(ClamsApp):
    """AmberNet-based Spoken-Language-ID wrapped as a CLAMS app."""

    def __init__(self):
        super().__init__()
        self._model = None
        self.labelset: List[str] = AMBERNET_LANGUAGES

    def _appmetadata(self):
        from metadata_ambernet import appmetadata
        return appmetadata()

    def _get_model(self):
        """Lazy load AmberNet model (cached)."""
        if self._model is None:
            import nemo.collections.asr as nemo_asr
            self.logger.debug("Loading AmberNet model (langid_ambernet)")
            # This will auto-download from NGC if not present
            self._model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("langid_ambernet")
            # Set to eval mode
            self._model.eval()
            self.logger.debug("AmberNet model loaded successfully")
        return self._model

    def _annotate(self, mmif_input: Union[str, dict, Mmif], **params) -> Mmif:
        # with TimeFrame annotations w/ISO 639-3 language labels, and `classification` dict of log prob
        if isinstance(mmif_input, Mmif):
            mmif = mmif_input
        else:
            mmif = Mmif(mmif_input)

        window_sec = float(params.get("chunk", 30))
        top_k = int(params.get("top", 3))

        model = self._get_model()

        # process both audio and video documents
        for doc in chain(mmif.get_documents_by_type(DocumentTypes.AudioDocument),
                         mmif.get_documents_by_type(DocumentTypes.VideoDocument)):
            audio_path = doc.location_path(nonexist_ok=False)
            wave, sr = load_audio_mono16(audio_path)

            view: View = mmif.new_view()
            self.sign_view(view, params)

            view.new_contain(
                AnnotationTypes.TimeFrame,
                document=doc.id,
                timeUnit="milliseconds",
            )

            for chunk, start_ms, end_ms in chunk_audio(wave, sr, window_sec):
                if len(chunk) == 0:
                    continue

                if len(chunk) < sr:
                    self.logger.debug(f"Skipping short chunk {start_ms}-{end_ms}ms")
                    continue

                # AmberNet inference on this chunk
                probs_sorted, iso_code = detect_language_by_chunk_ambernet(
                    model, chunk, sr, top_k
                )

                tf = view.new_annotation(AnnotationTypes.TimeFrame)
                tf.add_property("start", start_ms)
                tf.add_property("end", end_ms)
                
                # Clean ISO code (remove any non-alphanumeric characters)
                iso_code = ''.join(c for c in iso_code if c.isalnum())
                tf.add_property("label", iso_code)
                
                # Add top-k probabilities
                tf.add_property("classification",
                                dict(list(probs_sorted.items())[:top_k]))

        return mmif


def get_app():
    return SpokenLIDAmberNetWrapper()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    parsed_args = parser.parse_args()

    # Create the app instance
    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    # For running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # Development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
