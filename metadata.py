import re
from clams.app import ClamsApp
from clams.appmetadata import AppMetadata
from mmif import DocumentTypes, AnnotationTypes

def appmetadata() -> AppMetadata:
    md = AppMetadata(
        name="Spoken LID (VoxLingua107-ECAPA)",
        description=(
            "Chunk-level language identification using SpeechBrain's "
            "VoxLingua107 ECAPA model. Produces TimeFrame annotations "
            "with top-N language scores."
        ),
        app_license="Apache 2.0",
        url="https://apps.clams.ai/",
        identifier="https://apps.clams.ai/voxlingua-lid"
    )
    # I/O
    md.add_input(DocumentTypes.AudioDocument)
    md.add_output(AnnotationTypes.TimeFrame)

    # Parameters
    md.add_parameter(
        name="chunk",
        type="number",
        default=30.0,
        description="Chunk length in seconds for windowed LID."
    )
    md.add_parameter(
        name="top",
        type="integer",
        default=3,
        description="Top-N language scores to keep per chunk."
    )
    md.add_parameter(
        name="device",
        type="string",
        default="auto",
        choices=["cpu", "cuda", "auto"],
        description="Inference device selection."
    )
    return md

class App(ClamsApp):
    def __init__(self):
        super().__init__()
        self.__metadata = appmetadata()

    def _appmetadata(self):
        return self.__metadata

    def _annotate(self, mmif, **params):
        from mmif import Mmif
        from mmif.utils import generate_uuid
        from lapps.discriminators import Uri
        import json

        # Lazy import to keep init lightweight
        from app import VoxLinguaLID, load_audio_16k, chunk_audio

        # Resolve params
        chunk = float(params.get("chunk", 30.0))
        top = int(params.get("top", 3))
        device = params.get("device", "auto")

        # Prepare output MMIF structures
        mmif_obj = Mmif(mmif) if isinstance(mmif, (str, bytes, dict)) else mmif
        new_view = mmif_obj.new_view()
        self.sign_view(new_view, params)

        # Declare what this view will contain
        new_view.new_contain(
            AnnotationTypes.TimeFrame, 
            document=None,  # will set per-annotation
            timeUnit="milliseconds"
        )

        # Build model once
        model = VoxLinguaLID(device=device)

        # Iterate audio documents
        for doc in mmif_obj.documents:
            if doc.at_type != DocumentTypes.AudioDocument:
                continue

            # Try to resolve location
            loc = doc.location
            if loc and loc.startswith("file://"):
                path = loc.replace("file://", "")
            else:
                path = loc or ""

            if not path:
                continue

            # Load + chunk
            wave, sr = load_audio_16k(path)
            chunks = chunk_audio(wave, sr, chunk)

            for idx, chunk_wav in enumerate(chunks):
                if len(chunk_wav) == 0:
                    continue
                start_s = idx * chunk
                end_s = (idx + 1) * chunk

                pred, topN = model.infer_chunk(chunk_wav, top=top)

                # Create a TimeFrame
                tf = new_view.new_annotation(AnnotationTypes.TimeFrame)
                tf.add_property("start", int(start_s * 1000))
                tf.add_property("end", int(end_s * 1000))
                tf.add_property("label", pred)
                tf.add_property("scores", [{"label": l, "score": float(s)} for l, s in topN])
                tf.add_property("document", doc.id)

        return mmif_obj.serialize(pretty=True)
