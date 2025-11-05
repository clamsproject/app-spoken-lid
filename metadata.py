"""
The purpose of this file is to define the metadata of the app with minimal imports.

DO NOT CHANGE the name of the file
"""
import re
from clams.app import ClamsApp
from clams.appmetadata import AppMetadata
from mmif import DocumentTypes, AnnotationTypes

whisper_version = 'v' + [line.strip().rsplit('==')[-1]
                   for line in open('requirements.txt').readlines() if re.match(r'^openai-whisper==', line)][0]
whisper_lang_list = f"https://raw.githubusercontent.com/openai/whisper/refs/tags/{whisper_version}/whisper/tokenizer.py"


# DO NOT CHANGE the function name
def appmetadata():
    metadata = AppMetadata(
        name="Spoken Language Identification",
        identifier="spoken-lid",
        url="https://github.com/clamsproject/app-spoken-lid",
        description="Chunk-level language ID over audio based on OpenAI Whisper",
        app_license="Apache 2.0",
        analyzer_version=whisper_version,
        analyzer_license="MIT",
    )

    # input
    metadata.add_input_oneof(DocumentTypes.AudioDocument, DocumentTypes.VideoDocument)

    # output
    metadata.add_output(AnnotationTypes.TimeFrame, timeUnit="seconds", labalSet=whisper_lang_list)

    # parameters
    metadata.add_parameter(name="model", type="string", default="tiny", description="Whisper model size",
                           choices=["tiny", "base", "small", "medium", "large", "turbo"])
    metadata.add_parameter(name="chunk", type="number", default=30, description="chunk/window length in seconds")
    metadata.add_parameter(name="top", type="integer", default=3, description="top-k language scores")
    metadata.add_parameter(name="batchSize", type="integer", default=1,
                           description="number of windows processed in a batch")

    return metadata

    # DO NOT CHANGE the main block


if __name__ == '__main__':
    import sys

    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
