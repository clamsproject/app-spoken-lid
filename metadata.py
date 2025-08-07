"""
The purpose of this file is to define the metadata of the app with minimal imports.

DO NOT CHANGE the name of the file
"""
import re
from clams.app import ClamsApp
from clams.appmetadata import AppMetadata
from mmif import DocumentTypes, AnnotationTypes


whisper_version = [line.strip().rsplit('==')[-1]
                   for line in open('requirements.txt').readlines() if re.match(r'^openai-whisper==', line)][0]
whisper_lang_list = f"https://github.com/openai/whisper/blob/{whisper_version}/whisper/tokenizer.py"


# DO NOT CHANGE the function name
def appmetadata():
    metadata = AppMetadata(
        name="Spoken Language ID Wrapper",
        identifier="spoken-lid",
        url="https://github.com/clamsproject/app-spoken-lid",
        description="Chunk-level language ID over audio based on OpenAI Whisper",
        app_version="1.0.0",
        app_license="Apache 2.0",
        analyzer_version=whisper_version,
        analyzer_license="MIT",
    )

    
    # input
    metadata.add_input_oneof(DocumentTypes.AudioDocument, DocumentTypes.VideoDocument)
    # metadata.add_input(AnnotationTypes.TimeFrame, multiple=True, optional=True)

    # output
    timeframe_output = metadata.add_output(AnnotationTypes.TimeFrame, timeUnit="seconds")
    # timeframe_output.add_property(
    #    "label",
    #     "str",
    #     "language code for the most-likely language in this window",
    # )
    # timeframe_output.add_property(
    #     "classification",
    #     "dict",
    #     "{lang_code: probability} map of the topN lang probability scores",
    # )

    # view-level 
    # timeframe_output.add_metadata("lang_labels", whisper_lang_list)
    # timeframe_output.metadata["labelsetURL"] = whisper_lang_list
    

    # parameters
    metadata.add_parameter(name="modelSize", type="string", default="tiny", description="Whisper model size",
                       choices=["tiny","base","small","medium","large","turbo"])
    metadata.add_parameter(name="chunk", type="number", default=30, description="chunk/window length in seconds")
    metadata.add_parameter(name="top", type="integer", default=3, description="top languague scores")
    metadata.add_parameter(name="batchSize", type="integer", default=1, description="number of windows processed per forward pass")

    return metadata


    # DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
