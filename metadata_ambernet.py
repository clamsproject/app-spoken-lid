from clams.app import ClamsApp
from clams.appmetadata import AppMetadata
from mmif import DocumentTypes, AnnotationTypes

ambernet_version = 'v1.12.0'
ambernet_lang_list = "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/ambernet"


# DO NOT CHANGE the function name
def appmetadata():
    metadata = AppMetadata(
        name="Spoken Language Identification (AmberNet)",
        identifier="spoken-lid-ambernet",
        url="https://github.com/clamsproject/app-spoken-lid-ambernet",
        description="Chunk-level language ID over audio based on NVIDIA AmberNet (NeMo)",
        app_license="Apache 2.0",
        analyzer_version=ambernet_version,
        analyzer_license="CC-BY-4.0",  # NeMo models are typically CC-BY-4.0
    )

    # Input
    metadata.add_input_oneof(DocumentTypes.AudioDocument, DocumentTypes.VideoDocument)

    # Output
    metadata.add_output(
        AnnotationTypes.TimeFrame, 
        timeUnit="milliseconds",
        labelSet=ambernet_lang_list
    )

    # Parameters
    #  no "model" parameter
    metadata.add_parameter(
        name="chunk",
        type="number",
        default=30,
        description="chunk/window length in seconds for language detection"
    )
    
    metadata.add_parameter(
        name="top",
        type="integer",
        default=3,
        description="top-k language scores to include in classification"
    )

    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys

    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
