#!/usr/bin/env python3
"""
Thin CLI interface for the Spoken-LID CLAMS app.

Keeps all argument definitions in one place (metadata.py) so you
never have to update this file when you add or rename parameters.

DO NOT RENAME this file – CLAMS tooling looks specifically for
`cli.py` when generating container entrypoints.
"""

import argparse
import sys
from contextlib import redirect_stdout

import app                        
from clams import AppMetadata
import clams.app                 


def metadata_to_argparser(app_metadata: AppMetadata) -> argparse.ArgumentParser:
    """Generate ArgumentParser directly from metadata.py."""
    parser = argparse.ArgumentParser(
        description=(
            f"{app_metadata.name}: {app_metadata.description} "
            f"(see {app_metadata.url} for details)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


    for p in app_metadata.parameters:
        kw = dict(help=p.description)
        flag = f"--{p.name}"

        if p.multivalued:
            kw.update(nargs="+", action="extend", type=str)
        else:
            kw.update(nargs=1, action="store", type=str)

        arg = parser.add_argument(flag, **kw)

        if p.choices:
            arg.choices = p.choices
        if p.default is not None:
            default_help = f"(default: {p.default}"
            if p.type == "boolean":
                default_help += (
                    f", any value except "
                    f"{[v for v in clams.app.falsy_values if isinstance(v, str)]} "
                    "interpreted as True"
                )
            arg.help += " " + default_help + ")"

    
    parser.add_argument(
        "IN_MMIF_FILE",
        nargs="?",
        type=argparse.FileType("r"),
        default=None if sys.stdin.isatty() else sys.stdin,
        help=(
            "Input MMIF path, or STDIN if ‘-’ or omitted. "
            "When piping in containers, use `-i` to keep stdin open."
        ),
    )
    parser.add_argument(
        "OUT_MMIF_FILE",
        nargs="?",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help=(
            "Output MMIF path, or STDOUT if ‘-’ or omitted. "
            "If STDOUT, any print() output from the app is redirected to stderr."
        ),
    )
    return parser



if __name__ == "__main__":
    clamsapp = app.get_app()          
    arg_parser = metadata_to_argparser(clamsapp.metadata)
    args = arg_parser.parse_args()

    if args.IN_MMIF_FILE:
        in_mmif = args.IN_MMIF_FILE.read()

        params = {}
        for name, value in vars(args).items():
            if name in {"IN_MMIF_FILE", "OUT_MMIF_FILE"} or value is None:
                continue
            elif isinstance(value, list):
                params[name] = value  
            else:
                params[name] = [value] 

        if args.OUT_MMIF_FILE is sys.stdout:
            with redirect_stdout(sys.stderr):
                out_mmif = clamsapp.annotate(in_mmif, **params)
        else:
            out_mmif = clamsapp.annotate(in_mmif, **params)

        args.OUT_MMIF_FILE.write(out_mmif)
    else:
        arg_parser.print_help()
        sys.exit(1)
