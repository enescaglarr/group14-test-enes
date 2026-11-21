#!/usr/bin/env python3
"""
Submission entrypoint contract:
    python predict.py <input_dir> <output_dir>

<input_dir>  contains one image per table, named <table_id>.png or .jpg
<output_dir> must receive one <table_id>.json (preferred) or .html each

Runs tableparser on every image via tableparser_adapter (README §5 cell list,
one file written as soon as its table is done). The official run happens
offline (no network); Tesseract deu+eng and the ONNX weights are in the image.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tableparser_adapter import main  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python predict.py <input_dir> <output_dir>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
