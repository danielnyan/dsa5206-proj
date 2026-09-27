from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source.parent)
            parts = set(relative.parts)
            if ".git" in parts or "__pycache__" in parts or path.suffix == ".pyc":
                continue
            if path.is_file():
                archive.write(path, relative)
    print(args.output)


if __name__ == "__main__":
    main()
