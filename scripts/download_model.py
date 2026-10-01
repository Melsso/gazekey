import argparse
import sys
from pathlib import Path

from gazekey import config
from gazekey.vision.model import download_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("-o", "--output", type=Path, default=config.default_model_path())
    parser.add_argument("-f", "--force", action="store_true", help="re-download if present")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"{args.output} already exists (use -f to re-download).")
        return 0
    print(f"Downloading {config.MODEL_URL}\n  -> {args.output}")
    digest = download_model(config.MODEL_URL, args.output, expected_sha256=config.MODEL_SHA256)
    print(f"Done. SHA-256: {digest}")
    if config.MODEL_SHA256 is None:
        print("MODEL_SHA256 is not pinned yet; consider pinning the hash above in config.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
