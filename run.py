import argparse
import sys
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from src.pipeline import process_image


def check_env():
    if not os.environ.get("FAL_KEY"):
        print("ERROR: FAL_KEY environment variable is not set.")
        print("Get your key at https://fal.ai and run: export FAL_KEY=your_key_here")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Replace TV screens with an overlay image.")
    parser.add_argument("--input_dir", required=True, help="Directory of source images")
    parser.add_argument("--output_dir", required=True, help="Directory for output images")
    parser.add_argument("--overlay", default="image.jpg", help="Overlay image (default: image.jpg)")
    parser.add_argument("--debug", action="store_true", help="Print raw fal.ai API responses and save debug images")
    args = parser.parse_args()

    check_env()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    overlay_path = Path(args.overlay)

    if not input_dir.exists():
        print(f"ERROR: --input_dir does not exist: {input_dir}")
        sys.exit(1)
    if not overlay_path.exists():
        print(f"ERROR: overlay image not found: {overlay_path}")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir: Path | None = None
    if args.debug:
        debug_dir = Path("debug_output")
        debug_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in extensions)
    if not images:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} image(s) to process")

    ok = 0
    worker = partial(
        process_image,
        output_dir=output_dir,
        overlay_path=overlay_path,
        debug=args.debug,
        debug_dir=debug_dir,
    )
    with Pool(processes=4) as pool:
        try:
            for result, logs in pool.imap_unordered(worker, images):
                print("\n".join(logs))
                if result:
                    ok += 1
        except KeyboardInterrupt:
            print("\nInterrupted — terminating workers.")
            pool.terminate()
            sys.exit(1)

    print(f"\nDone: {ok}/{len(images)} succeeded.")


if __name__ == "__main__":
    main()
