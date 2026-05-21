"""
Download source images from s3://autohdr-tv-test-project/images/.

Only files with "src" in the filename are downloaded.
Files with "tar" in the name are reference/target examples and are skipped.

Usage:
    python download_images.py --output_dir ./input_images --limit 10
"""

import argparse
import sys
from pathlib import Path


BUCKET = "autohdr-tv-test-project"
DEFAULT_PREFIX = "images/"


def download_images(output_dir: str, limit: int | None, prefix: str) -> int:
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        from botocore.exceptions import ClientError
    except ImportError:
        print("ERROR: boto3 not installed. Add it to your environment.")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET, Prefix=prefix)

        downloaded = 0
        skipped_tar = 0
        skipped_no_src = 0
        already_present = 0

        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                filename = Path(key).name

                if not filename:
                    continue

                # Skip reference/target examples
                if "tar" in filename.lower():
                    skipped_tar += 1
                    continue

                # Only process source images
                if "src" not in filename.lower():
                    skipped_no_src += 1
                    continue

                local_path = output_path / filename

                if local_path.exists():
                    already_present += 1
                    continue

                print(f"  Downloading: {filename}")
                s3.download_file(BUCKET, key, str(local_path))
                downloaded += 1

                if limit is not None and downloaded >= limit:
                    print(f"\nReached limit of {limit}.")
                    _print_summary(downloaded, skipped_tar, skipped_no_src, already_present)
                    return downloaded

        _print_summary(downloaded, skipped_tar, skipped_no_src, already_present)
        return downloaded

    except ClientError as e:
        print(f"ERROR: AWS error: {e}")
        sys.exit(1)


def _print_summary(downloaded, skipped_tar, skipped_no_src, already_present):
    print(
        f"\nSummary:\n"
        f"  Downloaded:      {downloaded}\n"
        f"  Already present: {already_present}\n"
        f"  Skipped (tar):   {skipped_tar}\n"
        f"  Skipped (other): {skipped_no_src}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download src images from s3://autohdr-tv-test-project/images/"
    )
    parser.add_argument(
        "--output_dir",
        default="./input_images",
        help="Local directory to save images (default: ./input_images)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of images to download (default: all)",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"S3 key prefix to list under (default: {DEFAULT_PREFIX!r})",
    )
    args = parser.parse_args()

    print(f"Bucket:     s3://{BUCKET}/{args.prefix}")
    print(f"Output dir: {args.output_dir}")
    if args.limit:
        print(f"Limit:      {args.limit} images")
    print()

    download_images(args.output_dir, args.limit, args.prefix)


if __name__ == "__main__":
    main()
