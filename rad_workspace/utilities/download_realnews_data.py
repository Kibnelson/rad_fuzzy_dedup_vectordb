#!/usr/bin/env python3
"""
Download a file from a URL (with streaming, progress bar, and resume support).

Examples:
  python download.py --url https://storage.googleapis.com/grover-models/realnews.tar.gz
  python download.py --out realnews.tar.gz
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import urllib.error


DEFAULT_URL = "https://storage.googleapis.com/grover-models/realnews.tar.gz"


def _format_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}"


def download(url: str, out_path: str, chunk_size: int = 1024 * 1024) -> None:
    tmp_path = out_path + ".part"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    # Resume if partial file exists
    existing = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
    headers = {}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req) as resp:
            # Total size might be unknown or may be "remaining" when ranged.
            # Try to infer full size when possible.
            content_length = resp.headers.get("Content-Length")
            total = int(content_length) + existing if content_length is not None else None

            mode = "ab" if existing > 0 else "wb"
            downloaded = existing
            last_print = -1

            with open(tmp_path, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Progress
                    if total:
                        pct = int(downloaded * 100 / total)
                        if pct != last_print:
                            last_print = pct
                            bar_len = 30
                            filled = int(bar_len * pct / 100)
                            bar = "#" * filled + "-" * (bar_len - filled)
                            sys.stdout.write(
                                f"\r[{bar}] {pct:3d}%  "
                                f"{_format_bytes(downloaded)} / {_format_bytes(total)}"
                            )
                            sys.stdout.flush()
                    else:
                        # Unknown total
                        sys.stdout.write(f"\rDownloaded {_format_bytes(downloaded)}")
                        sys.stdout.flush()

        # Finish
        sys.stdout.write("\n")
        os.replace(tmp_path, out_path)
        print(f"Saved to: {out_path}")

    except urllib.error.HTTPError as e:
        # If server doesn't support Range, retry from scratch
        if existing > 0 and e.code in (416, 400, 403, 404):
            print("Resume not supported or range request failed; restarting download...")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return download(url, out_path, chunk_size=chunk_size)
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL, help="URL to download")
    ap.add_argument(
        "--out",
        default=None,
        help="Output file path (default: basename from URL in current dir)",
    )
    ap.add_argument(
        "--chunk-mb",
        type=int,
        default=4,
        help="Chunk size in MB (default: 4)",
    )
    args = ap.parse_args()

    out_path = args.out
    if out_path is None:
        out_path = os.path.basename(args.url.split("?")[0]) or "download.bin"

    download(args.url, out_path, chunk_size=max(1, args.chunk_mb) * 1024 * 1024)


if __name__ == "__main__":
    main()

# python download.py --url https://storage.googleapis.com/grover-models/realnews.tar.gz --out realnews.tar.gz
