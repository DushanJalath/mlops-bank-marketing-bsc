"""Download the UCI Bank Marketing data and extract bank-additional-full.csv.

The UCI archive contains zip files inside a zip file, so we search recursively.
"""

import io
import urllib.request
import zipfile

from src.config import ROOT, load_params

TARGET = "bank-additional-full.csv"


def find_csv(archive: zipfile.ZipFile) -> bytes:
    for name in archive.namelist():
        if name.endswith(TARGET) and "__MACOSX" not in name:
            return archive.read(name)
    for name in archive.namelist():
        if name.endswith(".zip"):
            inner = zipfile.ZipFile(io.BytesIO(archive.read(name)))
            try:
                return find_csv(inner)
            except FileNotFoundError:
                continue
    raise FileNotFoundError(f"{TARGET} not found in archive")


def main() -> None:
    params = load_params()["data"]
    out = ROOT / params["raw"]
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {params['url']}")
    with urllib.request.urlopen(params["url"], timeout=120) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    out.write_bytes(find_csv(archive))
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
