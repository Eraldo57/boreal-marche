"""Restore a checksummed, private runtime bundle into a Cloud Build workspace."""

import hashlib
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def main() -> None:
    uri, expected = sys.argv[1:]
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("Missing runtime bundle checksum")
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "runtime.tar.gz"
        subprocess.run(["gcloud", "storage", "cp", uri, str(archive)], check=True)
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("Runtime bundle checksum mismatch")
        with tarfile.open(archive) as bundle:
            bundle.extractall(Path.cwd(), filter="data")


if __name__ == "__main__":
    main()
