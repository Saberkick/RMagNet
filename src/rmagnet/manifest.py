"""Validate scene-separated JSONL metadata without loading image files."""

import argparse
import json
from pathlib import Path


REQUIRED = {"scene_id", "capture_id", "split", "input", "gt"}
SPLITS = {"train", "val", "test"}


def validate_manifest(path: Path) -> int:
    scenes: dict[str, str] = {}
    identities: set[tuple[str, str, str]] = set()
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            record = json.loads(raw)
            missing = REQUIRED - record.keys()
            if missing:
                raise ValueError(f"line {line_number}: missing {sorted(missing)}")
            if record["split"] not in SPLITS:
                raise ValueError(f"line {line_number}: invalid split")
            scene = str(record["scene_id"])
            split = str(record["split"])
            if scene in scenes and scenes[scene] != split:
                raise ValueError(f"line {line_number}: scene crosses data splits")
            scenes[scene] = split
            identity = (scene, str(record["capture_id"]), str(record["input"]))
            if identity in identities:
                raise ValueError(f"line {line_number}: duplicate input entry")
            identities.add(identity)
            count += 1
    if count == 0:
        raise ValueError("Manifest is empty")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    count = validate_manifest(args.manifest)
    print(f"manifest_ok records={count}")


if __name__ == "__main__":
    main()

