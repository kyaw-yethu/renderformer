#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import objaverse.xl as oxl


def parse_args():
    p = argparse.ArgumentParser(description="Download N OBJ repositories from Objaverse-XL only.")
    p.add_argument("--n", type=int, default=200, help="How many OBJ repositories to download.")
    p.add_argument("--out", type=str, default="./objaverse_objs", help="Where to save downloaded files.")
    p.add_argument("--cache", type=str, default=str(Path.home() / ".objaverse"),
                   help="Where to cache annotations/metadata (used by get_annotations).")
    p.add_argument("--processes", type=int, default=8, help="Parallel processes for download_objects.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    p.add_argument("--oversample", type=int, default=2,
                   help="Try oversample * remaining each round to compensate failures.")
    p.add_argument("--max_rounds", type=int, default=50, help="Safety cap for loop rounds.")
    p.add_argument(
        "--save-format",
        choices=["files", "zip", "tar", "tar.gz"],
        default="files",
        help="How to store downloaded repos locally (files keeps raw directory tree).",
    )
    p.add_argument(
        "--flat-dir",
        type=str,
        default="objs_flat",
        help="Name of flat directory (inside --out) that will contain only .obj files.",
    )
    return p.parse_args()


def flatten_and_cleanup(root: Path, objs_dir_name: str = "objs_flat") -> None:
    root = root.resolve()
    objs_dir = (root / objs_dir_name).resolve()
    objs_dir.mkdir(exist_ok=True, parents=True)

    moved = 0
    collisions = 0

    for obj_path in root.rglob("*.obj"):
        if not obj_path.exists():
            continue
        try:
            obj_path.relative_to(objs_dir)
            continue
        except ValueError:
            pass

        base = obj_path.name
        target = objs_dir / base

        if target.exists():
            collisions += 1
            stem, suffix = obj_path.stem, obj_path.suffix
            i = 1
            while True:
                cand = objs_dir / f"{stem}_{i}{suffix}"
                if not cand.exists():
                    target = cand
                    break
                i += 1

        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(obj_path), str(target))
            moved += 1
        except FileNotFoundError:
            continue

    print(f"[cleanup] Moved {moved} .obj files into {objs_dir}")
    print(f"[cleanup] Filename collisions handled: {collisions}")

    for path in root.iterdir():
        if path.resolve() == objs_dir:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    print(f"[cleanup] Removed all non-.obj files/dirs under {root}")


def main():
    args = parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading annotations (cache={cache_dir}) ...")
    ann = oxl.get_annotations(download_dir=str(cache_dir))

    if "fileType" not in ann.columns:
        raise RuntimeError(f"Expected column 'fileType' in annotations, got: {list(ann.columns)[:30]} ...")

    ann_obj = ann[ann["fileType"] == "obj"].reset_index(drop=True)
    if len(ann_obj) == 0:
        raise RuntimeError("No rows with fileType == 'obj' found in annotations.")

    ann_obj = ann_obj.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    downloaded = {}
    tried_file_ids = set()
    cursor = 0

    print(f"[2/3] OBJ candidates: {len(ann_obj)}. Target downloads: {args.n}")

    def get_dir_size_bytes(root: Path) -> int:
        total = 0
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                try:
                    fp = Path(dirpath) / name
                    total += fp.stat().st_size
                except OSError:
                    continue
        return total

    for round_idx in range(args.max_rounds):
        remaining = args.n - len(downloaded)
        if remaining <= 0:
            break

        batch_size = min(len(ann_obj) - cursor, max(remaining * args.oversample, remaining))
        if batch_size <= 0:
            break

        batch = ann_obj.iloc[cursor:cursor + batch_size].copy()
        cursor += batch_size

        if "fileIdentifier" in batch.columns:
            batch = batch[~batch["fileIdentifier"].isin(tried_file_ids)]

        if len(batch) == 0:
            continue

        if "fileIdentifier" in batch.columns:
            tried_file_ids.update(batch["fileIdentifier"].tolist())

        print(f"  - Round {round_idx+1}: trying {len(batch)} candidates (need {remaining})")

        try:
            paths = oxl.download_objects(
                objects=batch,
                download_dir=str(out_dir),
                processes=args.processes,
                save_repo_format=args.save_format,
            )
        except FileNotFoundError as exc:
            print(f"    ! Skipping batch due to missing file inside repo: {exc}")
            continue

        for fid, pth in (paths or {}).items():
            if not pth:
                continue
            candidate = Path(pth)
            if not candidate.is_absolute():
                candidate = (out_dir / candidate).resolve()
            if candidate.suffix.lower() == ".obj" and candidate.exists():
                downloaded.setdefault(fid, str(candidate))

        print(f"    -> downloaded so far: {len(downloaded)}/{args.n}")

    print(f"[3/3] Done. Final: {len(downloaded)}/{args.n}")

    manifest_path = out_dir / "manifest_obj.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(downloaded, f, ensure_ascii=False, indent=2)

    print(f"Manifest saved to: {manifest_path}")

    flatten_and_cleanup(out_dir, objs_dir_name=args.flat_dir)

if __name__ == "__main__":
    main()
