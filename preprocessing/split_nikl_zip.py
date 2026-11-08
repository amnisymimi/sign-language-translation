#!/usr/bin/env python3
"""Split raw NIKL ZIP archives by signer, using only Python's standard library.

Examples (quote paths with spaces):
  python split_nikl_zip.py --zip "sp_10.zip" --output nikl_split
  python split_nikl_zip.py --zip "sp_10.zip" "sp_11.zip" --output nikl_split
  python split_nikl_zip.py --zip "sp_10.zip" --ratios 0.8 0.1 0.1 --seed 42

Reads JSON annotations directly from ZIPs; videos stay in the original ZIPs.
Copies only selected original JSON bytes to output/annotations/. No landmark
extraction, normalization, or guessed landmark index mapping is performed.
Manifest rows refer to RAW annotations, not the processed-feature format used
by week1_sign_translation.py. Convert landmarks before training that model.
Validation selects settings/checkpoints; test is for final evaluation only.
Do not resplit once experiments begin. Keep the generated manifests fixed.

At least three distinct signers are required. Signer-group assignment targets
sample ratios approximately; unequal signer sizes prevent exact percentages.
Different signers can repeat the same sentence: text overlap is reported and
does not invalidate signer-independent evaluation, but limits claims about
generalizing to unseen sentences. No sentence-disjoint split is promised.
Keep source ZIPs available if you need to retrieve videos using manifest refs.
The output directory must not already exist, to avoid overwriting a split.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata
import zipfile


SPLITS = ("train", "val", "test")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(value):
    return " ".join(unicodedata.normalize("NFC", value).split())


def read_inventory(zip_paths, max_json_mb, max_total_json_mb):
    rows, skipped = [], []
    seen = set()
    total_bytes = 0
    for archive in zip_paths:
        with zipfile.ZipFile(archive) as z:
            members = z.infolist()
            names = [m.filename for m in members]
            if len(names) != len(set(names)):
                raise ValueError(f"ZIP contains duplicate member paths: {archive}")
            videos = defaultdict(list)
            for member in members:
                if member.filename.lower().endswith(".mp4"):
                    videos[Path(member.filename).stem].append(member.filename)
            for member in members:
                if member.is_dir() or not member.filename.lower().endswith(".json"):
                    continue
                if member.file_size > max_json_mb * 1024**2:
                    raise ValueError(f"JSON exceeds size limit: {member.filename}")
                total_bytes += member.file_size
                if total_bytes > max_total_json_mb * 1024**2:
                    raise ValueError("Total annotation bytes exceed configured limit")
                try:
                    raw = z.read(member)
                    data = json.loads(raw.decode("utf-8-sig"))
                except (UnicodeError, json.JSONDecodeError) as e:
                    raise ValueError(f"Invalid JSON: {archive.name}/{member.filename}") from e
                if not isinstance(data, dict) or not all(
                    key in data for key in ("id", "krlgg_sntenc", "potogrf", "landmarks")
                ):
                    skipped.append(dict(zip=str(archive), member=member.filename,
                                        reason="Not recognized as a NIKL landmark sample"))
                    continue
                sid = str(data["id"]).strip()
                signer = str(data["potogrf"].get("sl_speaker_id") or "").strip()
                text = data["krlgg_sntenc"].get("koreanText")
                frames = data["landmarks"]
                if not sid or not signer or not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Missing sample ID, signer ID, or text: {member.filename}")
                if not isinstance(frames, list) or not frames:
                    raise ValueError(f"Empty/invalid landmarks: {member.filename}")
                if sid in seen:
                    raise ValueError(f"Duplicate sample ID {sid}; remove overlapping archives")
                seen.add(sid)
                stem = str(data.get("vido_file_nm") or sid)
                candidates = videos.get(stem, [])
                # An annotation may arrive separately from its video. It can still
                # support landmark-based training; record absent/ambiguous video.
                fps = float(data["potogrf"].get("fps", 0))
                if not 0 < fps < 1000:
                    raise ValueError(f"Invalid FPS in {member.filename}")
                rows.append(dict(sample_id=sid, signer_id=signer,
                                 text=normalize_text(text), num_frames=len(frames), fps=fps,
                                 source_zip=str(archive), annotation_member=member.filename,
                                 annotation_sha256=hashlib.sha256(raw).hexdigest(),
                                 video_member=candidates[0] if len(candidates) == 1 else None,
                                 video_match_count=len(candidates)))
    if not rows:
        raise ValueError("No NIKL landmark samples found. Check archive contents (nested ZIPs are not scanned).")
    return sorted(rows, key=lambda r: r["sample_id"]), skipped


def assign_signers(rows, ratios, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["signer_id"]].append(row)
    if len(groups) < 3:
        raise ValueError("At least 3 signers are needed for nonempty signer-independent train/val/test splits")
    # Sort before shuffling, so filesystem/ZIP entry order cannot affect results.
    signers = sorted(groups)
    random.Random(seed).shuffle(signers)
    signers.sort(key=lambda s: len(groups[s]), reverse=True)
    targets = {name: len(rows) * ratio for name, ratio in zip(SPLITS, ratios)}
    counts = dict.fromkeys(SPLITS, 0)
    assignment = {}
    # Reserve one signer for each split, with the largest assigned to the largest
    # requested split. All later assignments reduce remaining sample deficits.
    initial = sorted(SPLITS, key=lambda name: targets[name], reverse=True)
    for index, signer in enumerate(signers):
        split = initial[index] if index < 3 else max(SPLITS, key=lambda n: targets[n] - counts[n])
        assignment[signer] = split
        counts[split] += len(groups[signer])
    return {name: [dict(row, split=name) for row in rows if assignment[row["signer_id"]] == name]
            for name in SPLITS}


def summarize(rows, splits, ratios, seed, skipped):
    report = dict(total_samples=len(rows), total_signers=len({r["signer_id"] for r in rows}),
                  seed=seed, requested_ratios=dict(zip(SPLITS, ratios)),
                  signer_inventory=dict(sorted(Counter(r["signer_id"] for r in rows).items())),
                  skipped_json=skipped,
                  video_missing_or_ambiguous=sum(r["video_match_count"] != 1 for r in rows),
                  splits={}, signer_overlap={}, text_overlap={}, warnings=[])
    for name, subset in splits.items():
        report["splits"][name] = dict(samples=len(subset), fraction=len(subset)/len(rows),
                                           signers=sorted({r["signer_id"] for r in subset}))
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        key = f"{a}_{b}"
        overlap = {r["signer_id"] for r in splits[a]} & {r["signer_id"] for r in splits[b]}
        assert not overlap, "Internal error: signer leakage"
        report["signer_overlap"][key] = sorted(overlap)
        ta = {r["text"] for r in splits[a]}
        tb = {r["text"] for r in splits[b]}
        common = ta & tb
        report["text_overlap"][key] = dict(unique_shared_sentences=len(common),
                                               examples=sorted(common)[:5])
    if report["total_signers"] < 10:
        report["warnings"].append("Few signers: results may vary substantially by held-out signer; report this limitation.")
    if any(v["unique_shared_sentences"] for v in report["text_overlap"].values()):
        report["warnings"].append("Some target sentences repeat across splits. This is a signer-independent split, not an unseen-sentence evaluation.")
    if skipped:
        report["warnings"].append("Some JSONs were unrecognized and skipped; review skipped_json before training.")
    return report


def copy_annotations(splits, output):
    # Never extract ZIP-provided paths: generated hash filenames prevent path
    # traversal and support Unicode sample IDs without unsafe filenames.
    by_archive = defaultdict(list)
    for subset in splits.values():
        for row in subset:
            by_archive[row["source_zip"]].append(row)
    folder = output / "annotations"
    folder.mkdir()
    for archive, rows in by_archive.items():
        with zipfile.ZipFile(archive) as z:
            for row in rows:
                raw = z.read(row["annotation_member"])
                if hashlib.sha256(raw).hexdigest() != row["annotation_sha256"]:
                    raise ValueError("Archive changed since inventory; rerun with a fresh output directory")
                filename = hashlib.sha256(row["sample_id"].encode()).hexdigest() + ".json"
                (folder / filename).write_bytes(raw)
                row["annotation_path"] = "annotations/" + filename


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", nargs="+", required=True, dest="archives")
    p.add_argument("--output", default="nikl_split")
    p.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-json-mb", type=float, default=100)
    p.add_argument("--max-total-json-mb", type=float, default=8192)
    args = p.parse_args()
    if any(not 0 < r < 1 for r in args.ratios) or abs(sum(args.ratios)-1) > 1e-6:
        p.error("Ratios must be positive fractions summing to 1")
    if args.max_json_mb <= 0 or args.max_total_json_mb <= 0:
        p.error("JSON limits must be positive")
    archives = sorted({Path(a).expanduser().resolve() for a in args.archives})
    output = Path(args.output)
    if output.exists():
        p.error("Output directory already exists; choose a new name to protect fixed splits")
    for archive in archives:
        if not archive.is_file() or not zipfile.is_zipfile(archive):
            p.error(f"Not a readable ZIP: {archive}")
    rows, skipped = read_inventory(archives, args.max_json_mb, args.max_total_json_mb)
    counts = dict(sorted(Counter(r["signer_id"] for r in rows).items()))
    print(f"Found {len(rows)} samples and {len(counts)} signers", flush=True)
    print("Samples per signer:", counts, flush=True)
    output.mkdir(parents=True)
    if len(counts) < 3:
        write_json(output / "inventory.json", dict(signer_inventory=counts, samples=rows, skipped_json=skipped))
        p.exit(2, "Cannot make three signer-independent splits. Inventory saved; add archives containing other signers and rerun into a new folder.\n")
    splits = assign_signers(rows, args.ratios, args.seed)
    copy_annotations(splits, output)
    for name, subset in splits.items():
        with (output / f"{name}_raw.jsonl").open("w", encoding="utf-8") as stream:
            for row in subset:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = summarize(rows, splits, args.ratios, args.seed, skipped)
    write_json(output / "split_report.json", report)
    for name, result in report["splits"].items():
        print(f"{name}: {result['samples']} samples ({result['fraction']:.1%}); signers: {result['signers']}")
    print("PASS: no signer overlap across train/val/test")
    for warning in report["warnings"]:
        print("NOTE:", warning)
    print("Saved raw JSON manifests and annotations to", output.resolve())
    print("Next: preprocess verified landmarks within these fixed splits; fit tokenizer on train text only.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, zipfile.BadZipFile, KeyError, TypeError) as error:
        raise SystemExit(f"ERROR: {error}")
