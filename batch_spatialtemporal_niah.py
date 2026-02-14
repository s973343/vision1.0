import argparse
import json
import os
import sys
import time
import uuid

import config
import main
import audio_processing
import video_processing
from datasets import load_dataset


VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".webm", ".mov", ".m4v"}


def build_video_index(video_dir):
    index = {}
    duplicates = {}
    for root, _, files in os.walk(video_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in VIDEO_EXTS:
                continue
            path = os.path.join(root, name)
            for key in (name, os.path.splitext(name)[0]):
                if key in index:
                    duplicates.setdefault(key, []).append(path)
                else:
                    index[key] = path
    return index, duplicates


def load_json_items(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "annotations", "items", "examples"):
            val = data.get(key)
            if isinstance(val, list):
                return val
    return []


def resolve_video_path(entry, video_dir, index):
    raw = entry.get("image") or entry.get("video") or entry.get("video_path")
    if not raw:
        return None
    raw = os.path.normpath(raw)
    if os.path.isabs(raw) and os.path.exists(raw):
        return raw
    candidate = os.path.join(video_dir, raw)
    if os.path.exists(candidate):
        return candidate
    base = os.path.basename(raw)
    if base in index:
        return index[base]
    stem = os.path.splitext(base)[0]
    return index.get(stem)


def normalize_video_key(value):
    if not value:
        return None
    if isinstance(value, str):
        base = os.path.basename(value)
        stem = os.path.splitext(base)[0]
        return base, stem
    if isinstance(value, dict):
        path = value.get("path") or value.get("url") or value.get("filename")
        if path:
            base = os.path.basename(path)
            stem = os.path.splitext(base)[0]
            return base, stem
    return None


def build_hf_stream_index(dataset_name, split, video_field, id_field, needed_keys):
    if not needed_keys:
        return {}
    ds = load_dataset(dataset_name, split=split, streaming=True)
    matches = {}
    remaining = set(needed_keys)
    for row in ds:
        raw_id = row.get(id_field) if id_field else None
        keys = set()
        if raw_id:
            if isinstance(raw_id, str):
                keys.add(raw_id)
                keys.add(os.path.splitext(os.path.basename(raw_id))[0])
        video_val = row.get(video_field)
        norm = normalize_video_key(video_val)
        if norm:
            keys.update(norm)
        hit = remaining.intersection(keys)
        if hit:
            for k in hit:
                matches[k] = video_val
            remaining -= hit
            if not remaining:
                break
    return matches


def materialize_hf_video(video_val, out_dir, name_hint):
    os.makedirs(out_dir, exist_ok=True)
    ext = ".mp4"
    if isinstance(video_val, dict):
        path = video_val.get("path")
        if path and os.path.exists(path):
            return path
        if video_val.get("url"):
            base = os.path.basename(video_val["url"])
            ext = os.path.splitext(base)[1] or ext
        if video_val.get("path"):
            ext = os.path.splitext(os.path.basename(video_val["path"]))[1] or ext
        data = video_val.get("bytes")
        if data:
            fname = f"{name_hint}_{uuid.uuid4().hex}{ext}"
            out_path = os.path.join(out_dir, fname)
            with open(out_path, "wb") as f:
                f.write(data)
            return out_path
    if isinstance(video_val, str):
        if os.path.exists(video_val):
            return video_val
        ext = os.path.splitext(os.path.basename(video_val))[1] or ext
    return None


def extract_query_text(entry):
    conversations = entry.get("conversations") or []
    if isinstance(conversations, list):
        for item in conversations:
            if not isinstance(item, dict):
                continue
            if item.get("from") == "human" and item.get("value"):
                return str(item["value"]).strip()
    return ""


def set_pipeline_context(video_path, query_text, frame_dir, audio_out):
    # Update config
    config.VIDEO_INPUT = video_path
    config.USER_DESCRIPTION = query_text or config.USER_DESCRIPTION
    config.FRAME_DIR = frame_dir
    config.AUDIO_OUTPUT = audio_out

    # Update imported module globals (they use "from config import ...")
    main.VIDEO_INPUT = video_path
    main.USER_DESCRIPTION = query_text or main.USER_DESCRIPTION
    audio_processing.VIDEO_INPUT = video_path
    audio_processing.AUDIO_OUTPUT = audio_out
    video_processing.VIDEO_INPUT = video_path
    video_processing.FRAME_DIR = frame_dir


def run_batch(json_path, video_dir, frames_root, audio_root, limit, hf_dataset, hf_split, hf_video_field, hf_id_field, hf_tmp_dir):
    if not os.path.isfile(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return 1
    if video_dir and not os.path.isdir(video_dir):
        print(f"Error: Video dir not found: {video_dir}")
        return 1

    items = load_json_items(json_path)
    if not items:
        print("No items found in JSON.")
        return 1

    index = {}
    duplicates = {}
    if video_dir:
        print(f"Indexing videos in: {video_dir}")
        index, duplicates = build_video_index(video_dir)
        if duplicates:
            print(f"Warning: {len(duplicates)} duplicate video keys found. Using first match.")

    hf_index = {}
    if hf_dataset:
        needed = set()
        for entry in items:
            raw = entry.get("image") or entry.get("video") or entry.get("video_path")
            if not raw:
                continue
            base = os.path.basename(raw)
            needed.add(base)
            needed.add(os.path.splitext(base)[0])
        print(f"Streaming HF dataset: {hf_dataset} ({hf_split})")
        hf_index = build_hf_stream_index(hf_dataset, hf_split, hf_video_field, hf_id_field, needed)

    total = 0
    missing = 0
    failures = 0
    json_stem = os.path.splitext(os.path.basename(json_path))[0]

    for entry in items:
        if limit and total >= limit:
            print(f"Reached limit {limit}, stopping.")
            break

        video_path = None
        if hf_index:
            raw = entry.get("image") or entry.get("video") or entry.get("video_path")
            if raw:
                base = os.path.basename(raw)
                stem = os.path.splitext(base)[0]
                video_val = hf_index.get(base) or hf_index.get(stem)
                if video_val is not None:
                    video_path = materialize_hf_video(video_val, hf_tmp_dir, stem or base)
        if not video_path:
            video_path = resolve_video_path(entry, video_dir, index)
        if not video_path:
            missing += 1
            continue

        query_text = extract_query_text(entry)
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        frame_dir = os.path.join(frames_root, json_stem, video_stem)
        audio_out = os.path.join(audio_root, f"{json_stem}_{video_stem}.mp3")
        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(audio_root, exist_ok=True)

        set_pipeline_context(video_path, query_text, frame_dir, audio_out)

        total += 1
        print(f"\n[{total}] {video_stem}")
        start = time.time()
        try:
            main.run_ingestion_pipeline()
        except Exception as e:
            failures += 1
            print(f"Error: {e}")
            continue
        finally:
            elapsed = time.time() - start
            print(f"Elapsed: {elapsed:.2f}s")

    print(f"\nDone. processed={total} missing={missing} failures={failures}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Batch SpatialTemporal NIAH ingestion runner")
    parser.add_argument("--json", default="spatialtemporal_niah_qa.json", help="Path to JSON file")
    parser.add_argument("--video-dir", default="data", help="Root folder containing videos (fallback)")
    parser.add_argument("--frames-root", default="data/frames/spatialtemporal_niah", help="Root for extracted frames")
    parser.add_argument("--audio-root", default="data/audio/spatialtemporal_niah", help="Root for extracted audio")
    parser.add_argument("--limit", type=int, default=0, help="Max total items to process (0 = no limit)")
    parser.add_argument("--hf-dataset", default="TIGER-Lab/VISTA-400K", help="HF dataset name (streaming)")
    parser.add_argument("--hf-split", default="train", help="HF dataset split")
    parser.add_argument("--hf-video-field", default="video", help="HF column holding video")
    parser.add_argument("--hf-id-field", default="", help="HF column used as ID (optional)")
    parser.add_argument("--hf-tmp-dir", default="data/tmp_videos/spatialtemporal_niah", help="Temp dir for streamed videos")
    return parser.parse_args()


def main_cli():
    args = parse_args()
    hf_id_field = args.hf_id_field.strip() or None
    return run_batch(
        args.json,
        args.video_dir,
        args.frames_root,
        args.audio_root,
        args.limit,
        args.hf_dataset,
        args.hf_split,
        args.hf_video_field,
        hf_id_field,
        args.hf_tmp_dir,
    )


if __name__ == "__main__":
    sys.exit(main_cli())
