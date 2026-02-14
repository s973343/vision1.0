import argparse
import hashlib
import json
import os
import sys
import time
from urllib.parse import urlparse
from urllib.request import urlopen

import config
import main
import audio_processing
import video_processing


VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".webm", ".mov", ".m4v"}
VIDEO_KEYS = (
    "video",
    "video_path",
    "video_id",
    "video_file",
    "video_name",
    "clip_path",
    "clip_name",
    "file",
    "path",
    "filepath",
    "video_url",
    "url",
    "s3_url",
)
QUESTION_KEYS = (
    "question",
    "query",
    "prompt",
    "instruction",
    "caption",
    "description",
)
OPTION_KEYS = (
    "options",
    "candidates",
    "choices",
    "answers",
)
TASK_KEYS = (
    "task",
    "task_name",
    "subset",
    "category",
    "type",
    "dataset",
    "source",
)


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


def iter_json_items(json_path):
    ext = os.path.splitext(json_path)[1].lower()
    if ext == ".jsonl":
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for item in data:
            yield item
        return
    if isinstance(data, dict):
        for key in ("data", "annotations", "items", "examples"):
            val = data.get(key)
            if isinstance(val, list):
                for item in val:
                    yield item
                return


def _normalize_raw_value(raw):
    if isinstance(raw, dict):
        for key in ("path", "file", "filename", "name", "url"):
            value = raw.get(key)
            if value:
                raw = value
                break
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    return raw or None


def extract_video_raw(entry):
    for key in VIDEO_KEYS:
        if key in entry and entry[key]:
            raw = _normalize_raw_value(entry[key])
            if raw:
                return raw
    return None


def resolve_video_path(entry, video_dir, index):
    raw = extract_video_raw(entry)
    if not raw:
        return None

    parsed = urlparse(raw)
    if parsed.scheme in ("http", "https", "s3", "gs"):
        raw = os.path.basename(parsed.path)

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


def _stringify_option(option):
    if isinstance(option, dict):
        for key in ("text", "label", "option", "answer"):
            value = option.get(key)
            if value:
                return str(value)
        return str(option)
    return str(option)


def build_query_text(entry):
    question = ""
    for key in QUESTION_KEYS:
        value = entry.get(key)
        if value:
            question = str(value).strip()
            if question:
                break

    candidates = None
    for key in OPTION_KEYS:
        if key in entry and entry[key]:
            candidates = entry[key]
            break

    if isinstance(candidates, list) and candidates:
        opts = "; ".join(_stringify_option(x) for x in candidates)
        if question:
            return f"{question} Options: {opts}"
        return f"Options: {opts}"

    return question


def _write_bytes_to_cache(blob, cache_dir, suffix=".mp4"):
    os.makedirs(cache_dir, exist_ok=True)
    digest = hashlib.md5(blob).hexdigest()
    path = os.path.join(cache_dir, f"{digest}{suffix}")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(blob)
    return path


def _download_url_to_cache(url, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    parsed = urlparse(url)
    name = os.path.basename(parsed.path) or hashlib.md5(url.encode("utf-8")).hexdigest()
    path = os.path.join(cache_dir, name)
    if os.path.exists(path):
        return path
    with urlopen(url) as resp, open(path, "wb") as f:
        f.write(resp.read())
    return path


def resolve_streaming_video(entry, video_dir, cache_dir, download_urls):
    raw = extract_video_raw(entry)
    if raw is None:
        return None

    if isinstance(raw, dict):
        bytes_blob = raw.get("bytes")
        if bytes_blob:
            return _write_bytes_to_cache(bytes_blob, cache_dir)

        path = raw.get("path")
        if path:
            if os.path.isabs(path) and os.path.exists(path):
                return path
            if video_dir:
                candidate = os.path.join(video_dir, path)
                if os.path.exists(candidate):
                    return candidate
            parsed = urlparse(path)
            if download_urls and parsed.scheme in ("http", "https"):
                return _download_url_to_cache(path, cache_dir)
        return None

    if isinstance(raw, str):
        parsed = urlparse(raw)
        if parsed.scheme in ("http", "https"):
            if download_urls:
                return _download_url_to_cache(raw, cache_dir)
            return None
        if os.path.isabs(raw) and os.path.exists(raw):
            return raw
        if video_dir:
            candidate = os.path.join(video_dir, raw)
            if os.path.exists(candidate):
                return candidate
    return None


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


def resolve_default_video_dir(root_dir):
    for name in ("video", "videos", "clips", "media"):
        candidate = os.path.join(root_dir, name)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(root_dir, "video")


def run_batch(root_dir, json_dir, video_dir, frames_root, audio_root, limit, task):
    if not os.path.isdir(json_dir):
        print(f"Error: JSON dir not found: {json_dir}")
        return 1
    if not os.path.isdir(video_dir):
        print(f"Error: Video dir not found: {video_dir}")
        return 1

    print(f"Indexing videos in: {video_dir}")
    index, duplicates = build_video_index(video_dir)
    if duplicates:
        print(f"Warning: {len(duplicates)} duplicate video keys found. Using first match.")

    json_files = []
    for name in os.listdir(json_dir):
        lower = name.lower()
        if not (lower.endswith(".json") or lower.endswith(".jsonl")):
            continue
        if task and task != name and task != os.path.splitext(name)[0]:
            continue
        json_files.append(os.path.join(json_dir, name))
    json_files.sort()

    if not json_files:
        print("No JSON/JSONL files found to process.")
        return 1

    total = 0
    missing = 0
    failures = 0

    for json_path in json_files:
        task_name = os.path.splitext(os.path.basename(json_path))[0]
        print(f"\n=== Task: {task_name} ===")

        for entry in iter_json_items(json_path):
            if limit and total >= limit:
                print(f"Reached limit {limit}, stopping.")
                print(f"Done. processed={total} missing={missing} failures={failures}")
                return 0

            if not isinstance(entry, dict):
                continue

            video_path = resolve_video_path(entry, video_dir, index)
            if not video_path:
                missing += 1
                continue

            query_text = build_query_text(entry)
            video_stem = os.path.splitext(os.path.basename(video_path))[0]
            frame_dir = os.path.join(frames_root, task_name, video_stem)
            audio_out = os.path.join(audio_root, f"{task_name}_{video_stem}.mp3")
            os.makedirs(frame_dir, exist_ok=True)
            os.makedirs(audio_root, exist_ok=True)

            set_pipeline_context(video_path, query_text, frame_dir, audio_out)

            total += 1
            print(f"\n[{total}] {task_name} | {video_stem}")
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


def infer_task_name(entry, default="mvlu"):
    for key in TASK_KEYS:
        value = entry.get(key)
        if value:
            return str(value).strip()
    return default


def iter_hf_stream(dataset_name, split, token, config_name):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(f"datasets not available: {e}")

    kwargs = {"split": split, "streaming": True}
    if token:
        kwargs["token"] = token
    if config_name:
        kwargs["name"] = config_name

    dataset = load_dataset(dataset_name, **kwargs)
    for item in dataset:
        yield item


def run_streaming(
    dataset_name,
    split,
    token,
    config_name,
    video_dir,
    frames_root,
    audio_root,
    cache_dir,
    limit,
    download_urls,
    task_filter,
):
    total = 0
    missing = 0
    failures = 0

    for entry in iter_hf_stream(dataset_name, split, token, config_name):
        if limit and total >= limit:
            print(f"Reached limit {limit}, stopping.")
            print(f"Done. processed={total} missing={missing} failures={failures}")
            return 0

        if not isinstance(entry, dict):
            continue

        query_text = build_query_text(entry)
        task_name = infer_task_name(entry)
        if task_filter and task_name != task_filter:
            continue

        video_path = resolve_streaming_video(entry, video_dir, cache_dir, download_urls)
        if not video_path:
            missing += 1
            continue

        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        frame_dir = os.path.join(frames_root, task_name, video_stem)
        audio_out = os.path.join(audio_root, f"{task_name}_{video_stem}.mp3")
        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(audio_root, exist_ok=True)

        set_pipeline_context(video_path, query_text, frame_dir, audio_out)

        total += 1
        print(f"\n[{total}] {task_name} | {video_stem}")
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
    parser = argparse.ArgumentParser(description="Batch MVLU ingestion runner")
    parser.add_argument("--root", default="data/MVLU", help="MVLU root folder")
    parser.add_argument("--json-dir", default=None, help="Override JSON dir")
    parser.add_argument("--video-dir", default=None, help="Override video dir")
    parser.add_argument("--frames-root", default="data/frames/mvlu", help="Root for extracted frames")
    parser.add_argument("--audio-root", default="data/audio/mvlu", help="Root for extracted audio")
    parser.add_argument("--limit", type=int, default=0, help="Max total items to process (0 = no limit)")
    parser.add_argument("--task", default="", help="Process only one JSON task (name or filename)")
    parser.add_argument("--hf", action="store_true", help="Read from Hugging Face streaming dataset")
    parser.add_argument("--hf-dataset", default="MLVU/MVLU", help="HF dataset name")
    parser.add_argument("--hf-config", default="", help="HF config name (optional)")
    parser.add_argument("--hf-split", default="train", help="HF split name")
    parser.add_argument("--hf-token-env", default="HF_TOKEN", help="Env var containing HF token")
    parser.add_argument("--cache-dir", default="data/cache/mvlu", help="Cache for streamed videos")
    parser.add_argument("--download-urls", action="store_true", help="Download video URLs to cache")
    parser.add_argument("--hf-task", default="", help="Filter HF stream by task name")
    return parser.parse_args()


def main_cli():
    args = parse_args()
    if args.hf:
        token = os.getenv(args.hf_token_env)
        return run_streaming(
            args.hf_dataset,
            args.hf_split,
            token,
            args.hf_config or None,
            args.video_dir,
            args.frames_root,
            args.audio_root,
            args.cache_dir,
            args.limit,
            args.download_urls,
            args.hf_task,
        )

    root_dir = args.root
    json_dir = args.json_dir or root_dir
    video_dir = args.video_dir or resolve_default_video_dir(root_dir)
    return run_batch(root_dir, json_dir, video_dir, args.frames_root, args.audio_root, args.limit, args.task)


if __name__ == "__main__":
    sys.exit(main_cli())
