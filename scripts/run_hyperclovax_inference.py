import argparse
import shutil
import sys
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hyperclovax_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_NAME,
    create_client,
    request_summary,
)
from src.qwen_omni_dataset import load_jsonl


def stage_file(src: str, dst_dir: Path) -> Path:
    source = Path(src)
    if not source.exists():
        raise FileNotFoundError(source)
    dst_dir.mkdir(parents=True, exist_ok=True)
    staged = dst_dir / f"{uuid.uuid4().hex}{source.suffix.lower()}"
    shutil.copy2(source, staged)
    return staged


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HyperCLOVA X Omni inference through an OpenAI-compatible OmniServe endpoint.")
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--api_key", type=str, default="not-needed")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--public_base_url", type=str, required=True, help="Public base URL that can serve staged media files.")
    parser.add_argument("--stage_dir", type=str, default="out/hyperclovax_infer_cache")
    parser.add_argument("--image", type=str, action="append", default=[])
    parser.add_argument("--audio", type=str, default="")
    parser.add_argument("--script", type=str, default="")
    parser.add_argument("--sample_jsonl", type=str, default="")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=96)
    args = parser.parse_args()

    if args.sample_jsonl:
        samples = load_jsonl(args.sample_jsonl)
        sample = samples[args.sample_index]
        args.image = list(sample.get("image_paths", []))
        args.audio = str(sample.get("audio_path", ""))
        args.script = str(sample.get("script", ""))

    if not args.image:
        raise RuntimeError("Need at least one image path.")
    if not args.script:
        raise RuntimeError("Need script text.")

    stage_dir = Path(args.stage_dir)
    image_path = stage_file(args.image[0], stage_dir)
    image_url = f"{args.public_base_url.rstrip('/')}/{image_path.name}"

    audio_url = ""
    if args.audio:
        staged_audio = stage_file(args.audio, stage_dir)
        audio_url = f"{args.public_base_url.rstrip('/')}/{staged_audio.name}"

    client = create_client(args.base_url, args.api_key)
    print(
        request_summary(
            client=client,
            model_name=args.model_name,
            image_url=image_url,
            audio_url=audio_url,
            script=args.script,
            max_tokens=args.max_tokens,
        )
    )


if __name__ == "__main__":
    main()
