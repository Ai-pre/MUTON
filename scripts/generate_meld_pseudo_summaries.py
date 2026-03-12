import argparse
import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Iterable

import cv2
import torch
from openai import OpenAI
from PIL import Image


MELD_ID_RE = re.compile(r"meld_d(?P<dialogue>\d+)_u(?P<utterance>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pseudo summary targets for MELD using translated scripts and representative video frames.",
    )
    parser.add_argument("--input_pt", type=str, required=True, help="Source MELD .pt file")
    parser.add_argument("--videos_root", type=str, required=True, help="Directory that contains dia{d}_utt{u}.mp4")
    parser.add_argument("--output_pt", type=str, required=True, help="Destination .pt with pseudo target_text")
    parser.add_argument("--cache_json", type=str, default="", help="Optional cache file to resume generation")
    parser.add_argument("--style_examples_pt", type=str, default="out/fusion_dataset.pt")
    parser.add_argument("--style_examples", type=int, default=4)
    parser.add_argument("--model", type=str, required=True, help="OpenAI-compatible multimodal model name")
    parser.add_argument("--base_url", type=str, default="", help="Optional OpenAI-compatible base URL")
    parser.add_argument("--api_key", type=str, default="", help="Optional API key override")
    parser.add_argument("--num_frames", type=int, default=1, choices=[1, 3])
    parser.add_argument("--media_mode", type=str, default="base64", choices=["base64", "file"])
    parser.add_argument("--frame_cache_dir", type=str, default="out/meld_pseudo_frames")
    parser.add_argument("--max_side", type=int, default=768)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow_text_only", action="store_true")
    parser.add_argument("--use_emotion_label", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_cache(cache_path: str) -> dict:
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def save_cache(cache_path: str, cache: dict) -> None:
    if not cache_path:
        return
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, cache_path)


def parse_meld_video_path(videos_root: Path, sample_id: str) -> Path:
    match = MELD_ID_RE.match(sample_id)
    if not match:
        raise ValueError(f"Unexpected MELD sample id: {sample_id}")
    dialogue_id = int(match.group("dialogue"))
    utterance_id = int(match.group("utterance"))
    return videos_root / f"dia{dialogue_id}_utt{utterance_id}.mp4"


def choose_frame_indices(frame_count: int, num_frames: int) -> list[int]:
    if frame_count <= 1 or num_frames == 1:
        return [max(0, frame_count // 2)]
    return [
        max(0, int(frame_count * 0.2)),
        max(0, int(frame_count * 0.5)),
        max(0, int(frame_count * 0.8)),
    ]


def extract_frames(video_path: Path, num_frames: int) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = choose_frame_indices(frame_count, num_frames)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    capture.release()
    return frames


def resize_image(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, float(max_side) / max(width, height))
    if scale == 1.0:
        return image
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def image_to_data_url(image: Image.Image, jpeg_quality: int) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def save_frame_to_cache(
    image: Image.Image,
    frame_cache_dir: Path,
    sample_id: str,
    frame_index: int,
    jpeg_quality: int,
) -> str:
    frame_cache_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frame_cache_dir / f"{sample_id}_f{frame_index}.jpg"
    image.save(frame_path, format="JPEG", quality=jpeg_quality)
    return frame_path.resolve().as_uri()


def load_style_examples(style_examples_pt: str, limit: int) -> list[str]:
    path = Path(style_examples_pt)
    if not path.exists():
        return []
    dataset = torch.load(path, map_location="cpu")
    seen = set()
    examples = []
    for item in dataset:
        text = str(item.get("target_text", "")).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        examples.append(text)
        if len(examples) >= limit:
            break
    return examples


def build_prompts(style_examples: Iterable[str], transcript: str, emotion: str | None) -> tuple[str, str]:
    style_block = "\n".join(f"- {example}" for example in style_examples) or "- 눈썹을 찌푸리고 낮은 목소리로 불만을 천천히 말함."

    system_prompt = (
        "You are a multimodal annotator that writes short Korean observational summaries for short dialogue clips. "
        "Describe only observable cues from facial expression, tone, and spoken content. "
        "Do not mention model uncertainty, camera, or analysis steps. "
        "Avoid explicit emotion labels such as 화남, 슬픔, 행복, angry, sad, happy. "
        "Write one Korean sentence in the style of an accessible situation description."
    )

    label_hint = ""
    if emotion:
        label_hint = (
            "\n[참고 라벨]\n"
            f"- 이 샘플의 감정 라벨: {emotion}\n"
            "- 라벨은 문장을 베끼는 용도가 아니라, 관찰 가능한 표현을 놓치지 않기 위한 참고용이다.\n"
        )

    user_prompt = (
        "[스타일 예시]\n"
        f"{style_block}\n\n"
        "[현재 대사]\n"
        f"- {transcript}\n"
        f"{label_hint}\n"
        "[작성 규칙]\n"
        "1. 한 문장으로 작성한다.\n"
        "2. 얼굴 표정, 말투, 발화 내용에서 보이는 단서만 쓴다.\n"
        "3. 추측성 심리 분석 대신 관찰 문장으로 쓴다.\n"
        "4. 감정 라벨 단어를 직접 쓰지 않는다.\n"
        "5. 한국어 데이터셋 요약문처럼 자연스럽고 간결하게 쓴다.\n"
    )

    return system_prompt, user_prompt


def build_multimodal_content(
    user_prompt: str,
    frames: list[Image.Image],
    sample_id: str,
    media_mode: str,
    frame_cache_dir: Path,
    max_side: int,
    jpeg_quality: int,
) -> list[dict]:
    content = [{"type": "text", "text": user_prompt}]
    for index, frame in enumerate(frames):
        resized = resize_image(frame, max_side=max_side)
        if media_mode == "file":
            image_url = save_frame_to_cache(
                resized,
                frame_cache_dir=frame_cache_dir,
                sample_id=sample_id,
                frame_index=index,
                jpeg_quality=jpeg_quality,
            )
        else:
            image_url = image_to_data_url(resized, jpeg_quality=jpeg_quality)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
        )
    return content


def request_summary(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    frames: list[Image.Image],
    sample_id: str,
    media_mode: str,
    frame_cache_dir: Path,
    max_side: int,
    jpeg_quality: int,
    temperature: float,
    max_tokens: int,
) -> str:
    content = build_multimodal_content(
        user_prompt=user_prompt,
        frames=frames,
        sample_id=sample_id,
        media_mode=media_mode,
        frame_cache_dir=frame_cache_dir,
        max_side=max_side,
        jpeg_quality=jpeg_quality,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_pt)
    videos_root = Path(args.videos_root)
    output_path = Path(args.output_pt)
    cache_path = args.cache_json or str(output_path.with_suffix(".cache.json"))
    frame_cache_dir = Path(args.frame_cache_dir)

    dataset = torch.load(input_path, map_location="cpu")
    style_examples = load_style_examples(args.style_examples_pt, args.style_examples)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    cache = load_cache(cache_path)
    output = []
    processed = 0

    for item in dataset:
        sample = dict(item)
        sample_id = str(sample.get("id", "")).strip()
        transcript = str(sample.get("script", "")).strip()
        if not sample_id or not transcript:
            output.append(sample)
            continue

        if not args.overwrite and sample_id in cache:
            sample["original_target_text"] = sample.get("target_text", "")
            sample["target_text"] = cache[sample_id]
            sample["pseudo_target_text"] = cache[sample_id]
            output.append(sample)
            continue

        video_path = parse_meld_video_path(videos_root, sample_id)
        frames = []
        if video_path.exists():
            frames = extract_frames(video_path, args.num_frames)

        if not frames and not args.allow_text_only:
            print(f"[skip] no frames: {sample_id} -> {video_path}")
            output.append(sample)
            continue

        emotion = str(sample.get("emotion", "")).strip() if args.use_emotion_label else None
        system_prompt, user_prompt = build_prompts(style_examples, transcript, emotion)
        try:
            pseudo = request_summary(
                client=client,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                frames=frames,
                sample_id=sample_id,
                media_mode=args.media_mode,
                frame_cache_dir=frame_cache_dir,
                max_side=args.max_side,
                jpeg_quality=args.jpeg_quality,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as error:
            print(f"[error] {sample_id}: {error}")
            output.append(sample)
            continue

        cache[sample_id] = pseudo
        sample["original_target_text"] = sample.get("target_text", "")
        sample["target_text"] = pseudo
        sample["pseudo_target_text"] = pseudo
        sample["pseudo_summary_model"] = args.model
        sample["pseudo_summary_has_image"] = bool(frames)
        output.append(sample)

        processed += 1
        if processed % 20 == 0:
            print(f"[ok] generated {processed} summaries")
            save_cache(cache_path, cache)

        if args.limit and processed >= args.limit:
            break

    save_cache(cache_path, cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"saved: {output_path}")
    print(f"generated: {processed}")


if __name__ == "__main__":
    main()
