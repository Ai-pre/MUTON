import argparse
import sys
from pathlib import Path

import torch
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qwen_omni_dataset import DEFAULT_INSTRUCTION, DEFAULT_SYSTEM_PROMPT, build_messages, materialize_messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Qwen2.5-Omni Thinker inference on image+audio+text input.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--image", type=str, action="append", required=True, help="One or more image paths")
    parser.add_argument("--audio", type=str, required=True, help="Audio path")
    parser.add_argument("--script", type=str, required=True, help="Transcript/script text")
    parser.add_argument("--prompt", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--adapter", type=str, default="", help="Optional LoRA adapter directory")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max_new_tokens", type=int, default=96)
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]

    processor = Qwen2_5OmniProcessor.from_pretrained(args.adapter or args.model_name)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)

    messages = build_messages(
        image_paths=args.image,
        audio_path=args.audio,
        script=args.script,
        target_text="",
        instruction=args.prompt,
        system_prompt=args.system_prompt,
        emotion="",
    )[:-1]
    messages = materialize_messages(
        messages,
        audio_sampling_rate=getattr(processor.feature_extractor, "sampling_rate", 16000),
    )

    inputs = processor.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(model.device) if torch.is_tensor(value) else value for key, value in dict(inputs).items()}

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    prompt_len = inputs["input_ids"].shape[1]
    generated_text = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)[0]
    print(generated_text.strip())


if __name__ == "__main__":
    main()
