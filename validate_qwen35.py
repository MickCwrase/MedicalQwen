import argparse
import json

import torch
from transformers import AutoConfig

from model_utils import (
    get_model_path,
    get_tokenizer_path,
    load_qwen35_policy,
    load_qwen35_tokenizer,
    render_chat_template,
    validate_qwen35_config,
)


def main():
    parser = argparse.ArgumentParser(description="验证本地 Qwen3.5 模型与 HuatuoGPT-o1 模板契约")
    parser.add_argument("--model_path", default=get_model_path())
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path or get_tokenizer_path(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=False)
    text_config = validate_qwen35_config(config)
    tokenizer = load_qwen35_tokenizer(tokenizer_path)

    if tokenizer.pad_token_id != 248044 or tokenizer.eos_token_id != 248046:
        raise ValueError(
            f"特殊 token 与本地 Qwen3.5-0.8B 不一致: "
            f"pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id}"
        )

    answer = "## Thinking\n\nreason\n\n## Final Response\n\nanswer"
    full_text = render_chat_template(
        tokenizer,
        [
            {"role": "user", "content": "test question"},
            {"role": "assistant", "content": answer},
        ],
        add_generation_prompt=False,
    )
    query_text = render_chat_template(
        tokenizer,
        [{"role": "user", "content": "test question"}],
        add_generation_prompt=True,
    )
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    query_ids = tokenizer.encode(query_text, add_special_tokens=False)
    if full_ids[:len(query_ids)] != query_ids:
        raise ValueError("non-thinking query 不是完整 SFT 样本的 token 前缀")

    summary = {
        "model_type": config.model_type,
        "architecture": config.architectures[0],
        "num_hidden_layers": text_config.num_hidden_layers,
        "hidden_size": text_config.hidden_size,
        "num_attention_heads": text_config.num_attention_heads,
        "num_key_value_heads": text_config.num_key_value_heads,
        "rms_norm_eps": text_config.rms_norm_eps,
        "max_position_embeddings": text_config.max_position_embeddings,
        "linear_attention_layers": text_config.layer_types.count("linear_attention"),
        "full_attention_layers": text_config.layer_types.count("full_attention"),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "template_prefix_valid": True,
    }

    if args.load_model:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("--load_model 使用 CUDA 时必须存在可见 GPU")
        model = load_qwen35_policy(
            args.model_path,
            dtype_name="bfloat16",
            attn_implementation="auto",
            freeze_vision_encoder=True,
        )
        model.to(args.device)
        model.eval()
        inputs = tokenizer(query_text, return_tensors="pt").to(args.device)
        with torch.no_grad():
            output = model(**inputs, use_cache=False, return_dict=True)
            generated = model.generate(**inputs, max_new_tokens=4)
        if output.logits.shape[:2] != inputs.input_ids.shape:
            raise RuntimeError("Qwen3.5 文本前向输出维度异常")
        summary["forward_valid"] = True
        summary["generated_tokens"] = generated.shape[1] - inputs.input_ids.shape[1]

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
