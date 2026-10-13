import os
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str((PROJECT_ROOT.parent / "Qwen3.5-0.8B").resolve())


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"无法解析布尔值: {value}")


def get_model_path() -> str:
    return os.getenv("HUATUO_MODEL_PATH", DEFAULT_MODEL_PATH)


def get_tokenizer_path(model_path: Optional[str] = None) -> str:
    return os.getenv("HUATUO_TOKENIZER_PATH", model_path or get_model_path())


def resolve_torch_dtype(dtype_name: Optional[str]):
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    if dtype_name is None or str(dtype_name).lower() == "auto":
        return "auto"
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    normalized = str(dtype_name).lower()
    if normalized not in dtype_map:
        raise ValueError(f"不支持的模型精度: {dtype_name}")
    return dtype_map[normalized]


def _model_load_kwargs(dtype_name: Optional[str], attn_implementation: Optional[str]):
    kwargs = {"torch_dtype": resolve_torch_dtype(dtype_name)}
    if attn_implementation and attn_implementation.lower() not in {"auto", "none"}:
        kwargs["attn_implementation"] = attn_implementation
    return kwargs


def validate_qwen35_config(config):
    if config.model_type not in {"qwen3_5", "qwen3_5_text"}:
        raise ValueError(
            f"策略模型必须是 Qwen3.5，实际 model_type={config.model_type}"
        )

    text_config = getattr(config, "text_config", config)
    required_fields = (
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_parameters",
    )
    missing_fields = [field for field in required_fields if not hasattr(text_config, field)]
    if missing_fields:
        raise ValueError(f"Qwen3.5 配置缺少必要字段: {missing_fields}")

    if text_config.num_key_value_heads > text_config.num_attention_heads:
        raise ValueError("Qwen3.5 的 KV head 数不能大于 attention head 数")

    layer_types = getattr(text_config, "layer_types", [])
    if layer_types and not {"linear_attention", "full_attention"}.issubset(set(layer_types)):
        raise ValueError("Qwen3.5 配置未包含预期的线性注意力与全注意力混合层")
    return text_config


def load_qwen35_tokenizer(
    tokenizer_path: str,
    trust_remote_code: bool = False,
    padding_side: Optional[str] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
    )
    if not tokenizer.chat_template:
        raise ValueError("Qwen3.5 Tokenizer 必须提供原生 chat_template")
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Qwen3.5 Tokenizer 必须同时定义 pad_token_id 和 eos_token_id")
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("Qwen3.5 的 pad_token_id 与 eos_token_id 必须不同")
    if padding_side:
        tokenizer.padding_side = padding_side
    return tokenizer


def _freeze_vision_encoder(model) -> bool:
    backbone = getattr(model, "model", None)
    vision_encoder = getattr(backbone, "visual", None)
    if vision_encoder is None:
        vision_encoder = getattr(model, "visual", None)
    if vision_encoder is None:
        return False
    vision_encoder.requires_grad_(False)
    return True


def load_qwen35_policy(
    model_path: str,
    dtype_name: str = "bfloat16",
    attn_implementation: str = "auto",
    trust_remote_code: bool = False,
    freeze_vision_encoder: bool = True,
):
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    validate_qwen35_config(config)

    # Qwen3.5 完整权重是统一视觉-语言模型；纯文本变体仍走标准 CausalLM 接口。
    model_loader = (
        AutoModelForImageTextToText
        if config.model_type == "qwen3_5"
        else AutoModelForCausalLM
    )
    model = model_loader.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=trust_remote_code,
        **_model_load_kwargs(dtype_name, attn_implementation),
    )
    if freeze_vision_encoder:
        vision_frozen = _freeze_vision_encoder(model)
        if config.model_type == "qwen3_5" and not vision_frozen:
            raise RuntimeError("未找到 Qwen3.5 视觉编码器，无法确认纯文本冻结配置")
    return model


def configure_generation_tokens(model, tokenizer):
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id


def render_chat_template(
    tokenizer,
    messages,
    add_generation_prompt: bool,
    enable_thinking: Optional[bool] = None,
):
    # 默认关闭 Qwen 原生 thinking，保留 HuatuoGPT-o1 的显式输出格式。
    if enable_thinking is None:
        enable_thinking = str_to_bool(os.getenv("HUATUO_ENABLE_THINKING", "false"))
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
