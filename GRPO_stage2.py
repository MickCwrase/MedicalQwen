"""GRPO post-training entry point for the Qwen3.5 adaptation.

This keeps the verifiable-problem data and medical verifier used by the PPO
stage, but replaces the critic/value path with TRL's group-relative objective.
The script requires a TRL release that provides ``GRPOTrainer`` and
``GRPOConfig`` (the repository's PPO pin, TRL 0.13.0, is too old).
"""

from __future__ import annotations

import inspect
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

try:
    from trl import GRPOConfig, GRPOTrainer
except ImportError as exc:  # pragma: no cover - depends on the remote environment
    raise ImportError(
        "GRPO_stage2.py requires a TRL release with GRPOTrainer/GRPOConfig "
        "(install a current TRL release; the PPO environment's trl==0.13.0 "
        "does not provide this interface)."
    ) from exc

from model_utils import (
    configure_generation_tokens,
    get_model_path,
    get_tokenizer_path,
    load_qwen35_policy,
    load_qwen35_tokenizer,
    render_chat_template,
    resolve_torch_dtype,
    str_to_bool,
)


FINAL_HEADER = "## Final Response\n\n"
THINKING_HEADER = "## Thinking"
VERIFIER_PROMPT = """<Model Response>
{}
</Model Response>

<Reference Answer>
{}
</Reference Answer>

Your task is to evaluate the model response by comparing it to the reference answer. If the model response is correct and aligns with the reference answer, output \"True\" . If it is incorrect or fails to select the correct option (if options are provided), output \"False\" . {}"""


@dataclass
class GRPOStage2Arguments:
    """Arguments specific to this project's data and verifier."""

    model_name_or_path: str | None = field(default=None)
    tokenizer_path: str | None = field(default=None)
    reward_model_path: str | None = field(default=None)
    dataset_name: str = "data/medical-o1-verifiable-problem/medical_o1_verifiable_problem.json"
    response_length: int = 1024
    prompt_length_limit: int = 1000
    reward_max_length: int = 4000
    torch_dtype: str = field(default_factory=lambda: os.getenv("HUATUO_TORCH_DTYPE", "bfloat16"))
    attn_implementation: str = field(
        default_factory=lambda: os.getenv("HUATUO_REWARD_ATTN_IMPLEMENTATION", "sdpa")
    )
    freeze_vision_encoder: bool = field(
        default_factory=lambda: str_to_bool(os.getenv("HUATUO_FREEZE_VISION_ENCODER", "true"))
    )
    enable_thinking: bool = field(
        default_factory=lambda: str_to_bool(os.getenv("HUATUO_ENABLE_THINKING", "false"))
    )


def _completion_text(completion: Any) -> str:
    """Normalize TRL's string or chat-message completion representation."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        for message in reversed(completion):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content", ""))
        if completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", ""))
    return str(completion)


def _extract_final_response(response: str) -> str | None:
    """Apply the same strict output contract as the PPO verifier reward."""
    if response.count(FINAL_HEADER) != 1 or response.count(THINKING_HEADER) != 1:
        return None
    match = re.search(r"## Final Response\n\n(.*)", response, flags=re.S)
    return match.group(1).strip() if match else None


class MedicalVerifierReward:
    """Callable reward function consumed by TRL's GRPOTrainer."""

    def __init__(self, model_path: str, dtype_name: str, attn_implementation: str, max_length: int):
        dtype = resolve_torch_dtype(dtype_name)
        kwargs: dict[str, Any] = {"num_labels": 2, "torch_dtype": dtype}
        if attn_implementation.lower() not in {"auto", "none"}:
            kwargs["attn_implementation"] = attn_implementation
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length

    @torch.no_grad()
    def __call__(self, completions: list[Any], answer: list[str] | None = None, **_: Any) -> list[float]:
        answers = answer or [""] * len(completions)
        prompts: list[str] = []
        valid: list[bool] = []
        for completion, reference in zip(completions, answers):
            final_response = _extract_final_response(_completion_text(completion))
            valid.append(final_response is not None)
            prompts.append(
                VERIFIER_PROMPT.format(
                    final_response if final_response is not None else "I do not know the answer.",
                    reference,
                    self.tokenizer.eos_token or "",
                )
            )
        batch = self.tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self.max_length,
            padding=True,
            truncation=True,
        ).to(self.device)
        probabilities = torch.softmax(self.model(**batch, return_dict=True).logits, dim=-1)[:, 1]
        # Keep the PPO reward scale exactly: malformed=0, formatted but
        # verifier-negative=0.1, verifier-positive=1.0.
        return [0.0 if not ok else (1.0 if float(prob) > 0.4 else 0.1) for ok, prob in zip(valid, probabilities)]


def _load_dataset(path: str, tokenizer, enable_thinking: bool, max_prompt_length: int, seed: int) -> Any:
    from datasets import Dataset

    with open(path, encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")
    rows = []
    for record in records:
        question = str(record.get("Open-ended Verifiable Question", "")).strip()
        answer = str(record.get("Ground-True Answer", "")).strip()
        if not question or not answer:
            continue
        prompt = render_chat_template(
            tokenizer,
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if token_count <= max_prompt_length:
            rows.append({"prompt": prompt, "answer": answer})
    if not rows:
        raise ValueError(f"No usable records remain after filtering {path}")
    random.Random(seed).shuffle(rows)
    return Dataset.from_list(rows)


def main() -> None:
    parser = HfArgumentParser((GRPOStage2Arguments, GRPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    if not script_args.model_name_or_path:
        script_args.model_name_or_path = get_model_path()
    if not script_args.tokenizer_path:
        script_args.tokenizer_path = get_tokenizer_path(script_args.model_name_or_path)
    if not script_args.reward_model_path:
        script_args.reward_model_path = os.getenv("HUATUO_REWARD_MODEL_PATH")
    if not script_args.reward_model_path:
        raise ValueError("Provide --reward_model_path or HUATUO_REWARD_MODEL_PATH")

    random.seed(training_args.seed)
    tokenizer = load_qwen35_tokenizer(script_args.tokenizer_path, padding_side="left")
    policy = load_qwen35_policy(
        script_args.model_name_or_path,
        dtype_name=script_args.torch_dtype,
        attn_implementation=os.getenv("HUATUO_ATTN_IMPLEMENTATION", "auto"),
        freeze_vision_encoder=script_args.freeze_vision_encoder,
    )
    configure_generation_tokens(policy, tokenizer)
    dataset = _load_dataset(
        script_args.dataset_name,
        tokenizer,
        script_args.enable_thinking,
        script_args.prompt_length_limit,
        training_args.seed,
    )
    reward_fn = MedicalVerifierReward(
        script_args.reward_model_path,
        script_args.torch_dtype,
        script_args.attn_implementation,
        script_args.reward_max_length,
    )

    # GRPOConfig names this limit max_completion_length; keep the explicit
    # assignment here so the CLI remains easy to compare with PPO's
    # --response_length flag.
    if hasattr(training_args, "max_completion_length"):
        training_args.max_completion_length = script_args.response_length
    if hasattr(training_args, "remove_unused_columns"):
        training_args.remove_unused_columns = False

    trainer_kwargs: dict[str, Any] = {
        "model": policy,
        "reward_funcs": reward_fn,
        "train_dataset": dataset,
        "args": training_args,
    }
    parameter_names = inspect.signature(GRPOTrainer.__init__).parameters
    if "processing_class" in parameter_names:
        trainer_kwargs["processing_class"] = tokenizer
    else:  # compatibility with early GRPO releases
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()


if __name__ == "__main__":
    main()
