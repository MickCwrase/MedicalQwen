# MedicalQwen

HuatuoGPT-o1 的 Qwen3.5 适配版本，保留医学推理数据构建、SFT、PPO、GRPO 和评估的功能代码。
基于 [HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1)；方法见 [原论文](https://arxiv.org/abs/2412.18925)。

本仓库仅包含代码、依赖和运行配置。数据集、模型权重、训练日志、评估结果、论文附件和实验报告均不随仓库发布。

## 目录

- `SFT_stage1.py`：监督微调。
- `RL_stage2.py`、`ppo_utils/`：基于医疗 verifier 奖励的 PPO。
- `GRPO_stage2.py`：GRPO 训练入口，使用独立依赖环境。
- `model_utils.py`、`validate_qwen35.py`：Qwen3.5 模型加载、模板及配置验证。
- `construct_verifiable_medical_problems.py`：构建可验证医学问题。
- `search_for_complex_reasoning_path*.py`：英文及中文推理路径搜索。
- `evaluation/`：OpenAI 兼容服务评估与答案评分。
- `configs/`：Accelerate/DeepSpeed 配置及模型、GPU 参数。

## 环境与输入

训练建议使用 Linux、Python 3.11、CUDA GPU；安装前根据服务器 CUDA 环境准备 PyTorch。

```bash
python -m pip install -r requirements.txt
export HUATUO_MODEL_PATH=/path/to/Qwen3.5-0.8B
export HUATUO_TOKENIZER_PATH="$HUATUO_MODEL_PATH"
export HUATUO_REWARD_MODEL_PATH=/path/to/medical_o1_verifier_3B
source configs/qwen3_5.env
export GPU_PROFILE=a80080  # 或 vgpu48
source configs/gpu_profiles.env
python validate_qwen35.py --model_path "$HUATUO_MODEL_PATH"
```

GRPO 应在独立环境安装 `requirements-grpo.txt`，不要覆盖 PPO 的 `trl==0.13.0` 环境。
GPU 数量由下方 `--num_processes` 明确设置，需与 `CUDA_VISIBLE_DEVICES` 一致。
本地文件齐全后可设置 `TRANSFORMERS_OFFLINE=1`、`HF_HUB_OFFLINE=1`。

数据需自行准备：

- [SFT 数据](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)：JSON 记录包含 `Question`、`Complex_CoT`、`Response`。
- [PPO/GRPO 数据](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-verifiable-problem)：JSON 记录包含 `Open-ended Verifiable Question`、`Ground-True Answer`。
- [医疗 verifier](https://huggingface.co/FreedomIntelligence/medical_o1_verifier_3B)：下载后设置奖励模型路径。
- 评估 JSON 自行提供，字段格式以 `evaluation/eval.py` 和 `evaluation/scorer.py` 为准。

## SFT

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file "$ACCELERATE_CONFIG" --num_processes 1 \
  SFT_stage1.py \
  --model_path "$HUATUO_MODEL_PATH" \
  --tokenizer_path "$HUATUO_TOKENIZER_PATH" \
  --data_path /path/to/medical_o1_sft.json \
  --max_seq_len "$SFT_MAX_SEQ_LEN" \
  --train_bsz_per_gpu "$SFT_TRAIN_BSZ" \
  --gradient_accumulation_steps "$SFT_GRAD_ACCUM" \
  --gradient_checkpointing --output_dir ./ckpts
```

## PPO

`SFT_MODEL_PATH` 指向 SFT 导出的模型目录。value backbone 使用同一模型以保持词表一致。

```bash
export SFT_MODEL_PATH=/path/to/sft-checkpoint
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file "$ACCELERATE_CONFIG" --num_processes 1 \
  RL_stage2.py \
  --model_name_or_path "$SFT_MODEL_PATH" \
  --tokenizer_path "$SFT_MODEL_PATH" \
  --value_model_path "$SFT_MODEL_PATH" \
  --reward_model_path "$HUATUO_REWARD_MODEL_PATH" \
  --dataset_name /path/to/medical_o1_verifiable_problem.json \
  --response_length "$PPO_RESPONSE_LENGTH" \
  --local_rollout_forward_batch_size "$PPO_ROLLOUT_BATCH" \
  --total_episodes 1 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 --num_mini_batches 1 \
  --bf16 true --output_dir ./ckpts/ppo --report_to none
```

先用小数据集验证运行，再调整训练规模。GRPO 在独立环境中使用 `GRPO_stage2.py`，
传入模型、tokenizer、verifier、数据路径和 `--num_generations` 等参数。

## 评估

另行安装并启动支持目标模型的 SGLang OpenAI 兼容服务，然后运行：

```bash
python evaluation/eval.py \
  --model_name "$SFT_MODEL_PATH" --tokenizer_path "$SFT_MODEL_PATH" \
  --eval_file /path/to/eval_data.json --port 30000 --strict_prompt
```

评估输出留在本地；忽略规则防止数据、权重与常见运行产物进入版本控制。
