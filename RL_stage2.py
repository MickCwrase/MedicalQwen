import os
import warnings
from dataclasses import dataclass
import wandb
import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from datasets import load_dataset,load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer,PreTrainedTokenizerBase
import json,random


from trl import (
    ModelConfig,
    ScriptArguments
)

from ppo_utils.ppo_config_medo1 import PPOConfig
from ppo_utils.ppo_trainer_medo1 import PPOTrainer
from model_utils import (
    configure_generation_tokens,
    get_model_path,
    get_tokenizer_path,
    load_qwen35_policy,
    load_qwen35_tokenizer,
    render_chat_template,
    resolve_torch_dtype,
)


os.environ["WANDB_MODE"] = "offline"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'



from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser
)

class ppo_dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length = 1000,debug = 0):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
    
        newdata = []
        for da in self.data:
            if len(da['Open-ended Verifiable Question']) > 0 and len(da['Ground-True Answer']) > 0:
                newdata.append({'question':da['Open-ended Verifiable Question'],'answer':da['Ground-True Answer']})
        print(len(self.data),' -> ',len(newdata))
        self.data = newdata

        self.debug = debug     

    def __getitem__(self, index):
        return self.data[index]

    def get_prompt(self,da):
        message = [{"role": "user", "content": da['question']}]
        prompt = render_chat_template(
            self.tokenizer,
            message,
            add_generation_prompt=True,
        )

        input_token = self.tokenizer(
            prompt,
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )

        da['input_ids'] = input_token["input_ids"]
        return da

    def collate_fn(self, batch):
        data = [ self.get_prompt(da) for da in batch]
        input_ids = [item["input_ids"] for item in data]
        question = [item["question"] for item in data]
        answer = [item["answer"] for item in data]

        max_len = min(max(len(x) for x in input_ids), self.max_length)
        # Keep the chat-template suffix (including the generation prompt)
        # when a long question must be truncated.
        input_ids = [
            [self.tokenizer.pad_token_id] * max(max_len - len(item), 0) + item[-max_len:]
            for item in input_ids
        ]

        if self.debug > 0:
            print('[input_ids]',self.tokenizer.decode(input_ids[-1]))
            print('[question]',question[-1])
            print('[answer]',answer[-1])
            self.debug -= 1
        return {
                "input_ids": torch.LongTensor(input_ids),
                "question": question,
                "answer": answer
            }

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # TrainingArguments may reset Accelerate's shared state during parsing.
    # Supply the plugin explicitly to avoid reading an uninitialized
    # AcceleratorState, and reuse this instance in the custom trainer.
    deepspeed_plugin = getattr(training_args, 'deepspeed_plugin', None)
    if deepspeed_plugin is None and os.getenv('ACCELERATE_USE_DEEPSPEED', 'false').lower() == 'true':
        deepspeed_plugin = DeepSpeedPlugin()
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        deepspeed_plugin=deepspeed_plugin,
    )

    if not model_config.model_name_or_path:
        model_config.model_name_or_path = get_model_path()
    tokenizer_path = training_args.tokenizer_path or get_tokenizer_path(model_config.model_name_or_path)
    dtype_name = model_config.torch_dtype or os.getenv('HUATUO_TORCH_DTYPE', 'bfloat16')
    attn_implementation = getattr(model_config, 'attn_implementation', None) or os.getenv('HUATUO_ATTN_IMPLEMENTATION', 'auto')
    torch_dtype = resolve_torch_dtype(dtype_name)
    if not training_args.reward_model_path:
        raise ValueError('必须通过配置文件或命令行提供 reward_model_path')
    # A Qwen3.5 value backbone must use the same vocabulary as the policy.
    # Unless explicitly overridden, initialize it from the PPO policy checkpoint.
    if not training_args.value_model_path:
        training_args.value_model_path = model_config.model_name_or_path

    output_dir = training_args.output_dir
    run_name = training_args.run_name
    if run_name not in output_dir:
        output_dir = os.path.join(output_dir,run_name)
        training_args.output_dir = output_dir
    
    tokenizer = load_qwen35_tokenizer(
        tokenizer_path,
        trust_remote_code=model_config.trust_remote_code,
        padding_side='left',
    )
    reward_model_kwargs = {"num_labels": 2, "torch_dtype": torch_dtype}
    # The custom PPO wrapper reads ``hidden_states[-1]`` from the value
    # backbone.  Keep this explicit because most sequence-classification
    # configs disable hidden-state outputs by default.
    value_model_kwargs = {"num_labels": 1, "torch_dtype": torch_dtype, "output_hidden_states": True}
    if training_args.reward_attn_implementation not in {None, 'auto', 'none'}:
        reward_model_kwargs["attn_implementation"] = training_args.reward_attn_implementation
    if training_args.value_attn_implementation not in {None, 'auto', 'none'}:
        value_model_kwargs["attn_implementation"] = training_args.value_attn_implementation

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        **reward_model_kwargs,
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.value_model_path,
        trust_remote_code=model_config.trust_remote_code,
        **value_model_kwargs,
    )

    ref_policy = load_qwen35_policy(
        model_config.model_name_or_path,
        dtype_name=dtype_name,
        attn_implementation=attn_implementation,
        trust_remote_code=model_config.trust_remote_code,
        freeze_vision_encoder=training_args.freeze_vision_encoder,
    )
    policy = load_qwen35_policy(
        model_config.model_name_or_path,
        dtype_name=dtype_name,
        attn_implementation=attn_implementation,
        trust_remote_code=model_config.trust_remote_code,
        freeze_vision_encoder=training_args.freeze_vision_encoder,
    )
    configure_generation_tokens(ref_policy, tokenizer)
    configure_generation_tokens(policy, tokenizer)

    reward_tokenizer = AutoTokenizer.from_pretrained(training_args.reward_model_path)
    if reward_tokenizer.pad_token_id is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    assert tokenizer.pad_token_id != tokenizer.eos_token_id

    training_args.stop_token_id = tokenizer.eos_token_id

    eval_ratio = 0.1
    eval_max_num = 200
    with open(script_args.dataset_name) as f:
        data = json.load(f)
    random.seed(training_args.seed)
    random.shuffle(data)
    eval_num = min(int(len(data) * eval_ratio),eval_max_num)
    train_dataset = ppo_dataset(data[eval_num:],tokenizer, debug = 1)
    eval_dataset = ppo_dataset(data[:eval_num],tokenizer)

    trainer = PPOTrainer(
        accelerator=accelerator,
        config=training_args,
        processing_class=tokenizer,
        reward_processing_class = reward_tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator = train_dataset.collate_fn
    )
    trainer.train()
