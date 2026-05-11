"""
AIPA-Eval LoRA 파인튜닝 (로컬 RTX 4070 Ti)
"""
import json
import random
import torch
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ========== 설정 (튜닝용 config dict) ==========
CONFIG = {
    # Model
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    # Data
    "data_path": Path(__file__).parent / "data" / "training_data_deduped.jsonl",
    "train_split": 0.9,
    # Output
    "output_dir": Path(__file__).parent / "models",
    # LoRA hyperparameters
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # Training hyperparameters
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "optim": "paged_adamw_8bit",
}

MODEL_NAME = CONFIG["model_name"]
DATA_PATH = CONFIG["data_path"]
OUTPUT_DIR = CONFIG["output_dir"]
TRAINING_LOG_PATH = OUTPUT_DIR / "training_log.jsonl"


class JSONLLoggerCallback(TrainerCallback):
    """Logs training loss to a JSONL file for easy post-hoc analysis."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            entry = {
                "step": state.global_step,
                "epoch": round(state.epoch, 4) if state.epoch else None,
                "loss": logs.get("loss"),
                "eval_loss": logs.get("eval_loss"),
                "learning_rate": logs.get("learning_rate"),
                "timestamp": datetime.now().isoformat(),
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("=" * 50)
print("AIPA-Eval LoRA Fine-tuning (Local)")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
print("=" * 50)

# ========== 1. 데이터 로드 ==========
print("\n[1/5] 학습 데이터 로드...")

def format_training_example(item: dict) -> str:
    inp = item["input"]
    out = item["output"]
    prompt = f"""당신은 소비자 반응 평가 전문가입니다.

다음 자극물에 대해 주어진 페르소나의 관점에서 평가해주세요.

[자극물]
유형: {inp['stimulus_type']}
내용: {inp['stimulus']}

[페르소나]
연령대: {inp['persona_age_group']}
성별: {inp['persona_gender']}
직업: {inp['persona_occupation']}
소득: {inp['persona_income']}
특성: {', '.join(inp['persona_traits'])}

[평가 축]
{', '.join(inp['axes'])}

각 축에 대해 1-10점 점수와 근거를 JSON으로 응답하세요."""
    response = json.dumps(out, ensure_ascii=False)
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"

training_texts = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            training_texts.append(format_training_example(json.loads(line)))

random.shuffle(training_texts)
split_idx = int(len(training_texts) * CONFIG["train_split"])
train_dataset = Dataset.from_dict({"text": training_texts[:split_idx]})
eval_dataset = Dataset.from_dict({"text": training_texts[split_idx:]})
print(f"  학습: {len(train_dataset)}건, 검증: {len(eval_dataset)}건")

# ========== 2. 모델 로드 ==========
print("\n[2/5] 모델 로드 (4bit 양자화)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
print(f"  GPU 메모리: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

# ========== 3. LoRA 설정 ==========
print("\n[3/5] LoRA 어댑터 설정...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=CONFIG["target_modules"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========== 4. 학습 ==========
print("\n[4/5] 학습 시작...")
checkpoint_dir = OUTPUT_DIR / "checkpoints"
training_args = SFTConfig(
    output_dir=str(checkpoint_dir),
    num_train_epochs=CONFIG["num_train_epochs"],
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    lr_scheduler_type=CONFIG["lr_scheduler_type"],
    warmup_steps=CONFIG["warmup_steps"],
    fp16=False,
    bf16=True,
    optim=CONFIG["optim"],
    logging_steps=CONFIG["logging_steps"],
    save_strategy=CONFIG["save_strategy"],
    eval_strategy=CONFIG["eval_strategy"],
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    callbacks=[JSONLLoggerCallback(TRAINING_LOG_PATH)],
)

# Check for existing checkpoints to resume from
resume_from = None
if checkpoint_dir.exists():
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    if checkpoints:
        resume_from = str(checkpoints[-1])
        print(f"  Resuming from checkpoint: {resume_from}")

trainer.train(resume_from_checkpoint=resume_from)
print("\n학습 완료!")

# ========== 5. 저장 ==========
print("\n[5/5] 어댑터 저장...")
save_path = str(OUTPUT_DIR / "aipa-eval-lora")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

import os
total_size = sum(os.path.getsize(os.path.join(save_path, f)) for f in os.listdir(save_path))
print(f"  저장 경로: {save_path}")
print(f"  어댑터 크기: {total_size / 1024 / 1024:.1f}MB")
print("\n" + "=" * 50)
print("AIPA-Eval 파인튜닝 완료!")
print("=" * 50)
