"""
AIPA 이유 생성 모델 v2 - 플랫폼 톤 학습 통합

기존 train_reasoning.py 를 확장:
  - task='evaluation'    : 페르소나 평가 이유 생성 (기존)
  - task='survey'        : 설문 보기 선택 + 이유 (기존)
  - task='platform_reaction' : SNS 플랫폼 톤 반응 생성 (신규)

입력:
  training/data/reasoning_with_platform.jsonl
  (training/combine_platform_data.py 로 생성)

사용법:
  python training/train_reasoning_v2.py
  python training/train_reasoning_v2.py --epochs 3 --lr 2e-4
"""

import json
import random
import argparse
import sys
from pathlib import Path
from datetime import datetime

# 윈도우 콘솔 print 즉시 출력 (buffering 비활성)
import os
os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None
print(">>> 스크립트 시작", flush=True)

# !! 중요: 윈도우에서 torch 보다 datasets/pyarrow 를 먼저 import 해야 silent crash 방지
from datasets import Dataset
print(">>> datasets 로드", flush=True)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
print(">>> transformers 로드", flush=True)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
print(">>> peft 로드", flush=True)

from trl import SFTTrainer, SFTConfig
print(">>> trl 로드", flush=True)

import torch
print(f">>> torch 로드 ({torch.__version__}, cuda={torch.cuda.is_available()})", flush=True)

import bitsandbytes
print(f">>> bitsandbytes 로드 ({bitsandbytes.__version__})", flush=True)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
print(">>> 모든 import 완료", flush=True)

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────

CONFIG = {
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "data_path": Path(__file__).parent / "data" / "reasoning_with_platform.jsonl",
    "output_dir": Path(__file__).parent / "models" / "reasoning_v2",
    "train_split": 0.9,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "num_train_epochs": 2,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "optim": "paged_adamw_8bit",
}


# ─────────────────────────────────────────────
# 데이터 포맷터 (멀티태스크)
# ─────────────────────────────────────────────

def format_training_example(item: dict) -> str:
    """학습 데이터를 ChatML 프롬프트로 변환"""
    inp = item["input"]
    out = item["output"]
    task = inp.get("task", "evaluation")
    traits_str = ", ".join(inp.get("persona_traits", [])) or "없음"

    if task == "platform_reaction":
        # 신규: 플랫폼 톤 반응
        platform_name = inp.get("platform_name", inp.get("platform", "SNS"))
        scores_text = "\n".join([f"- {s['name']}: {s['score']}점" for s in inp.get("scores", [])])

        prompt = f"""당신은 SNS 플랫폼 사용자 시뮬레이션 전문가입니다.
다음 페르소나가 {platform_name}에서 작성할 법한 반응을 그 플랫폼 톤으로 작성하세요.

[플랫폼]
{platform_name}

[자극물]
유형: {inp.get('stimulus_type', '')}
내용: {inp.get('stimulus', '')[:200]}

[페르소나]
연령대: {inp['persona_age_group']}
성별: {inp['persona_gender']}
직업: {inp['persona_occupation']}
특성: {traits_str}

[참고 점수]
{scores_text}

이 사용자가 {platform_name}에서 쓸 법한 1~2문장 반응을 그 플랫폼 톤 그대로 작성하세요."""

    elif task == "survey":
        choices_text = "\n".join([f"  {i+1}. {c}" for i, c in enumerate(inp.get("choices", []))])
        scores_text = "\n".join([f"- {s['name']}: {s['score']}점" for s in inp.get("scores", [])])

        prompt = f"""당신은 설문 응답 시뮬레이션 전문가입니다.
다음 페르소나로 설문에 답변하세요.

[설문]
카테고리: {inp.get('stimulus_type', '')}
질문: {inp.get('stimulus', '')}
보기:
{choices_text}

[페르소나]
연령대: {inp['persona_age_group']}
성별: {inp['persona_gender']}
직업: {inp['persona_occupation']}
특성: {traits_str}

[참고 점수]
{scores_text}

보기 중 하나를 선택하고 이유를 1문장으로 설명하세요."""

    else:
        # evaluation (기존)
        scores_text = "\n".join([f"- {s['name']}: {s['score']}점" for s in inp.get("scores", [])])

        prompt = f"""당신은 소비자 반응 평가 전문가입니다.
다음 평가 결과의 이유를 페르소나 관점에서 설명해주세요.

[자극물]
유형: {inp['stimulus_type']}
내용: {inp['stimulus'][:200]}

[페르소나]
연령대: {inp['persona_age_group']}
성별: {inp['persona_gender']}
직업: {inp['persona_occupation']}
특성: {traits_str}

[평가 점수]
{scores_text}

각 점수의 이유를 1문장씩 설명하고 한줄평을 작성하세요."""

    response = json.dumps(out, ensure_ascii=False)
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"


# ─────────────────────────────────────────────
# 메인 학습
# ─────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print("AIPA 이유 생성 모델 v2 학습 (Qwen 2.5-0.5B + 플랫폼 톤)")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("GPU 없음 - CPU 학습 (매우 느림)")
    print("=" * 60)

    data_path = args.data or CONFIG["data_path"]
    output_dir = Path(args.output or CONFIG["output_dir"])

    # 1. 데이터 로드
    print("\n[1/5] 데이터 로드...")
    by_task = {}
    training_texts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            task = item.get("input", {}).get("task", "evaluation")
            by_task[task] = by_task.get(task, 0) + 1
            training_texts.append(format_training_example(item))

    print(f"  태스크별 분포:")
    for task, count in by_task.items():
        print(f"    {task}: {count}건")

    random.seed(42)
    random.shuffle(training_texts)
    split_idx = int(len(training_texts) * CONFIG["train_split"])
    train_dataset = Dataset.from_dict({"text": training_texts[:split_idx]})
    eval_dataset = Dataset.from_dict({"text": training_texts[split_idx:]})
    print(f"  학습: {len(train_dataset)}건, 검증: {len(eval_dataset)}건")

    # 2. 모델 로드 (4bit 양자화)
    model_name = args.model or CONFIG["model_name"]
    print(f"\n[2/5] 모델 로드: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        print(f"  GPU 메모리: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    # 3. LoRA 설정
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

    # 4. 학습
    epochs = args.epochs or CONFIG["num_train_epochs"]
    print(f"\n[4/5] 학습 시작 (epochs: {epochs})...")
    checkpoint_dir = output_dir / "checkpoints"

    training_args = SFTConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=args.lr or CONFIG["learning_rate"],
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
    )

    trainer.train()
    print("\n학습 완료!")

    # 5. 어댑터 저장
    print("\n[5/5] 어댑터 저장...")
    save_path = str(output_dir / "aipa-reasoning-platform-lora")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    import os
    total_size = sum(os.path.getsize(os.path.join(save_path, f)) for f in os.listdir(save_path))
    print(f"  저장 경로: {save_path}")
    print(f"  어댑터 크기: {total_size / 1024 / 1024:.1f}MB")
    print()
    print("다음 단계:")
    print("  1. python training/merge_and_export.py     # 어댑터 + 베이스 모델 병합")
    print("  2. GGUF 변환 후 Cloud Run 배포")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()
    main(args)
