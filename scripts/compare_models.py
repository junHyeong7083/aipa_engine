"""
기존 0.5B 모델 vs 플랫폼 학습 v2 모델 비교 시연

기존 모델:
  training/models/reasoning/aipa-reasoning-lora (또는 merged)
  → 일반 평가 톤

v2 모델:
  training/models/reasoning_v2/aipa-reasoning-platform-lora
  → 플랫폼 톤 학습 추가 (유튜브 / 디시 / Claude 합성)

발표 시연용:
  같은 프롬프트(페르소나 + 자극물 + 점수)를 양쪽에 던지고 응답 비교.
  플랫폼 톤이 v2에서 어떻게 다른지 시각적으로 보여줌.

사용법:
  python scripts/compare_models.py
"""

import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

# 윈도우 import 순서 (datasets 먼저)
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = Path(__file__).parent.parent
EXISTING_LORA = ROOT / "training" / "models" / "reasoning" / "aipa-reasoning-lora"
V2_LORA = ROOT / "training" / "models" / "reasoning_v2" / "aipa-reasoning-platform-lora"


def load_base_model_and_tokenizer():
    """베이스 모델 + 토크나이저를 1번만 로드"""
    print(f"  베이스 모델 로드: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def apply_adapter(base_model, lora_path: Path, adapter_name: str):
    """베이스 모델에 LoRA 어댑터 부착"""
    if not lora_path.exists():
        print(f"  [경고] {adapter_name}: LoRA 없음 → 베이스 모델로 추론")
        return base_model
    print(f"  {adapter_name} 어댑터 적용: {lora_path.name}")
    model = PeftModel.from_pretrained(base_model, str(lora_path), adapter_name=adapter_name)
    return model


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


# 시연용 프롬프트: 같은 페르소나 + 자극물을 다른 플랫폼 컨텍스트로
SCENARIOS = [
    {
        "title": "디시인사이드 사용자 — 무선이어폰 평가",
        "prompt": """당신은 SNS 플랫폼 사용자 시뮬레이션 전문가입니다.
다음 페르소나가 디시인사이드에서 작성할 법한 반응을 그 플랫폼 톤으로 작성하세요.

[플랫폼] 디시인사이드

[자극물]
유형: 가전/전자
내용: 무선 이어폰 ProAir 5세대 18만원, 노이즈캔슬링 신모델

[페르소나]
연령대: 20대
성별: male
직업: 대학생
특성: 가성비 중시, 익명적, 냉소적, 솔직함

[참고 점수]
- 관심도: 60점

이 사용자가 디시인사이드에서 쓸 법한 1~2문장 반응을 그 플랫폼 톤 그대로 작성하세요.""",
    },
    {
        "title": "유튜브 사용자 — 무선이어폰 평가",
        "prompt": """당신은 SNS 플랫폼 사용자 시뮬레이션 전문가입니다.
다음 페르소나가 유튜브에서 작성할 법한 반응을 그 플랫폼 톤으로 작성하세요.

[플랫폼] 유튜브

[자극물]
유형: 가전/전자
내용: 무선 이어폰 ProAir 5세대 18만원, 노이즈캔슬링 신모델

[페르소나]
연령대: 20대
성별: male
직업: 대학생
특성: 가성비 중시, 알고리즘 의존, 콘텐츠 충성도, 리뷰 신뢰

[참고 점수]
- 관심도: 75점

이 사용자가 유튜브에서 쓸 법한 1~2문장 반응을 그 플랫폼 톤 그대로 작성하세요.""",
    },
    {
        "title": "(기존 평가 톤) 동일 페르소나 — 일반 평가",
        "prompt": """당신은 소비자 반응 평가 전문가입니다.
다음 평가 결과의 이유를 페르소나 관점에서 설명해주세요.

[자극물]
유형: 가전/전자
내용: 무선 이어폰 ProAir 5세대 18만원, 노이즈캔슬링 신모델

[페르소나]
연령대: 20대
성별: male
직업: 대학생
특성: 가성비 중시

[평가 점수]
- 관심도: 70점

각 점수의 이유를 1문장씩 설명하고 한줄평을 작성하세요.""",
    },
]


def main():
    print("=" * 70)
    print("모델 비교 시연 - 기존 vs v2 (플랫폼 톤 학습)")
    print("=" * 70)

    print("\n[1/2] 베이스 모델 로드 (한 번만)...")
    base_model, tokenizer = load_base_model_and_tokenizer()

    print("\n[2/2] 두 LoRA 어댑터 동시 부착...")
    if EXISTING_LORA.exists():
        model = PeftModel.from_pretrained(base_model, str(EXISTING_LORA), adapter_name="existing")
        print(f"  existing 어댑터 적용: {EXISTING_LORA.name}")
    else:
        print(f"  [경고] 기존 LoRA 없음 - 베이스로 비교")
        model = base_model

    if V2_LORA.exists():
        if hasattr(model, "load_adapter"):
            model.load_adapter(str(V2_LORA), adapter_name="v2")
            print(f"  v2 어댑터 적용: {V2_LORA.name}")
        else:
            print("  [경고] v2 어댑터 불러올 수 없음")

    print()
    print("=" * 70)

    for scenario in SCENARIOS:
        print(f"\n### {scenario['title']}")
        print("-" * 70)

        # 기존 어댑터 활성화
        if hasattr(model, "set_adapter"):
            try:
                model.set_adapter("existing")
            except Exception:
                pass
        print("\n[기존 모델 응답]")
        existing_out = generate(model, tokenizer, scenario["prompt"])
        print(f"  {existing_out}")

        # v2 어댑터 활성화
        if hasattr(model, "set_adapter"):
            try:
                model.set_adapter("v2")
            except Exception:
                pass
        print("\n[v2 모델 응답]")
        v2_out = generate(model, tokenizer, scenario["prompt"])
        print(f"  {v2_out}")

        print()
        print("=" * 70)


if __name__ == "__main__":
    main()
