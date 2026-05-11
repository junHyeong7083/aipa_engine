"""
LoRA 어댑터를 베이스 모델에 병합 → GGUF 변환

1단계: LoRA + 베이스 모델 병합 (merged 폴더에 저장)
2단계: GGUF 변환 (llama.cpp 사용)

사용법:
  python training/merge_and_export.py
"""

import os
import sys
import shutil
from pathlib import Path

# 경로 설정
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = Path(__file__).parent / "models" / "reasoning" / "aipa-reasoning-lora"
MERGED_PATH = Path(__file__).parent / "models" / "reasoning" / "merged"
GGUF_PATH = Path(__file__).parent / "models" / "reasoning" / "aipa-reasoning.gguf"


def step1_merge():
    """LoRA 어댑터를 베이스 모델에 병합"""
    print("=" * 50)
    print("[1/2] LoRA 병합")
    print("=" * 50)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"  베이스 모델: {BASE_MODEL}")
    print(f"  어댑터: {ADAPTER_PATH}")

    # 베이스 모델 로드 (FP16)
    print("  베이스 모델 로드 중...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # LoRA 어댑터 로드 + 병합
    print("  LoRA 어댑터 병합 중...")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
    model = model.merge_and_unload()

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH), trust_remote_code=True)

    # 병합된 모델 저장
    print(f"  저장 중: {MERGED_PATH}")
    MERGED_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_PATH), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_PATH))

    size_mb = sum(
        os.path.getsize(os.path.join(str(MERGED_PATH), f))
        for f in os.listdir(str(MERGED_PATH))
    ) / 1024 / 1024
    print(f"  병합 완료! ({size_mb:.0f}MB)")


def step2_convert_gguf():
    """병합된 모델을 GGUF 형식으로 변환"""
    print("\n" + "=" * 50)
    print("[2/2] GGUF 변환")
    print("=" * 50)

    # llama.cpp의 convert 스크립트 확인
    # pip install llama-cpp-python 으로 설치되어 있어야 함
    try:
        import subprocess

        # 방법 1: llama-cpp-python의 변환 도구 사용
        # transformers에서 직접 GGUF로 변환
        print("  GGUF 변환 중...")

        # gguf 패키지로 직접 변환
        result = subprocess.run(
            [
                sys.executable, "-m", "llama_cpp.convert",
                "--outfile", str(GGUF_PATH),
                "--outtype", "q4_k_m",
                str(MERGED_PATH),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  llama_cpp.convert 실패, 대체 방법 시도...")
            # 대체 방법: gguf 패키지 직접 사용
            _convert_with_gguf_lib()

    except Exception as e:
        print(f"  자동 변환 실패: {e}")
        print(f"\n  수동 변환 방법:")
        print(f"  1. pip install gguf")
        print(f"  2. huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        print(f"     또는")
        print(f"  3. llama.cpp 빌드 후:")
        print(f"     python llama.cpp/convert_hf_to_gguf.py {MERGED_PATH} --outtype q4_k_m --outfile {GGUF_PATH}")
        return False

    if GGUF_PATH.exists():
        size_mb = GGUF_PATH.stat().st_size / 1024 / 1024
        print(f"\n  GGUF 변환 완료!")
        print(f"  파일: {GGUF_PATH}")
        print(f"  크기: {size_mb:.0f}MB")
        print(f"\n  다음 단계: Cloud Run 배포")
        return True

    return False


def _convert_with_gguf_lib():
    """gguf 라이브러리로 직접 변환 시도"""
    import subprocess
    # convert_hf_to_gguf.py 스크립트 찾기
    scripts = [
        "convert_hf_to_gguf.py",
        "llama.cpp/convert_hf_to_gguf.py",
    ]

    for script in scripts:
        if os.path.exists(script):
            subprocess.run([
                sys.executable, script,
                str(MERGED_PATH),
                "--outtype", "q4_k_m",
                "--outfile", str(GGUF_PATH),
            ])
            return

    # 못 찾으면 pip로 시도
    merged_str = str(MERGED_PATH).replace("\\", "/")
    subprocess.run([
        sys.executable, "-c",
        f'import transformers; from transformers import AutoModelForCausalLM; '
        f'model = AutoModelForCausalLM.from_pretrained("{merged_str}"); '
        f'model.save_pretrained("{merged_str}", safe_serialization=False)'
    ])
    print("  PyTorch 모델로 저장 완료. 수동 GGUF 변환 필요.")


if __name__ == "__main__":
    step1_merge()

    try:
        step2_convert_gguf()
    except Exception as e:
        print(f"\n  GGUF 변환 스킵: {e}")
        print(f"  병합된 모델은 {MERGED_PATH}에 저장됨")
        print(f"  수동으로 GGUF 변환 후 배포하세요")
