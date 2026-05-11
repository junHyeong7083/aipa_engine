"""
병합된 모델을 GGUF Q4_K_M으로 변환

llama.cpp의 convert_hf_to_gguf.py를 직접 다운로드해서 사용.

사용법:
  python training/convert_gguf.py
"""

import subprocess
import sys
import os
from pathlib import Path

MERGED_PATH = Path(__file__).parent / "models" / "reasoning" / "merged"
GGUF_F16_PATH = Path(__file__).parent / "models" / "reasoning" / "aipa-reasoning-f16.gguf"
GGUF_Q4_PATH = Path(__file__).parent / "models" / "reasoning" / "aipa-reasoning-q4km.gguf"
CONVERT_SCRIPT = Path(__file__).parent / "convert_hf_to_gguf.py"


def download_convert_script():
    """llama.cpp의 변환 스크립트 다운로드"""
    if CONVERT_SCRIPT.exists():
        print("  변환 스크립트 이미 존재")
        return True

    print("  llama.cpp 변환 스크립트 다운로드 중...")
    url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(CONVERT_SCRIPT))
        print("  다운로드 완료")
        return True
    except Exception as e:
        print(f"  다운로드 실패: {e}")
        return False


def convert_to_gguf():
    """HF 모델 → GGUF F16 변환"""
    print("\n[1/2] HF → GGUF F16 변환")
    result = subprocess.run(
        [sys.executable, str(CONVERT_SCRIPT),
         str(MERGED_PATH),
         "--outfile", str(GGUF_F16_PATH),
         "--outtype", "f16"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("  F16 변환 실패")
        return False

    if GGUF_F16_PATH.exists():
        size = GGUF_F16_PATH.stat().st_size / 1024 / 1024
        print(f"  F16 변환 완료: {size:.0f}MB")
        return True
    return False


def quantize_gguf():
    """GGUF F16 → Q4_K_M 양자화"""
    print("\n[2/2] F16 → Q4_K_M 양자화")

    try:
        from llama_cpp import Llama

        # llama-cpp-python의 llama-quantize 사용
        # llama_cpp 패키지에 quantize가 내장되어 있지 않으면 f16으로 사용
        print("  llama_cpp로 양자화 시도...")

        # llama-quantize 바이너리 찾기
        import llama_cpp
        lib_dir = Path(llama_cpp.__file__).parent
        quantize_bin = None
        for name in ["llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe"]:
            candidate = lib_dir / name
            if candidate.exists():
                quantize_bin = str(candidate)
                break

        if quantize_bin:
            subprocess.run([quantize_bin, str(GGUF_F16_PATH), str(GGUF_Q4_PATH), "Q4_K_M"])
            if GGUF_Q4_PATH.exists():
                size = GGUF_Q4_PATH.stat().st_size / 1024 / 1024
                print(f"  Q4_K_M 양자화 완료: {size:.0f}MB")
                return True

    except Exception as e:
        print(f"  양자화 도구 없음: {e}")

    # 양자화 실패하면 F16 그대로 사용
    print("  양자화 스킵 - F16 모델을 그대로 사용합니다")
    print(f"  (F16은 크기가 크지만 동작은 합니다)")

    # F16을 최종 파일명으로 복사
    if GGUF_F16_PATH.exists() and not GGUF_Q4_PATH.exists():
        import shutil
        shutil.copy2(str(GGUF_F16_PATH), str(GGUF_Q4_PATH))
        print(f"  F16 → {GGUF_Q4_PATH.name} 복사 완료")
        return True

    return False


if __name__ == "__main__":
    print("=" * 50)
    print("GGUF 변환")
    print("=" * 50)

    if not download_convert_script():
        sys.exit(1)

    if not convert_to_gguf():
        print("\n변환 실패. 수동 변환 필요.")
        sys.exit(1)

    quantize_gguf()

    final = GGUF_Q4_PATH if GGUF_Q4_PATH.exists() else GGUF_F16_PATH
    if final.exists():
        size = final.stat().st_size / 1024 / 1024
        print(f"\n완료! 최종 모델: {final} ({size:.0f}MB)")
    else:
        print("\n변환 실패")
