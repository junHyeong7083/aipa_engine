"""
병합된 모델을 llama-cpp-python으로 직접 로드 테스트.
GGUF 변환 대신, llama-cpp-python이 HF 모델을 직접 사용할 수 있는지 확인.

안 되면 ctransformers 또는 직접 PyTorch로 서빙.
"""

import sys
from pathlib import Path

MERGED_PATH = Path(__file__).parent / "models" / "reasoning" / "merged"
GGUF_PATH = Path(__file__).parent / "models" / "reasoning" / "aipa-reasoning.gguf"


def try_transformers_to_gguf():
    """transformers 모델을 직접 GGUF로 변환 (gguf 라이브러리 사용)"""
    print("방법 1: gguf 라이브러리로 직접 변환")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from gguf import GGUFWriter, GGMLQuantizationType
        import numpy as np

        print("  모델 로드 중...")
        model = AutoModelForCausalLM.from_pretrained(
            str(MERGED_PATH),
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(str(MERGED_PATH))

        config = model.config
        print(f"  모델: {config.architectures}")
        print(f"  히든 크기: {config.hidden_size}")
        print(f"  레이어: {config.num_hidden_layers}")
        print(f"  파라미터: {sum(p.numel() for p in model.parameters()):,}")

        # state_dict를 numpy로 변환해서 저장
        print("  GGUF 파일 생성 중...")
        writer = GGUFWriter(str(GGUF_PATH), "qwen2")

        # 모델 메타데이터
        writer.add_block_count(config.num_hidden_layers)
        writer.add_context_length(config.max_position_embeddings)
        writer.add_embedding_length(config.hidden_size)
        writer.add_feed_forward_length(config.intermediate_size)
        writer.add_head_count(config.num_attention_heads)
        writer.add_head_count_kv(config.num_key_value_heads)
        writer.add_layer_norm_rms_eps(config.rms_norm_eps)
        writer.add_rope_freq_base(config.rope_theta)
        writer.add_vocab_size(config.vocab_size)

        # 토큰 데이터
        print("  토크나이저 변환 중...")
        vocab = tokenizer.get_vocab()
        tokens = [b""] * len(vocab)
        scores = [0.0] * len(vocab)
        token_types = [0] * len(vocab)  # NORMAL

        for text, idx in vocab.items():
            if idx < len(tokens):
                tokens[idx] = text.encode("utf-8", errors="replace")
                scores[idx] = -idx  # 기본 점수

        writer.add_tokenizer_model("gpt2")
        writer.add_token_list(tokens)
        writer.add_token_scores(scores)
        writer.add_token_types(token_types)

        # 가중치 텐서들
        print("  가중치 변환 중 (F16)...")
        state_dict = model.state_dict()
        for name, tensor in state_dict.items():
            # GGUF 이름 변환
            gguf_name = name.replace("model.", "")
            data = tensor.cpu().numpy().astype(np.float16)

            writer.add_tensor(gguf_name, data)

        print("  파일 저장 중...")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        if GGUF_PATH.exists():
            size = GGUF_PATH.stat().st_size / 1024 / 1024
            print(f"\n  GGUF 변환 완료! {GGUF_PATH} ({size:.0f}MB)")
            return True

    except Exception as e:
        print(f"  실패: {e}")
        import traceback
        traceback.print_exc()

    return False


def try_load_test():
    """변환된 GGUF 로드 테스트"""
    if not GGUF_PATH.exists():
        print("\nGGUF 파일 없음, 로드 테스트 스킵")
        return

    print("\n로드 테스트...")
    try:
        from llama_cpp import Llama
        llm = Llama(
            model_path=str(GGUF_PATH),
            n_ctx=512,
            n_threads=2,
            verbose=False,
        )
        output = llm("Hello", max_tokens=10)
        print(f"  로드 성공! 테스트 출력: {output['choices'][0]['text'][:50]}")
    except Exception as e:
        print(f"  로드 실패: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("GGUF 변환 (간단 버전)")
    print("=" * 50)

    if try_transformers_to_gguf():
        try_load_test()
    else:
        print("\n변환 실패. 대안: HuggingFace에서 이미 변환된 GGUF 다운로드")
        print("  huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf")
        print("  → 이 파일을 베이스로 사용하고, 파인튜닝은 PyTorch로 서빙")
