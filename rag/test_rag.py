"""
AIPA RAG 테스트
검색 + AIPA-Eval 모델 추론 통합 테스트
"""
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from query import AIPARetriever

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = str(Path(__file__).parent.parent / "training" / "models" / "aipa-eval-lora")

print("=" * 50)
print("AIPA RAG + Eval 통합 테스트")
print("=" * 50)

# 1. RAG 검색
print("\n[1/3] RAG 검색...")
retriever = AIPARetriever()

test_query = "20대 여성 건강식품 유기농 샐러드"
context = retriever.build_context(test_query)
print(f"\n검색 쿼리: {test_query}")
print(f"\n--- 검색된 컨텍스트 ---")
print(context)
print("---")

# 2. 모델 로드
print("\n[2/3] AIPA-Eval 모델 로드...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# 3. RAG 컨텍스트 + 평가 프롬프트
print("\n[3/3] RAG 강화 추론...")
prompt = f"""당신은 소비자 반응 평가 전문가입니다.

다음 자극물에 대해 주어진 페르소나의 관점에서 평가해주세요.
아래 시장 데이터를 참고하여 현실적인 평가를 해주세요.

{context}

[자극물]
유형: 광고
내용: 프리미엄 유기농 샐러드 배달 서비스 - 매일 신선한 재료로 만든 맞춤형 샐러드

[페르소나]
연령대: 20대
성별: 여성
직업: IT 개발자
소득: 중상
특성: 건강관심, 바쁜일상, SNS활발

[평가 축]
구매의향, 브랜드호감도, 가격적절성, 재구매의사, 추천의향

각 축에 대해 1-10점 점수와 근거를 JSON으로 응답하세요."""

inputs = tokenizer(
    f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.1,
    )

result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("\n=== RAG 강화 AIPA-Eval 출력 ===")
print(result)

try:
    parsed = json.loads(result)
    print("\n=== JSON 파싱 성공 ===")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))
except json.JSONDecodeError:
    print("\n[!] JSON 파싱 실패")
