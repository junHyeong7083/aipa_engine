"""
AIPA 설문조사 시뮬레이션
RAG + AIPA-Eval 모델로 다양한 페르소나의 설문 응답 생성
"""
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from query import AIPARetriever

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = str(Path(__file__).parent.parent / "training" / "models" / "aipa-eval-lora")

# ========== 설문지 1: 유튜브 시청 영향 ==========
SURVEY_1 = {
    "title": "유튜브 시청이 미치는 영향 연구",
    "questions": [
        {"id": "SQ1", "text": "귀하의 성별은?", "type": "single", "options": ["남성", "여성"]},
        {"id": "SQ2", "text": "귀하의 연령은? (출생연도)", "type": "open_number"},
        {"id": "SQ3", "text": "최근 1개월 이내 이용경험이 있는 서비스를 모두 선택", "type": "multi",
         "options": ["유튜브", "검색엔진", "인스타그램", "페이스북", "커뮤니티", "블로그", "X(트위터)", "이 중 없음"]},
        {"id": "SQ4", "text": "유튜브를 통해 시청한 경험이 있는 카테고리를 모두 선택", "type": "multi",
         "options": ["영화", "음악", "예능", "정치·시사", "여행", "음식·먹방", "뷰티", "기타"]},
        {"id": "SQ5", "text": "정치·시사 유튜브 영상을 언제 시청했는지 최근 기준", "type": "single",
         "options": ["1개월 이내", "3개월 이내", "6개월 이내", "1년 이내", "기타"]},
        {"id": "SQ6", "text": "정치·시사 유튜브 채널 유형별 이용 정도 (1=전혀~7=매우 많이)", "type": "scale_7",
         "sub_items": ["방송사·언론사 채널", "정치 인플루언서 채널", "정치인·정당 채널"]},
        {"id": "SQ7", "text": "일주일 평균 정치·시사 유튜브 시청 빈도 (1=거의없음~5=매일)", "type": "scale_5",
         "sub_items": ["방송사·언론사 채널", "정치 인플루언서 채널", "정치인·정당 채널"]},
        {"id": "Q1", "text": "하루 평균 정치·시사 유튜브 시청 시간 (1=10분미만~5=2시간이상)", "type": "scale_5",
         "sub_items": ["방송사·언론사 채널", "정치 인플루언서 채널", "정치인·정당 채널"]},
        {"id": "Q2", "text": "정기적으로 시청하는 정치·시사 채널 수 (1=없음~5=7개이상)", "type": "scale_5",
         "sub_items": ["방송사·언론사 채널", "정치 인플루언서 채널", "정치인·정당 채널"]},
        {"id": "Q3", "text": "정치·시사 유튜브 영상 발견 경로", "type": "single",
         "options": ["구독 채널", "유튜브 추천 알고리즘", "검색", "지인 추천", "기타"]},
        {"id": "Q5", "text": "미디어 이용 행태 - 유튜브 정보 신뢰도 (1=전혀~7=매우 그렇다)", "type": "scale_7",
         "sub_items": ["유튜브 정보가 사실적이다", "유튜브 정보가 여론을 반영한다", "유튜브 정보가 종합적이다", "유튜브 정보가 정확하다", "유튜브 정보가 전문적이다"]},
        {"id": "Q6", "text": "이용 만족도 (1=전혀~7=매우 그렇다)", "type": "scale_7",
         "sub_items": ["유튜브 시청이 성취감을 준다", "유튜브 시청을 더 많이 하고 싶다", "전반적으로 만족스럽다", "유튜브 시청을 즐긴다", "시청 욕구를 충족시켜 준다"]},
        {"id": "Q9", "text": "본인의 정치적 성향 (1=매우진보~7=매우보수)", "type": "scale_7_single"},
        {"id": "Q10", "text": "정치 및 사회적 이슈 관심도 (1=전혀없음~7=매우있음)", "type": "scale_7_single"},
        {"id": "DQ2", "text": "최종 학력", "type": "single",
         "options": ["중졸 이하", "고졸", "대학 재학 중", "대졸", "대학원 재학 이상"]},
        {"id": "DQ5", "text": "월 평균 가구 소득", "type": "single",
         "options": ["100만원 미만", "100~200만원", "200~300만원", "300~400만원", "400~500만원", "500~600만원", "600만원 이상"]},
    ]
}

# 페르소나 5명 정의
PERSONAS = [
    {
        "name": "김민수",
        "age": 28, "gender": "남성", "occupation": "IT 개발자",
        "income": "400~500만원", "education": "대졸",
        "traits": ["테크 얼리어답터", "정치에 관심 많음", "진보 성향", "유튜브 헤비유저"],
    },
    {
        "name": "이수진",
        "age": 35, "gender": "여성", "occupation": "초등학교 교사",
        "income": "300~400만원", "education": "대학원 재학 이상",
        "traits": ["교육에 관심", "중도 성향", "뉴스 꼼꼼히 챙김", "미디어 리터러시 높음"],
    },
    {
        "name": "박준혁",
        "age": 52, "gender": "남성", "occupation": "자영업자",
        "income": "200~300만원", "education": "고졸",
        "traits": ["보수 성향", "유튜브로 뉴스 시청", "정치 유튜브 구독 많음", "경제에 관심"],
    },
    {
        "name": "최예린",
        "age": 22, "gender": "여성", "occupation": "대학생",
        "income": "100만원 미만", "education": "대학 재학 중",
        "traits": ["SNS 활발", "정치 관심 낮음", "예능·뷰티 위주 시청", "알고리즘 의존"],
    },
    {
        "name": "정태호",
        "age": 45, "gender": "남성", "occupation": "공무원",
        "income": "500~600만원", "education": "대졸",
        "traits": ["중도보수 성향", "균형잡힌 시각 추구", "다양한 채널 시청", "팩트체크 습관"],
    },
]

print("=" * 60)
print("AIPA 설문조사 시뮬레이션")
print(f"설문: {SURVEY_1['title']}")
print(f"페르소나: {len(PERSONAS)}명")
print("=" * 60)

# RAG 컨텍스트
print("\n[1/3] RAG 컨텍스트 검색...")
retriever = AIPARetriever()
context = retriever.build_context("정치 시사 유튜브 시청 미디어 이용 행태 20대 30대 40대 50대")
print(context[:300] + "...")

# 모델 로드
print("\n[2/3] AIPA-Eval 모델 로드...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# 설문 시뮬레이션
print("\n[3/3] 설문 응답 생성...")
results = []

for i, persona in enumerate(PERSONAS):
    print(f"\n{'='*50}")
    print(f"[{i+1}/{len(PERSONAS)}] {persona['name']} ({persona['age']}세 {persona['gender']}, {persona['occupation']})")
    print(f"  특성: {', '.join(persona['traits'])}")

    questions_text = ""
    for q in SURVEY_1["questions"]:
        questions_text += f"\n- {q['id']}: {q['text']}"
        if q["type"] == "single" and "options" in q:
            questions_text += f" (선택지: {', '.join(q['options'])})"
        elif q["type"] == "multi" and "options" in q:
            questions_text += f" (복수선택: {', '.join(q['options'])})"
        elif "sub_items" in q:
            questions_text += f" (하위항목: {', '.join(q['sub_items'])})"

    prompt = f"""당신은 설문조사 응답자 시뮬레이터입니다.

아래 페르소나의 관점에서 설문조사에 응답해주세요.
시장 데이터를 참고하여 현실적으로 응답하세요.

{context}

[페르소나]
이름: {persona['name']}
나이: {persona['age']}세
성별: {persona['gender']}
직업: {persona['occupation']}
소득: {persona['income']}
학력: {persona['education']}
특성: {', '.join(persona['traits'])}

[설문지: {SURVEY_1['title']}]
{questions_text}

[응답 규칙]
- 모든 질문에 대해 이 페르소나의 관점에서 현실적으로 응답하세요.
- JSON 형식으로 응답하세요. 각 질문 ID를 키로, 응답값을 값으로.
- scale_7 문항: 반드시 1~7 범위의 정수로 응답 (1=전혀 아니다, 7=매우 그렇다)
- scale_5 문항: 반드시 1~5 범위의 정수로 응답
- scale_7_single 문항: 반드시 1~7 범위의 정수로 응답
- single 문항: 선택한 옵션 텍스트 그대로 응답
- multi 문항: 선택한 옵션들을 리스트로 응답
- open_number 문항: 숫자로 응답
- sub_items가 있는 경우 딕셔너리로 응답

[응답 JSON 형식 예시]
```json
{{
  "SQ1": "남성",
  "SQ2": 28,
  "SQ3": ["유튜브", "검색엔진"],
  "SQ4": ["음악", "정치·시사"],
  "SQ5": "1개월 이내",
  "SQ6": {{"방송사·언론사 채널": 5, "정치 인플루언서 채널": 3, "정치인·정당 채널": 2}},
  "SQ7": {{"방송사·언론사 채널": 3, "정치 인플루언서 채널": 2, "정치인·정당 채널": 1}},
  "Q1": {{"방송사·언론사 채널": 2, "정치 인플루언서 채널": 1, "정치인·정당 채널": 1}},
  "Q2": {{"방송사·언론사 채널": 3, "정치 인플루언서 채널": 1, "정치인·정당 채널": 1}},
  "Q3": "유튜브 추천 알고리즘",
  "Q5": {{"유튜브 정보가 사실적이다": 4, "유튜브 정보가 여론을 반영한다": 5, "유튜브 정보가 종합적이다": 3, "유튜브 정보가 정확하다": 3, "유튜브 정보가 전문적이다": 3}},
  "Q6": {{"유튜브 시청이 성취감을 준다": 4, "유튜브 시청을 더 많이 하고 싶다": 5, "전반적으로 만족스럽다": 5, "유튜브 시청을 즐긴다": 6, "시청 욕구를 충족시켜 준다": 5}},
  "Q9": 3,
  "Q10": 6,
  "DQ2": "대졸",
  "DQ5": "400~500만원"
}}
```
위 예시는 형식 참고용입니다. 페르소나에 맞는 값을 사용하세요."""

    inputs = tokenizer(
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1,
        )

    result_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    try:
        parsed = json.loads(result_text)

        # 후처리: 스케일 값을 올바른 범위로 클램핑
        scale_ranges = {}
        for q in SURVEY_1["questions"]:
            if q["type"] == "scale_7" or q["type"] == "scale_7_single":
                scale_ranges[q["id"]] = (1, 7)
            elif q["type"] == "scale_5":
                scale_ranges[q["id"]] = (1, 5)

        for qid, (lo, hi) in scale_ranges.items():
            if qid not in parsed:
                continue
            val = parsed[qid]
            if isinstance(val, dict):
                # sub_items가 있는 경우 각 값을 클램핑
                for sub_key, sub_val in val.items():
                    if isinstance(sub_val, (int, float)):
                        val[sub_key] = max(lo, min(hi, int(sub_val)))
                parsed[qid] = val
            elif isinstance(val, (int, float)):
                parsed[qid] = max(lo, min(hi, int(val)))

        results.append({"persona": persona, "responses": parsed})
        print(f"  -> JSON 파싱 성공")
        # 주요 응답 미리보기
        for key in ["SQ3", "Q3", "Q9", "Q10"]:
            if key in parsed:
                print(f"     {key}: {parsed[key]}")
    except json.JSONDecodeError:
        results.append({"persona": persona, "responses_raw": result_text})
        print(f"  -> JSON 파싱 실패, 원문 저장")
        print(f"     출력: {result_text[:200]}...")

# 결과 저장
output_path = Path(__file__).parent / "survey_results_1.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"설문 시뮬레이션 완료!")
print(f"응답자: {len(results)}명")
print(f"결과 저장: {output_path}")
print(f"{'='*60}")
