"""
설문 응답 학습 데이터 생성

Claude API로 "이 페르소나가 이 설문에 답하면?" 데이터를 생성.
인간 응답 편향(중심화 경향, 묵종 편향, 사회적 바람직성 등)이 반영된 데이터.

사용법:
    python data/scripts/generate_survey_data.py --count 1000
    python data/scripts/generate_survey_data.py --count 500 --concurrent 5
"""

import asyncio
import json
import re
import random
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import anthropic

# ─────────────────────────────────────────────
# 설문 질문 풀 (10개 카테고리 × 10~15개 질문)
# ─────────────────────────────────────────────

SURVEY_QUESTIONS = {
    "소비/쇼핑": [
        {"q": "온라인 쇼핑과 오프라인 쇼핑 중 어느 것을 더 선호하시나요?", "choices": ["온라인 쇼핑", "오프라인 쇼핑", "둘 다 비슷", "상황에 따라 다름"]},
        {"q": "월 평균 온라인 쇼핑 횟수는?", "choices": ["거의 안 함", "1~2회", "3~5회", "6~10회", "10회 이상"]},
        {"q": "물건 구매 시 가장 중요하게 보는 것은?", "choices": ["가격", "품질", "브랜드", "후기/리뷰", "디자인"]},
        {"q": "충동구매를 자주 하는 편인가요?", "choices": ["매우 그렇다", "그렇다", "보통", "아니다", "전혀 아니다"]},
        {"q": "정기 구독 서비스를 이용하고 있나요?", "choices": ["3개 이상", "1~2개", "이전에 했지만 해지", "관심 있지만 안 함", "관심 없음"]},
        {"q": "중고 거래 플랫폼을 이용하시나요?", "choices": ["자주 이용", "가끔 이용", "거의 안 함", "한 번도 안 해봄"]},
        {"q": "새 제품 출시 시 얼마나 빨리 구매하시나요?", "choices": ["출시 당일 구매", "1주일 이내", "한달 이내", "충분히 후기 확인 후", "거의 안 삼"]},
        {"q": "해외직구 경험이 있으신가요?", "choices": ["자주 함", "가끔 함", "한두 번 해봄", "해본 적 없음"]},
        {"q": "배달앱 주문 빈도는?", "choices": ["거의 매일", "주 3~4회", "주 1~2회", "월 1~2회", "거의 안 함"]},
        {"q": "쿠폰이나 할인 정보를 적극적으로 찾아보시나요?", "choices": ["매우 그렇다", "그렇다", "보통", "아니다", "전혀 아니다"]},
    ],
    "미디어/콘텐츠": [
        {"q": "하루 평균 스마트폰 사용 시간은?", "choices": ["1시간 미만", "1~3시간", "3~5시간", "5~7시간", "7시간 이상"]},
        {"q": "뉴스는 주로 어디서 접하시나요?", "choices": ["포털 사이트", "SNS", "TV", "신문/잡지", "유튜브"]},
        {"q": "유튜브 하루 시청 시간은?", "choices": ["안 봄", "30분 미만", "30분~1시간", "1~3시간", "3시간 이상"]},
        {"q": "넷플릭스 등 OTT 서비스를 이용하시나요?", "choices": ["2개 이상 구독", "1개 구독", "무료 체험만", "이전에 했지만 해지", "관심 없음"]},
        {"q": "SNS에서 가장 많이 사용하는 플랫폼은?", "choices": ["인스타그램", "유튜브", "틱톡", "트위터/X", "페이스북", "카카오스토리", "안 함"]},
        {"q": "팟캐스트나 오디오 콘텐츠를 듣나요?", "choices": ["자주 듣는다", "가끔 듣는다", "거의 안 듣는다", "들어본 적 없다"]},
        {"q": "웹소설/웹툰을 읽으시나요?", "choices": ["매일 본다", "주 2~3회", "가끔", "거의 안 봄"]},
        {"q": "TV 예능/드라마를 주로 어떻게 보시나요?", "choices": ["실시간 TV", "OTT 다시보기", "유튜브 클립", "안 봄"]},
        {"q": "게임을 하시나요?", "choices": ["매일 한다", "주 2~3회", "가끔", "거의 안 함", "전혀 안 함"]},
        {"q": "디지털 콘텐츠(이모티콘, 음원 등)에 돈을 쓰시나요?", "choices": ["월 1만원 이상", "월 5천원 정도", "가끔 구매", "거의 안 씀", "전혀 안 씀"]},
    ],
    "건강/웰빙": [
        {"q": "규칙적으로 운동을 하시나요?", "choices": ["매일", "주 3~5회", "주 1~2회", "거의 안 함", "전혀 안 함"]},
        {"q": "건강보조식품(비타민 등)을 복용하시나요?", "choices": ["매일 복용", "가끔 복용", "이전에 했지만 중단", "관심은 있음", "관심 없음"]},
        {"q": "식단 관리를 하시나요?", "choices": ["철저히 함", "어느 정도 신경 씀", "가끔 신경 씀", "거의 안 함", "전혀 안 함"]},
        {"q": "스트레스 해소 방법은?", "choices": ["운동", "수면", "취미활동", "술/음식", "대화/상담", "SNS/영상"]},
        {"q": "수면 시간은 보통 얼마나 되나요?", "choices": ["5시간 미만", "5~6시간", "6~7시간", "7~8시간", "8시간 이상"]},
        {"q": "정기 건강검진을 받으시나요?", "choices": ["매년 받음", "2년에 한 번", "비정기적", "거의 안 받음", "한 번도 안 받음"]},
        {"q": "유기농/친환경 식품을 선호하시나요?", "choices": ["매우 그렇다", "그렇다", "보통", "아니다", "전혀 아니다"]},
        {"q": "정신건강(우울, 불안)에 대해 관심을 가지고 있나요?", "choices": ["매우 관심 있음", "관심 있음", "보통", "별로 없음", "전혀 없음"]},
        {"q": "하루 물 섭취량은?", "choices": ["2L 이상", "1~2L", "0.5~1L", "0.5L 미만", "잘 모르겠음"]},
        {"q": "체중 관리에 신경 쓰시나요?", "choices": ["매우 그렇다", "그렇다", "보통", "아니다", "전혀 아니다"]},
    ],
    "직장/커리어": [
        {"q": "현재 직장에 만족하시나요?", "choices": ["매우 만족", "만족", "보통", "불만족", "매우 불만족"]},
        {"q": "이직을 고려하고 있나요?", "choices": ["적극적으로 준비 중", "관심 있음", "기회가 오면", "현재 만족", "해당 없음"]},
        {"q": "재택근무를 선호하시나요?", "choices": ["매우 그렇다", "그렇다", "보통", "아니다", "매우 아니다"]},
        {"q": "워라밸(일과 삶의 균형)에 만족하시나요?", "choices": ["매우 만족", "만족", "보통", "불만족", "매우 불만족"]},
        {"q": "자기계발에 시간을 투자하시나요?", "choices": ["매일", "주 2~3회", "주 1회", "거의 안 함", "전혀 안 함"]},
        {"q": "부업/투잡에 관심이 있으신가요?", "choices": ["이미 하고 있음", "준비 중", "관심 있음", "관심 없음"]},
        {"q": "AI가 내 직업을 대체할 수 있다고 생각하시나요?", "choices": ["매우 그렇다", "그렇다", "보통", "아니다", "전혀 아니다"]},
        {"q": "직장 내 인간관계에 만족하시나요?", "choices": ["매우 만족", "만족", "보통", "불만족", "매우 불만족"]},
        {"q": "현재 연봉에 만족하시나요?", "choices": ["매우 만족", "만족", "보통", "불만족", "매우 불만족"]},
        {"q": "회사의 복지 제도에 만족하시나요?", "choices": ["매우 만족", "만족", "보통", "불만족", "매우 불만족"]},
    ],
    "기술/디지털": [
        {"q": "새로운 기술/앱을 얼마나 빨리 사용해보시나요?", "choices": ["출시 즉시", "주변 반응 보고", "충분히 검증 후", "필요할 때만", "거의 안 씀"]},
        {"q": "AI 서비스(ChatGPT 등)를 사용해보셨나요?", "choices": ["자주 사용", "가끔 사용", "한두 번 해봄", "알고는 있지만 안 해봄", "모름"]},
        {"q": "스마트홈 기기를 사용하시나요?", "choices": ["3개 이상", "1~2개", "관심 있지만 없음", "관심 없음"]},
        {"q": "개인정보 보호에 대해 얼마나 신경 쓰시나요?", "choices": ["매우 신경 씀", "신경 쓰는 편", "보통", "별로 안 씀", "전혀 안 씀"]},
        {"q": "무인 매장(키오스크)이 편하신가요?", "choices": ["매우 편함", "편한 편", "보통", "불편한 편", "매우 불편함"]},
        {"q": "전자결제(카드/페이)와 현금 중 선호하는 것은?", "choices": ["전자결제만", "전자결제 위주", "반반", "현금 위주", "현금만"]},
        {"q": "자율주행 자동차를 탈 의향이 있으신가요?", "choices": ["바로 타겠다", "기술이 더 발전하면", "잘 모르겠다", "좀 무섭다", "절대 안 탐"]},
        {"q": "메타버스/가상현실에 관심이 있으신가요?", "choices": ["매우 관심", "관심 있음", "보통", "별로 없음", "전혀 없음"]},
        {"q": "코딩/프로그래밍을 배워본 적 있나요?", "choices": ["현업에서 사용", "배운 적 있음", "독학 중", "관심은 있음", "관심 없음"]},
        {"q": "로봇이 서비스하는 카페/식당에 가보시겠어요?", "choices": ["꼭 가보고 싶다", "관심 있다", "상관없다", "별로다", "절대 안 감"]},
    ],
    "사회/정치": [
        {"q": "투표에 꼭 참여하시나요?", "choices": ["항상 참여", "대부분 참여", "가끔 참여", "거의 안 함", "해당 없음"]},
        {"q": "사회 이슈에 대해 얼마나 관심이 있으신가요?", "choices": ["매우 관심", "관심 있음", "보통", "별로 없음", "전혀 없음"]},
        {"q": "기부나 봉사활동을 하시나요?", "choices": ["정기적으로 함", "가끔 함", "해본 적 있음", "관심은 있음", "관심 없음"]},
        {"q": "양성평등이 잘 이루어지고 있다고 생각하시나요?", "choices": ["매우 그렇다", "그렇다", "보통", "아니다", "매우 아니다"]},
        {"q": "현재 한국 경제 상황을 어떻게 보시나요?", "choices": ["매우 좋음", "좋은 편", "보통", "나쁜 편", "매우 나쁨"]},
        {"q": "세금을 더 내더라도 복지가 늘어나야 한다고 생각하시나요?", "choices": ["매우 그렇다", "그렇다", "보통", "아니다", "매우 아니다"]},
        {"q": "외국인 노동자에 대해 어떻게 생각하시나요?", "choices": ["더 필요하다", "현재 적절", "줄여야 한다", "잘 모르겠다"]},
        {"q": "사형제도에 대해 어떻게 생각하시나요?", "choices": ["유지해야 한다", "조건부 유지", "폐지해야 한다", "잘 모르겠다"]},
        {"q": "정치인을 신뢰하시나요?", "choices": ["매우 신뢰", "어느 정도 신뢰", "보통", "별로 안 함", "전혀 안 함"]},
        {"q": "뉴스 미디어를 신뢰하시나요?", "choices": ["매우 신뢰", "어느 정도 신뢰", "보통", "별로 안 함", "전혀 안 함"]},
    ],
    "교육": [
        {"q": "온라인 강의를 들어본 적 있나요?", "choices": ["자주 듣는다", "가끔 듣는다", "한두 번 들어봄", "관심 있지만 안 해봄", "관심 없음"]},
        {"q": "자녀 교육비 지출이 부담되시나요?", "choices": ["매우 부담", "부담됨", "보통", "괜찮음", "해당 없음"]},
        {"q": "평생교육의 필요성을 느끼시나요?", "choices": ["매우 그렇다", "그렇다", "보통", "아니다", "전혀 아니다"]},
        {"q": "외국어 학습에 관심이 있으신가요?", "choices": ["현재 공부 중", "계획 있음", "관심은 있음", "별로 없음", "전혀 없음"]},
        {"q": "대학 교육이 취업에 필수적이라고 생각하시나요?", "choices": ["매우 그렇다", "그렇다", "보통", "아니다", "매우 아니다"]},
        {"q": "자격증 취득에 관심이 있으신가요?", "choices": ["현재 준비 중", "계획 있음", "관심 있음", "별로 없음", "전혀 없음"]},
        {"q": "독서를 얼마나 하시나요?", "choices": ["월 4권 이상", "월 1~3권", "분기 1~2권", "거의 안 함", "전혀 안 함"]},
        {"q": "AI를 활용한 교육에 긍정적이신가요?", "choices": ["매우 긍정", "긍정적", "보통", "부정적", "매우 부정"]},
    ],
    "주거/부동산": [
        {"q": "내 집 마련 계획이 있으신가요?", "choices": ["이미 있음", "5년 내 계획", "계획은 있지만 어려움", "전세/월세 만족", "해당 없음"]},
        {"q": "현재 주거 형태에 만족하시나요?", "choices": ["매우 만족", "만족", "보통", "불만족", "매우 불만족"]},
        {"q": "부동산 투자에 관심이 있으신가요?", "choices": ["이미 투자 중", "관심 많음", "관심 있음", "별로 없음", "전혀 없음"]},
        {"q": "주거비(월세/대출)가 부담되시나요?", "choices": ["매우 부담", "부담됨", "보통", "괜찮음", "해당 없음"]},
        {"q": "살고 싶은 지역은?", "choices": ["서울 강남권", "서울 기타", "수도권 신도시", "지방 대도시", "시골/자연"]},
        {"q": "1인 가구용 소형 주택에 관심이 있으신가요?", "choices": ["매우 관심", "관심 있음", "보통", "별로 없음", "해당 없음"]},
        {"q": "공유 주거(셰어하우스)에 거주할 의향이 있나요?", "choices": ["예", "조건에 따라", "아니오", "잘 모르겠음"]},
    ],
    "환경": [
        {"q": "분리수거를 철저히 하시나요?", "choices": ["매우 철저히", "하는 편", "보통", "가끔", "거의 안 함"]},
        {"q": "친환경 제품에 추가 비용을 지불할 의향이 있나요?", "choices": ["10% 이상 더 내겠다", "5~10% 정도", "같은 가격이면", "추가 비용 싫다", "관심 없다"]},
        {"q": "전기차 구매에 관심이 있으신가요?", "choices": ["이미 보유", "구매 계획", "관심 있음", "아직 이름", "관심 없음"]},
        {"q": "일회용품 사용을 줄이려고 노력하시나요?", "choices": ["매우 노력함", "노력하는 편", "보통", "별로", "전혀 안 함"]},
        {"q": "기후변화가 심각하다고 생각하시나요?", "choices": ["매우 심각", "심각함", "보통", "별로", "전혀 아님"]},
        {"q": "텀블러/에코백을 사용하시나요?", "choices": ["항상 사용", "자주 사용", "가끔", "거의 안 함", "전혀 안 함"]},
        {"q": "채식에 관심이 있으신가요?", "choices": ["이미 실천 중", "관심 많음", "가끔 시도", "별로 없음", "전혀 없음"]},
    ],
    "가족/관계": [
        {"q": "결혼에 대해 어떻게 생각하시나요?", "choices": ["꼭 해야 한다", "하면 좋다", "선택 사항", "안 해도 된다", "하고 싶지 않다"]},
        {"q": "자녀 계획이 있으신가요?", "choices": ["이미 있음", "계획 있음", "아직 모르겠음", "계획 없음", "해당 없음"]},
        {"q": "반려동물을 키우고 있나요?", "choices": ["키우고 있음", "키울 계획", "관심 있음", "관심 없음", "키울 수 없는 환경"]},
        {"q": "부모님과의 관계에 만족하시나요?", "choices": ["매우 만족", "만족", "보통", "불만족", "매우 불만족"]},
        {"q": "명절 스트레스가 있으신가요?", "choices": ["매우 크다", "있는 편", "보통", "별로 없다", "전혀 없다"]},
        {"q": "가족과 함께하는 시간이 충분하다고 느끼시나요?", "choices": ["매우 충분", "충분한 편", "보통", "부족함", "매우 부족"]},
        {"q": "노후 준비를 하고 있으신가요?", "choices": ["체계적으로 준비 중", "어느 정도 함", "해야 하는데 못 함", "아직 이름", "해당 없음"]},
    ],
}

# 페르소나 풀 (generate_training_data.py와 동일)
PERSONAS = [
    {"age_group": "10대", "gender": "female", "occupation": "고등학생", "income": "용돈 월 10만원", "traits": ["SNS 활발", "트렌드 민감", "또래 의식"]},
    {"age_group": "10대", "gender": "male", "occupation": "고등학생", "income": "용돈 월 15만원", "traits": ["게임 좋아함", "가성비 중시", "유튜브 시청"]},
    {"age_group": "20대", "gender": "female", "occupation": "대학생", "income": "월 150만원(알바)", "traits": ["SNS 활발", "뷰티 관심", "가성비 중시"]},
    {"age_group": "20대", "gender": "male", "occupation": "대학생", "income": "월 100만원(알바)", "traits": ["자취", "배달 자주", "가격 민감"]},
    {"age_group": "20대", "gender": "female", "occupation": "직장인(신입)", "income": "월 250만원", "traits": ["자기계발", "소확행", "인스타 활발"]},
    {"age_group": "20대", "gender": "male", "occupation": "직장인(신입)", "income": "월 280만원", "traits": ["재테크 관심", "운동", "효율 중시"]},
    {"age_group": "20대", "gender": "male", "occupation": "군인", "income": "월 100만원", "traits": ["외출 제한", "모바일 위주", "저축 중"]},
    {"age_group": "30대", "gender": "female", "occupation": "직장인(과장)", "income": "월 350만원", "traits": ["워라밸", "육아", "프리미엄 선호"]},
    {"age_group": "30대", "gender": "male", "occupation": "직장인(대리)", "income": "월 320만원", "traits": ["내집마련", "실용적", "브랜드 의식"]},
    {"age_group": "30대", "gender": "female", "occupation": "프리랜서", "income": "월 300만원(변동)", "traits": ["자유로움", "건강 관심", "미니멀"]},
    {"age_group": "30대", "gender": "male", "occupation": "스타트업 대표", "income": "월 400만원(변동)", "traits": ["리스크 감수", "네트워킹", "트렌드 파악"]},
    {"age_group": "30대", "gender": "female", "occupation": "전업맘", "income": "가구소득 월 500만원", "traits": ["육아 정보", "가성비", "안전 중시"]},
    {"age_group": "40대", "gender": "male", "occupation": "직장인(부장)", "income": "월 500만원", "traits": ["가족 중심", "안정 추구", "골프"]},
    {"age_group": "40대", "gender": "female", "occupation": "자영업", "income": "월 400만원(변동)", "traits": ["실용적", "경험 중시", "교육열"]},
    {"age_group": "40대", "gender": "male", "occupation": "공무원", "income": "월 380만원", "traits": ["안정 지향", "보수적", "가족 중심"]},
    {"age_group": "50대", "gender": "male", "occupation": "임원", "income": "월 700만원", "traits": ["건강 관심", "보수적", "품질 중시"]},
    {"age_group": "50대", "gender": "female", "occupation": "주부", "income": "가구소득 월 600만원", "traits": ["가족 건강", "알뜰 소비", "TV 시청"]},
    {"age_group": "50대", "gender": "male", "occupation": "택시기사", "income": "월 280만원", "traits": ["장시간 근무", "실용적", "라디오 청취"]},
    {"age_group": "60대+", "gender": "male", "occupation": "은퇴", "income": "연금 월 200만원", "traits": ["건강 최우선", "보수적", "디지털 약함"]},
    {"age_group": "60대+", "gender": "female", "occupation": "은퇴", "income": "연금 월 150만원", "traits": ["손주", "건강식", "전통 선호"]},
]


def build_prompt(question: dict, category: str, persona: dict) -> str:
    persona_desc = (
        f"- 연령대: {persona['age_group']}\n"
        f"- 성별: {'여성' if persona['gender'] == 'female' else '남성'}\n"
        f"- 직업: {persona['occupation']}\n"
        f"- 소득: {persona['income']}\n"
        f"- 특성: {', '.join(persona['traits'])}"
    )

    choices_text = "\n".join([f"  {i+1}. {c}" for i, c in enumerate(question['choices'])])

    return f"""당신은 한국 설문조사 시뮬레이션 엔진입니다.
아래 페르소나가 설문에 답변합니다.

## 페르소나
{persona_desc}

## 설문 (카테고리: {category})
질문: {question['q']}
보기:
{choices_text}

## 규칙 (실제 한국인처럼 답변)
- 극단적 답변(1번, 마지막 번)은 확신이 있을 때만 선택
- 사회적으로 바람직해 보이는 답에 약간 끌리는 경향
- 잘 모르는 주제는 "보통"이나 중간을 선택하는 경향
- 페르소나의 연령/직업/특성에 맞게 현실적으로 답변
- 10대~20대는 솔직하게, 40대 이상은 신중하게

## 출력 (JSON만, 다른 텍스트 없이)
{{"selected": "선택한 보기 텍스트", "reasoning": "1문장 이유 (1인칭, 페르소나 말투)", "confidence": 0.5~0.9}}"""


def parse_response(text: str) -> dict | None:
    text = text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def make_dedup_key(question: str, persona: dict) -> str:
    return f"{question}|{persona['age_group']}|{persona['gender']}|{persona['occupation']}"


def load_existing_keys(path: str) -> set:
    keys = set()
    p = Path(path)
    if not p.exists():
        return keys
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                inp = item.get("input", {})
                key = f"{inp.get('question', '')}|{inp.get('persona_age_group', '')}|{inp.get('persona_gender', '')}|{inp.get('persona_occupation', '')}"
                keys.add(key)
            except json.JSONDecodeError:
                continue
    return keys


async def generate_one(client, question, category, persona, model, semaphore):
    async with semaphore:
        prompt = build_prompt(question, category, persona)
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            output = parse_response(text)

            if output is None or "selected" not in output or "reasoning" not in output:
                return None

            # 선택한 보기가 실제 보기에 있는지 확인
            if output["selected"] not in question["choices"]:
                return None

            return {
                "input": {
                    "question": question["q"],
                    "choices": question["choices"],
                    "category": category,
                    "persona_age_group": persona["age_group"],
                    "persona_gender": persona["gender"],
                    "persona_occupation": persona["occupation"],
                    "persona_income": persona["income"],
                    "persona_traits": persona["traits"],
                },
                "output": output,
            }
        except anthropic.RateLimitError:
            await asyncio.sleep(30)
            return None
        except Exception as e:
            print(f"  Error: {e}")
            return None


async def generate_batch(count, output_path, model, concurrent):
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrent)

    categories = list(SURVEY_QUESTIONS.keys())
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    existing_keys = load_existing_keys(output_path)
    print(f"Loaded {len(existing_keys)} existing entries")
    print(f"Generating {count} survey response examples...")
    print(f"Model: {model}")
    print(f"Categories: {len(categories)}, Questions: {sum(len(v) for v in SURVEY_QUESTIONS.values())}, Personas: {len(PERSONAS)}")

    generated = 0
    errors = 0
    skipped_dup = 0
    batch_size = min(concurrent * 2, count)

    with open(output_file, "a", encoding="utf-8") as f:
        for batch_start in range(0, count, batch_size):
            batch_end = min(batch_start + batch_size, count)
            tasks = []

            for i in range(batch_start, batch_end):
                category = random.choice(categories)
                question = random.choice(SURVEY_QUESTIONS[category])
                persona = random.choice(PERSONAS)
                dedup_key = make_dedup_key(question["q"], persona)

                tasks.append((
                    generate_one(client, question, category, persona, model, semaphore),
                    f"[{i+1}/{count}] {category} | {persona['age_group']} {persona['gender']}",
                    dedup_key,
                ))

            results = await asyncio.gather(*[t[0] for t in tasks], return_exceptions=True)

            for (_, desc, dedup_key), result in zip(tasks, results):
                if isinstance(result, Exception) or result is None:
                    errors += 1
                elif dedup_key in existing_keys:
                    skipped_dup += 1
                else:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    existing_keys.add(dedup_key)
                    generated += 1
                    print(f"{desc} → OK")

            print(f"  Progress: {generated}/{count} generated, {errors} errors, {skipped_dup} dups")

    print(f"\nDone! {generated} examples generated")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="설문 응답 학습 데이터 생성")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--output", type=str, default="training/data/survey_responses.jsonl")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001")
    parser.add_argument("--concurrent", type=int, default=5)
    args = parser.parse_args()

    asyncio.run(generate_batch(args.count, args.output, args.model, args.concurrent))


if __name__ == "__main__":
    main()
