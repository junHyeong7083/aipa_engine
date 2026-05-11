"""
응답 생성기 (Response Generator)

AI 페르소나의 설문 응답을 생성하는 서비스.
통계적 사전 분포(Prior) + 페르소나 특성 + 인간 응답 편향을 반영하여 현실적인 응답을 생성.

실제 설문조사 방법론 연구를 기반으로 6가지 인간적 특성을 시뮬레이션:
1. 중심화 경향 (극단 회피)
2. 묵종 편향 (긍정 편향)
3. 사회적 바람직성
4. 직선응답 (피로도)
5. 무응답 시뮬레이션
6. 그룹 상관관계 (또래 효과)
7. 시간대/요일 효과
8. 점수 노이즈
"""

import random
import math
from typing import Optional
from datetime import datetime

import numpy as np

from ..models.persona import Persona, AgeGroup, Gender
from ..models.survey import SurveyQuestion, SurveyResponse, QuestionType
from .llm_service import LLMService

import logging
logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    설문 응답 생성기

    단순 랜덤이 아니라 페르소나의 연령/성별/특성에 따라
    응답 확률이 달라지는 확률적 모델링 + 인간 응답 편향 시뮬레이션.
    """

    # === 클래스 레벨 상수 ===
    TRAIT_MODIFIER_DEFAULT = 1.3
    TRAIT_MODIFIER_STRONG = 1.5
    PRICE_SENSITIVITY_BASE = 0.5
    TECH_ADOPTION_BASE = 0.5

    # === 키워드 동의어 매핑 ===
    KEYWORD_SYNONYMS = {
        "가격": ["가격", "비용", "저렴", "싸", "싸다", "할인", "저가", "경제적"],
        "기술": ["앱", "온라인", "디지털", "ai", "테크", "IT", "인터넷", "모바일", "스마트"],
        "프리미엄": ["프리미엄", "고급", "명품", "럭셔리", "하이엔드"],
        "친환경": ["친환경", "에코", "그린", "지속가능", "유기농", "자연"],
        "건강": ["건강", "웰빙", "운동", "헬스", "영양", "다이어트"],
        "편의": ["편리", "편의", "간편", "쉬운", "빠른", "신속"],
        "안전": ["안전", "보안", "신뢰", "믿을"],
        "브랜드": ["브랜드", "유명", "인기", "인지도"],
    }

    # === 특성-키워드 매핑 ===
    TRAIT_KEYWORD_MAP = {
        "가성비 중시": ("가격", TRAIT_MODIFIER_DEFAULT),
        "품질 중시": ("프리미엄", TRAIT_MODIFIER_DEFAULT),
        "친환경 선호": ("친환경", TRAIT_MODIFIER_STRONG),
        "건강 관심": ("건강", TRAIT_MODIFIER_DEFAULT),
        "편의 추구": ("편의", TRAIT_MODIFIER_DEFAULT),
        "디지털 친숙": ("기술", TRAIT_MODIFIER_DEFAULT),
        "오프라인 선호": ("기술", 0.7),
        "브랜드 선호": ("브랜드", TRAIT_MODIFIER_DEFAULT),
        "트렌디": ("기술", TRAIT_MODIFIER_DEFAULT),
        "보수적": ("안전", TRAIT_MODIFIER_DEFAULT),
        "실용적": ("가격", TRAIT_MODIFIER_DEFAULT),
        "진보적": ("친환경", TRAIT_MODIFIER_DEFAULT),
        "안전 중시": ("안전", TRAIT_MODIFIER_STRONG),
        "미니멀리스트": ("편의", TRAIT_MODIFIER_DEFAULT),
        "럭셔리 선호": ("프리미엄", TRAIT_MODIFIER_STRONG),
        "자연주의": ("친환경", TRAIT_MODIFIER_DEFAULT),
        "다이어터": ("건강", TRAIT_MODIFIER_DEFAULT),
        "운동 매니아": ("건강", TRAIT_MODIFIER_STRONG),
        "얼리어답터": ("기술", TRAIT_MODIFIER_STRONG),
        "가격 민감": ("가격", TRAIT_MODIFIER_STRONG),
    }

    # === 사회적 바람직성 키워드 ===
    SOCIAL_DESIRABILITY_KEYWORDS = [
        "환경", "기부", "봉사", "건강", "운동", "독서", "교육", "절약",
        "재활용", "분리수거", "친환경", "지속가능", "윤리", "공정",
    ]

    # === 연령대별 응답 특성 ===
    AGE_RESPONSE_TRAITS = {
        AgeGroup.TEENS: {"극단회피": 0.15, "묵종": 0.10, "직선응답률": 0.12, "무응답률": 0.03},
        AgeGroup.TWENTIES: {"극단회피": 0.20, "묵종": 0.08, "직선응답률": 0.08, "무응답률": 0.03},
        AgeGroup.THIRTIES: {"극단회피": 0.25, "묵종": 0.12, "직선응답률": 0.05, "무응답률": 0.04},
        AgeGroup.FORTIES: {"극단회피": 0.30, "묵종": 0.15, "직선응답률": 0.06, "무응답률": 0.05},
        AgeGroup.FIFTIES: {"극단회피": 0.30, "묵종": 0.18, "직선응답률": 0.08, "무응답률": 0.06},
        AgeGroup.SIXTIES_PLUS: {"극단회피": 0.25, "묵종": 0.22, "직선응답률": 0.15, "무응답률": 0.10},
    }

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        self._setup_priors()
        # 그룹 상관관계용: 그룹별 이전 응답 기억
        self._group_responses: dict[str, list[int]] = {}
        # 직선응답용: 페르소나별 이전 응답 기억
        self._persona_prev_response: dict[str, str] = {}
        self._persona_question_count: dict[str, int] = {}

        # ★ 자체 모델 연결: 임베딩 모델(점수 예측) + 0.5B 모델(이유 생성)
        self._eval_service = None
        self._eval_service_loaded = False

    def _ensure_eval_service(self) -> bool:
        """EvalService(임베딩 모델 + 0.5B 모델) 로드"""
        if self._eval_service is not None:
            return self._eval_service.available
        if self._eval_service_loaded:
            return False

        self._eval_service_loaded = True
        try:
            from .eval_service import EvalService
            self._eval_service = EvalService()
            available = self._eval_service.available
            if available:
                logger.info("ResponseGenerator: 임베딩 모델 연결 성공")
            else:
                logger.warning("ResponseGenerator: 임베딩 모델 로드 실패, 확률 엔진으로 폴백")
            return available
        except Exception as e:
            logger.warning(f"EvalService 초기화 실패: {e}")
            return False

    def _predict_score_with_embedding(self, persona: Persona, question: SurveyQuestion) -> Optional[int]:
        """
        ★ 임베딩 모델로 점수 예측 (0~100)
        "이 페르소나가 이 질문에 얼마나 긍정적인가?"
        """
        if not self._ensure_eval_service():
            return None

        try:
            import torch

            # 페르소나 속성 → 임베딩 모델 입력
            age_map = {
                AgeGroup.TEENS: "10대", AgeGroup.TWENTIES: "20대", AgeGroup.THIRTIES: "30대",
                AgeGroup.FORTIES: "40대", AgeGroup.FIFTIES: "50대", AgeGroup.SIXTIES_PLUS: "60대+",
            }
            gender_map = {Gender.MALE: "남성", Gender.FEMALE: "여성"}

            svc = self._eval_service
            age_idx = torch.tensor([svc._safe_index("age_groups", age_map.get(persona.attributes.age_group, "30대"))])
            gender_idx = torch.tensor([svc._safe_index("genders", gender_map.get(persona.attributes.gender, "남성"))])
            occ_idx = torch.tensor([svc._safe_index("occupations", persona.attributes.occupation)])
            trait_idx = torch.tensor([svc._encode_traits(persona.attributes.traits)])

            # 질문 텍스트에서 카테고리 추정
            cat_name = self._guess_category(question.text)
            cat_idx = torch.tensor([svc._safe_index("categories", cat_name)])

            # 축: 설문이므로 "선호도" 사용
            axis_idx = torch.tensor([svc._safe_index("axes", "선호도")])

            with torch.no_grad():
                score = svc.model(age_idx, gender_idx, occ_idx, trait_idx, cat_idx, axis_idx)
                score_100 = int(score.item() * 100)

            # ★ 트렌드 보정 (네이버 데이터랩 RAG)
            trend_adj = self._get_trend_adjustment(question.text, cat_name)
            if trend_adj != 0:
                score_100 = max(0, min(100, score_100 + trend_adj))
                logger.debug(f"트렌드 보정: {trend_adj:+d} → 최종 {score_100}점")

            return max(0, min(100, score_100))

        except Exception as e:
            logger.warning(f"임베딩 점수 예측 실패: {e}")
            return None

    def _get_trend_adjustment(self, question_text: str, category: str) -> int:
        """
        ★ 네이버 데이터랩 트렌드 기반 점수 보정
        관련 검색 트렌드가 상승 중이면 점수 상향, 하락이면 하향
        """
        if not self._ensure_eval_service():
            return 0

        svc = self._eval_service
        if not svc._ensure_rag_loaded():
            return 0

        try:
            query = f"{category} {question_text[:50]}"
            results = svc._rag.search_trends(query, n_results=3)

            documents = results.get("documents", [[]])[0]
            if not documents:
                return 0

            trend_count = len(documents)
            base_boost = min(trend_count * 3, 10)  # 최대 +10점

            trend_text = " ".join(documents).lower()
            positive_kw = ["증가", "상승", "인기", "급등", "성장", "확대", "호조"]
            negative_kw = ["감소", "하락", "위축", "둔화", "축소", "부진"]

            pos = sum(1 for kw in positive_kw if kw in trend_text)
            neg = sum(1 for kw in negative_kw if kw in trend_text)
            sentiment = (pos - neg) * 2

            return base_boost + sentiment

        except Exception as e:
            logger.debug(f"트렌드 보정 실패 (무시): {e}")
            return 0

    def _guess_category(self, question_text: str) -> str:
        """질문 텍스트에서 임베딩 모델 카테고리를 추정"""
        category_keywords = {
            "식품": ["음식", "식품", "맛", "배달", "레스토랑", "카페", "음료", "유기농", "식단"],
            "앱서비스": ["앱", "서비스", "구독", "플랫폼", "온라인", "디지털"],
            "광고": ["광고", "마케팅", "브랜드", "캠페인", "홍보"],
            "제품": ["제품", "상품", "구매", "가격", "품질"],
            "교육": ["교육", "학습", "강의", "수업", "학교"],
            "건강": ["건강", "운동", "의료", "병원", "약", "헬스"],
            "환경": ["환경", "친환경", "에코", "재활용", "지속가능"],
            "정책": ["정책", "법", "규제", "정부", "세금", "복지"],
            "패션": ["패션", "옷", "의류", "스타일", "뷰티"],
            "테크": ["기술", "IT", "AI", "로봇", "자동화"],
        }
        text_lower = question_text.lower()
        for cat, keywords in category_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return cat
        return "제품"  # 기본값

    def _score_to_choice_index(self, score: int, n_choices: int) -> list[float]:
        """
        임베딩 점수(0~100) → 선택지별 확률 분포 변환
        점수가 높을수록 앞쪽(긍정) 선택지에 높은 확률
        """
        if n_choices <= 1:
            return [1.0]

        # 점수를 0~1로 정규화
        normalized = score / 100.0

        # 선택지 위치별 기대값 (0=가장 긍정, n-1=가장 부정)
        probs = []
        for i in range(n_choices):
            # 선택지 위치: 0이 긍정, n-1이 부정
            pos = i / (n_choices - 1)  # 0~1
            # 점수가 높으면 pos=0(긍정) 근처에 높은 확률
            # 가우시안 커널: 점수 위치에서 멀수록 확률 낮음
            distance = abs(pos - (1.0 - normalized))
            prob = math.exp(-4.0 * distance * distance)
            probs.append(prob)

        total = sum(probs)
        return [p / total for p in probs]

    def _generate_explanation_local(self, persona: Persona, question: SurveyQuestion, selected_choice: str, score: Optional[int]) -> Optional[str]:
        """
        ★ 0.5B 파인튜닝 모델로 이유 생성 (Claude API 미사용)
        """
        if not self._ensure_eval_service():
            return None

        svc = self._eval_service
        if not svc._ensure_reasoning_model_loaded():
            return None

        try:
            import torch

            age_map = {
                AgeGroup.TEENS: "10대", AgeGroup.TWENTIES: "20대", AgeGroup.THIRTIES: "30대",
                AgeGroup.FORTIES: "40대", AgeGroup.FIFTIES: "50대", AgeGroup.SIXTIES_PLUS: "60대+",
            }
            gender_map = {Gender.MALE: "남성", Gender.FEMALE: "여성"}

            age_str = age_map.get(persona.attributes.age_group, "30대")
            gender_str = gender_map.get(persona.attributes.gender, "남성")
            traits_str = ", ".join(persona.attributes.traits[:3]) if persona.attributes.traits else "없음"
            score_str = f" (긍정도: {score}점/100점)" if score is not None else ""

            prompt = f"""당신은 설문 응답자입니다.
다음 페르소나의 관점에서 설문 답변 이유를 1문장으로 설명하세요.

[페르소나] {age_str} {gender_str} {persona.attributes.occupation} (특성: {traits_str})
[질문] {question.text[:100]}
[선택한 답변] {selected_choice}{score_str}

이유:"""

            inputs = svc._reasoning_tokenizer(
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = svc._reasoning_model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.7,
                    do_sample=True,
                )

            raw = svc._reasoning_tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()

            # 첫 번째 문장만
            for sep in [".", "다.", "요.", "습니다."]:
                if sep in raw:
                    raw = raw[:raw.index(sep) + len(sep)]
                    break

            if len(raw) > 5:
                return raw

        except Exception as e:
            logger.warning(f"0.5B 이유 생성 실패: {e}")

        return None

    def _setup_priors(self):
        """사전 확률 분포 설정 (임베딩 모델 없을 때 폴백용)"""
        self.price_sensitivity = {
            AgeGroup.TEENS: 0.8, AgeGroup.TWENTIES: 0.7, AgeGroup.THIRTIES: 0.5,
            AgeGroup.FORTIES: 0.4, AgeGroup.FIFTIES: 0.4, AgeGroup.SIXTIES_PLUS: 0.6,
        }
        self.tech_adoption = {
            AgeGroup.TEENS: 0.9, AgeGroup.TWENTIES: 0.85, AgeGroup.THIRTIES: 0.7,
            AgeGroup.FORTIES: 0.5, AgeGroup.FIFTIES: 0.35, AgeGroup.SIXTIES_PLUS: 0.2,
        }

    # ================================================================
    # 인간 응답 편향 시뮬레이션
    # ================================================================

    def _apply_central_tendency(self, probs: list[float], persona: Persona) -> list[float]:
        """
        중심화 경향 (Extreme Response Avoidance)
        실제 사람은 극단적 답변(1번, 5번)을 피하고 중간을 선호하는 경향.
        한국인은 특히 이 경향이 강함.
        """
        n = len(probs)
        if n < 3:
            return probs

        rate = self.AGE_RESPONSE_TRAITS.get(
            persona.attributes.age_group, {}
        ).get("극단회피", 0.25)

        probs = probs.copy()
        # 양 끝(첫번째, 마지막) 확률을 줄이고 중앙으로 이동
        steal_first = probs[0] * rate
        steal_last = probs[-1] * rate
        probs[0] -= steal_first
        probs[-1] -= steal_last

        mid = n // 2
        probs[mid] += steal_first + steal_last

        return probs

    def _apply_acquiescence_bias(self, probs: list[float], persona: Persona) -> list[float]:
        """
        묵종 편향 (Acquiescence Bias)
        "그렇다" 쪽에 동의하는 경향. 연령이 높을수록 강해짐.
        앞쪽 보기(긍정적)에 확률 부스트.
        """
        n = len(probs)
        if n < 2:
            return probs

        rate = self.AGE_RESPONSE_TRAITS.get(
            persona.attributes.age_group, {}
        ).get("묵종", 0.12)

        probs = probs.copy()
        # 앞쪽 절반 (긍정적 보기)에 확률 추가
        positive_half = n // 2
        boost_per = (sum(probs) * rate) / max(positive_half, 1)
        for i in range(positive_half):
            probs[i] += boost_per

        # 정규화
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]

        return probs

    def _apply_social_desirability(self, probs: list[float], question: SurveyQuestion) -> list[float]:
        """
        사회적 바람직성 편향 (Social Desirability Bias)
        환경, 건강, 기부 등 사회적으로 바람직한 주제에서
        실제보다 더 긍정적으로 답하는 경향.
        """
        question_text = question.text.lower()
        is_socially_sensitive = any(
            kw in question_text for kw in self.SOCIAL_DESIRABILITY_KEYWORDS
        )

        if not is_socially_sensitive:
            return probs

        n = len(probs)
        if n < 2:
            return probs

        probs = probs.copy()
        # 긍정적 보기(앞쪽)에 20% 부스트
        boost_rate = 0.20
        positive_count = max(n // 2, 1)
        for i in range(positive_count):
            probs[i] *= (1 + boost_rate)

        # 정규화
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]

        return probs

    def _check_straightlining(self, persona: Persona) -> Optional[str]:
        """
        직선응답 (Straight-lining)
        설문이 길어지면 피로해서 이전과 같은 답을 반복하는 현상.
        질문이 5개 이상 진행되면 확률적으로 발생.
        """
        pid = persona.id
        count = self._persona_question_count.get(pid, 0)
        prev = self._persona_prev_response.get(pid)

        if count < 5 or prev is None:
            return None

        rate = self.AGE_RESPONSE_TRAITS.get(
            persona.attributes.age_group, {}
        ).get("직선응답률", 0.08)

        # 질문이 많을수록 직선응답 확률 증가 (피로도)
        fatigue_multiplier = min(count / 10, 2.0)
        effective_rate = rate * fatigue_multiplier

        if random.random() < effective_rate:
            return prev  # 이전과 같은 답 반복

        return None

    def _check_nonresponse(self, persona: Persona, question: SurveyQuestion) -> bool:
        """
        무응답 시뮬레이션 (Non-response)
        일정 확률로 문항을 스킵. 민감한 질문은 스킵률 더 높음.
        """
        base_rate = self.AGE_RESPONSE_TRAITS.get(
            persona.attributes.age_group, {}
        ).get("무응답률", 0.04)

        # 민감 키워드가 있으면 스킵률 2배
        sensitive_keywords = ["소득", "월급", "연봉", "정치", "종교", "성생활", "투표"]
        question_text = question.text.lower()
        if any(kw in question_text for kw in sensitive_keywords):
            base_rate *= 2.0

        return random.random() < base_rate

    def _apply_group_correlation(self, probs: list[float], persona: Persona) -> list[float]:
        """
        그룹 상관관계 (Group Correlation / Peer Effect)
        같은 연령대+성별 그룹의 이전 응답에 영향을 받는 경향.
        또래끼리 비슷한 답을 하는 현상 시뮬레이션.
        """
        group_key = f"{persona.attributes.age_group.value}_{persona.attributes.gender.value}"
        prev_indices = self._group_responses.get(group_key, [])

        if not prev_indices:
            return probs

        n = len(probs)
        probs = probs.copy()

        # 그룹 내 이전 응답들의 평균 인덱스 계산
        valid_indices = [i for i in prev_indices[-5:] if i < n]  # 최근 5개
        if not valid_indices:
            return probs

        avg_idx = sum(valid_indices) / len(valid_indices)

        # 평균 인덱스 근처 보기에 확률 부스트 (동조 효과)
        peer_strength = 0.15  # 동조 강도 15%
        for i in range(n):
            distance = abs(i - avg_idx)
            boost = peer_strength * max(0, 1 - distance / n)
            probs[i] *= (1 + boost)

        # 정규화
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]

        return probs

    def _apply_time_effect(self, probs: list[float]) -> list[float]:
        """
        시간대/요일 효과 (Time-of-Day / Day-of-Week Effect)
        - 야간(22~6시): 충동적, 극단적 응답 증가
        - 월요일 오전: 부정적 경향
        - 금요일 오후: 긍정적 경향
        """
        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()  # 0=월, 4=금, 6=일

        n = len(probs)
        if n < 2:
            return probs

        probs = probs.copy()

        # 야간 효과 (22~6시): 극단적 응답 증가
        if hour >= 22 or hour < 6:
            if n >= 3:
                probs[0] *= 1.15    # 첫번째(가장 긍정) 증가
                probs[-1] *= 1.10   # 마지막(가장 부정) 증가
                mid = n // 2
                probs[mid] *= 0.90  # 중간 감소

        # 월요일 오전 (부정적)
        elif weekday == 0 and hour < 12:
            negative_half = n // 2
            for i in range(negative_half, n):
                probs[i] *= 1.10  # 부정적 보기 증가

        # 금요일 오후 (긍정적)
        elif weekday == 4 and hour >= 12:
            positive_half = n // 2
            for i in range(positive_half):
                probs[i] *= 1.10  # 긍정적 보기 증가

        # 정규화
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]

        return probs

    def _add_score_noise(self, score: float, persona: Persona) -> float:
        """
        점수 노이즈 (Response Noise)
        같은 사람도 매번 다른 점수를 줌. ±5~10점 변동.
        """
        # 연령이 높을수록 일관적 (노이즈 작음)
        age_noise_map = {
            AgeGroup.TEENS: 8.0, AgeGroup.TWENTIES: 7.0, AgeGroup.THIRTIES: 5.0,
            AgeGroup.FORTIES: 4.0, AgeGroup.FIFTIES: 4.0, AgeGroup.SIXTIES_PLUS: 5.0,
        }
        noise_std = age_noise_map.get(persona.attributes.age_group, 5.0)
        noise = np.random.normal(0, noise_std)
        return score + noise

    # ================================================================
    # 메인 응답 생성 로직
    # ================================================================

    async def generate_response(
        self,
        persona: Persona,
        question: SurveyQuestion,
        generate_explanation: bool = True,
    ) -> SurveyResponse:
        """페르소나의 설문 응답 생성 (인간 편향 반영)"""

        # 질문 카운트 증가
        pid = persona.id
        self._persona_question_count[pid] = self._persona_question_count.get(pid, 0) + 1

        # 무응답 체크
        if self._check_nonresponse(persona, question):
            return SurveyResponse(
                persona_id=persona.id,
                question_id=question.id,
                selected_choice=None,
                explanation="(무응답)",
            )

        if question.question_type == QuestionType.SINGLE_CHOICE:
            return await self._generate_single_choice_response(
                persona, question, generate_explanation
            )
        elif question.question_type == QuestionType.LIKERT_SCALE:
            return await self._generate_likert_response(
                persona, question, generate_explanation
            )
        elif question.question_type == QuestionType.OPEN_ENDED:
            return await self._generate_open_response(persona, question)
        else:
            return await self._generate_single_choice_response(
                persona, question, generate_explanation
            )

    async def _generate_single_choice_response(
        self,
        persona: Persona,
        question: SurveyQuestion,
        generate_explanation: bool,
    ) -> SurveyResponse:
        """단일 선택 응답 생성 - 임베딩 모델 점수 → 편향 적용 → 0.5B 이유 생성"""
        if not question.choices:
            return SurveyResponse(
                persona_id=persona.id,
                question_id=question.id,
                selected_choice=None,
            )

        # 직선응답 체크 (피로도)
        straightline = self._check_straightlining(persona)
        if straightline and straightline in question.choices:
            self._persona_prev_response[persona.id] = straightline
            return SurveyResponse(
                persona_id=persona.id,
                question_id=question.id,
                selected_choice=straightline,
                explanation="(동일 응답 반복)" if not generate_explanation else None,
                probability=0.0,
            )

        # ★ 1단계: 임베딩 모델로 점수 예측 (0.01초)
        embedding_score = self._predict_score_with_embedding(persona, question)

        if embedding_score is not None:
            # 임베딩 점수 기반 확률 분포
            probs = self._score_to_choice_index(embedding_score, len(question.choices))
            logger.debug(f"임베딩 점수: {embedding_score} → 확률: {[f'{p:.2f}' for p in probs]}")
        else:
            # 폴백: 기존 확률 엔진
            probs = self._calculate_choice_probabilities(persona, question)

        # ★ 2단계: 인간 편향 적용 (순서 중요)
        probs = self._apply_central_tendency(probs, persona)      # 극단 회피
        probs = self._apply_acquiescence_bias(probs, persona)     # 묵종 편향
        probs = self._apply_social_desirability(probs, question)  # 사회적 바람직성
        probs = self._apply_group_correlation(probs, persona)     # 그룹 동조
        probs = self._apply_time_effect(probs)                    # 시간대 효과

        # 최종 정규화
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]

        # 응답 선택
        choices = question.choices
        selected_idx = np.random.choice(len(choices), p=probs)
        selected_choice = choices[selected_idx]

        # 그룹 상관관계 + 직선응답 기록
        group_key = f"{persona.attributes.age_group.value}_{persona.attributes.gender.value}"
        self._group_responses.setdefault(group_key, []).append(selected_idx)
        self._persona_prev_response[persona.id] = selected_choice

        # ★ 3단계: 이유 생성 (0.5B 모델 → 템플릿 폴백, Claude API 사용 안 함)
        explanation = None
        if generate_explanation:
            # 우선순위 1: 자체 0.5B 파인튜닝 모델
            explanation = self._generate_explanation_local(
                persona, question, selected_choice, embedding_score
            )
            # 우선순위 2: 템플릿 (0.5B 모델 없을 때)
            if not explanation:
                explanation = self._generate_template_explanation(
                    persona, question, selected_choice, embedding_score
                )

        return SurveyResponse(
            persona_id=persona.id,
            question_id=question.id,
            selected_choice=selected_choice,
            explanation=explanation,
            probability=probs[selected_idx],
        )

    def _generate_template_explanation(
        self, persona: Persona, question: SurveyQuestion,
        selected_choice: str, score: Optional[int],
    ) -> str:
        """0.5B 모델 없을 때 템플릿 기반 이유 생성 (다양한 표현)"""
        age_map = {
            AgeGroup.TEENS: "10대", AgeGroup.TWENTIES: "20대", AgeGroup.THIRTIES: "30대",
            AgeGroup.FORTIES: "40대", AgeGroup.FIFTIES: "50대", AgeGroup.SIXTIES_PLUS: "60대+",
        }
        gender_map = {Gender.MALE: "남성", Gender.FEMALE: "여성"}
        age = age_map.get(persona.attributes.age_group, "30대")
        gender = gender_map.get(persona.attributes.gender, "")
        occ = persona.attributes.occupation
        traits = persona.attributes.traits[:2] if persona.attributes.traits else []
        trait_str = ", ".join(traits) if traits else ""

        # 질문 키워드 추출 (짧게)
        q_short = question.text[:30].rstrip("?").strip() if question.text else ""

        # 다양한 템플릿 풀에서 랜덤 선택
        high_templates = [
            f"{age} {occ}으로서 '{selected_choice}'에 강하게 공감합니다. {trait_str} 성향이 크게 작용했습니다.",
            f"제 직업({occ})과 {trait_str} 성격을 고려하면 '{selected_choice}'가 자연스러운 선택이었습니다.",
            f"{age} {gender}의 시각에서 이 질문은 명확합니다. '{selected_choice}'가 제 경험과 일치합니다.",
            f"평소 {trait_str} 성향이 강해서 '{selected_choice}'를 망설임 없이 선택했습니다.",
            f"{occ} 일을 하다 보면 이런 질문에 대한 답이 명확해집니다. '{selected_choice}'입니다.",
        ]
        mid_templates = [
            f"{age} {occ}의 입장에서 '{selected_choice}'가 가장 현실적인 답이라고 봅니다.",
            f"고민을 좀 했지만, {trait_str} 성향을 감안하면 '{selected_choice}'가 맞는 것 같습니다.",
            f"제 또래({age}) 사이에서는 '{selected_choice}'가 일반적인 의견일 것 같습니다.",
            f"{occ}의 경험상 '{selected_choice}'가 무난한 선택이라고 생각합니다.",
            f"{gender} {age}로서 이 질문에는 '{selected_choice}'가 솔직한 답변입니다.",
        ]
        low_templates = [
            f"솔직히 확신은 없지만, {age} {occ}의 관점에서 '{selected_choice}'를 선택했습니다.",
            f"이 질문이 좀 어렵긴 한데, 굳이 고르자면 '{selected_choice}'입니다.",
            f"{trait_str} 성향이지만 이 부분은 잘 모르겠어서 '{selected_choice}'로 답했습니다.",
            f"제 경험이 부족한 영역이라 '{selected_choice}'가 맞는지 모르겠습니다.",
            f"{age} {gender}으로서 크게 관심 없는 주제라 '{selected_choice}'를 소극적으로 선택했습니다.",
        ]

        if score is not None:
            if score >= 70:
                pool = high_templates
            elif score >= 40:
                pool = mid_templates
            else:
                pool = low_templates
        else:
            pool = mid_templates

        return random.choice(pool)

    def _calculate_choice_probabilities(
        self,
        persona: Persona,
        question: SurveyQuestion,
    ) -> list[float]:
        """선택지별 확률 분포 계산 (기본 로직, 편향 적용 전)"""
        n_choices = len(question.choices)
        probs = np.ones(n_choices) / n_choices

        for i, choice in enumerate(question.choices):
            modifier = 1.0
            choice_lower = choice.lower()

            if self._matches_keyword_category(choice_lower, "가격"):
                price_sens = self.price_sensitivity.get(
                    persona.attributes.age_group, self.PRICE_SENSITIVITY_BASE
                )
                modifier *= 1.0 + (price_sens - self.PRICE_SENSITIVITY_BASE)

            if self._matches_keyword_category(choice_lower, "기술"):
                tech_adopt = self.tech_adoption.get(
                    persona.attributes.age_group, self.TECH_ADOPTION_BASE
                )
                modifier *= 1.0 + (tech_adopt - self.TECH_ADOPTION_BASE)

            for trait in persona.attributes.traits:
                for trait_key, (category, mult) in self.TRAIT_KEYWORD_MAP.items():
                    if trait_key in trait and self._matches_keyword_category(choice_lower, category):
                        modifier *= mult

            probs[i] *= modifier

        probs = probs / probs.sum()
        return probs.tolist()

    def _matches_keyword_category(self, text: str, category: str) -> bool:
        """텍스트가 키워드 카테고리의 동의어 중 하나와 매칭되는지 확인"""
        synonyms = self.KEYWORD_SYNONYMS.get(category, [category])
        return any(word in text for word in synonyms)

    async def _generate_likert_response(
        self,
        persona: Persona,
        question: SurveyQuestion,
        generate_explanation: bool,
    ) -> SurveyResponse:
        """리커트 척도 응답 생성 - 임베딩 모델 점수 기반 + 편향 반영"""
        # ★ 임베딩 모델로 기본 점수 예측
        embedding_score = self._predict_score_with_embedding(persona, question)

        if embedding_score is not None:
            # 임베딩 점수(0~100)를 리커트 척도(1~5)로 변환
            scale_range = question.scale_max - question.scale_min
            base_value = question.scale_min + (embedding_score / 100.0) * scale_range
        else:
            base_value = 3.0

        std_dev = 0.8
        value = np.random.normal(base_value, std_dev)

        # 노이즈 추가
        value = self._add_score_noise(value, persona)

        # 중심화 경향: 극단값을 중앙으로 당김
        rate = self.AGE_RESPONSE_TRAITS.get(
            persona.attributes.age_group, {}
        ).get("극단회피", 0.25)
        mid = (question.scale_min + question.scale_max) / 2
        value = value + (mid - value) * rate * 0.5

        # 시간대 효과
        hour = datetime.now().hour
        if hour >= 22 or hour < 6:
            if value > mid:
                value += 0.3
            else:
                value -= 0.3

        value = int(np.clip(value, question.scale_min, question.scale_max))

        # 이유 생성 (자체 모델, Claude 미사용)
        explanation = None
        if generate_explanation:
            explanation = self._generate_explanation_local(
                persona, question, str(value), embedding_score
            )
            if not explanation:
                explanation = f"{value}점을 선택했습니다."

        return SurveyResponse(
            persona_id=persona.id,
            question_id=question.id,
            scale_value=value,
            explanation=explanation,
        )

    async def _generate_open_response(
        self,
        persona: Persona,
        question: SurveyQuestion,
    ) -> SurveyResponse:
        """주관식 응답 생성 - 0.5B 모델 사용"""
        # 무응답 체크
        if random.random() < 0.08:
            return SurveyResponse(
                persona_id=persona.id,
                question_id=question.id,
                open_response="",
            )

        # 0.5B 모델로 생성 시도
        response_text = self._generate_explanation_local(
            persona, question, "(주관식)", None
        )
        if not response_text:
            # 템플릿 폴백
            age_map = {
                AgeGroup.TEENS: "10대", AgeGroup.TWENTIES: "20대", AgeGroup.THIRTIES: "30대",
                AgeGroup.FORTIES: "40대", AgeGroup.FIFTIES: "50대", AgeGroup.SIXTIES_PLUS: "60대+",
            }
            age = age_map.get(persona.attributes.age_group, "30대")
            response_text = f"{age} {persona.attributes.occupation}의 관점에서 답변합니다."

        return SurveyResponse(
            persona_id=persona.id,
            question_id=question.id,
            open_response=response_text,
        )

    def reset_session(self):
        """세션 초기화 (새 시뮬레이션 시작 시 호출)"""
        self._group_responses.clear()
        self._persona_prev_response.clear()
        self._persona_question_count.clear()
