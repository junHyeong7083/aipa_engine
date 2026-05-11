"""
AIPA-Eval 서비스 (페르소나 임베딩 모델 + LLM 보조)

점수 예측: 자체 임베딩 모델 (CPU, 0.01초, 무료)
이유 설명: Claude API (있으면) / 없으면 간단 템플릿

C#으로 비유하면 ML.NET 모델로 예측하고, 부가 설명만 외부 API 사용하는 구조.
"""

import json
import logging
from typing import Optional
from pathlib import Path

from ..config import get_settings
from ..models.evaluation import (
    EvaluationRequest,
    EvaluationResponse,
    EvaluationAxis,
    DEFAULT_AXES,
)

logger = logging.getLogger(__name__)

# 모델 경로 (여러 위치에서 자동 탐색)
def _find_model_root() -> Path:
    """Docker(/app/) 또는 로컬 프로젝트 루트를 자동 탐색"""
    candidates = [
        Path("/app"),  # Docker
        Path(__file__).parent.parent.parent.parent,  # 로컬 (src/aipa_engine/services/ → 프로젝트 루트)
    ]
    for root in candidates:
        model_path = root / "training" / "models" / "embedding" / "persona_embedding_model.pt"
        if model_path.exists():
            return root
    return candidates[-1]  # fallback

_PROJECT_ROOT = _find_model_root()
_EMBEDDING_MODEL_PATH = _PROJECT_ROOT / "training" / "models" / "embedding" / "persona_embedding_model.pt"
_RAG_DIR = _PROJECT_ROOT / "rag"


def _build_embedding_model(embed_dim, hidden_dim, n_ages, n_genders, n_occupations, n_traits, n_categories, n_axes):
    """임베딩 모델 생성 (torch lazy import)"""
    import torch
    import torch.nn as nn

    class PersonaEmbeddingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.age_embed = nn.Embedding(n_ages, embed_dim)
            self.gender_embed = nn.Embedding(n_genders, embed_dim // 2)
            self.occupation_embed = nn.Embedding(n_occupations, embed_dim)
            self.trait_embed = nn.Embedding(n_traits, embed_dim)
            self.category_embed = nn.Embedding(n_categories, embed_dim)
            self.axis_embed = nn.Embedding(n_axes, embed_dim)

            self.trait_attention = nn.Linear(embed_dim, 1)

            persona_input_dim = embed_dim + (embed_dim // 2) + embed_dim + embed_dim
            self.persona_encoder = nn.Sequential(
                nn.Linear(persona_input_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(hidden_dim, embed_dim),
            )
            context_input_dim = embed_dim + embed_dim
            self.context_encoder = nn.Sequential(
                nn.Linear(context_input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
            )
            self.predictor = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
            )

        def forward(self, age, gender, occupation, traits, category, axis):
            age_vec = self.age_embed(age)
            gender_vec = self.gender_embed(gender)
            occ_vec = self.occupation_embed(occupation)
            trait_vecs = self.trait_embed(traits)
            trait_weights = torch.softmax(self.trait_attention(trait_vecs), dim=1)
            trait_vec = (trait_vecs * trait_weights).sum(dim=1)
            persona_input = torch.cat([age_vec, gender_vec, occ_vec, trait_vec], dim=-1)
            persona_vec = self.persona_encoder(persona_input)
            cat_vec = self.category_embed(category)
            axis_vec = self.axis_embed(axis)
            context_input = torch.cat([cat_vec, axis_vec], dim=-1)
            context_vec = self.context_encoder(context_input)
            combined = torch.cat([persona_vec, context_vec], dim=-1)
            return self.predictor(combined).squeeze(-1)

    return PersonaEmbeddingModel()


# ========== EvalService ==========

class EvalService:
    """
    평가 서비스

    우선순위:
    1. 임베딩 모델로 점수 예측 (CPU, 0.01초) ← 핵심 기술
    2. Claude API로 이유 설명 생성 (선택)
    3. 임베딩 모델 없으면 Claude 전체 처리 (fallback)
    4. 둘 다 없으면 Mock
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else _EMBEDDING_MODEL_PATH
        self.model = None
        self.vocab = None
        self._load_attempted = False
        self._rag = None
        self._rag_load_attempted = False
        # 0.5B 이유 생성 모델
        self._reasoning_model = None
        self._reasoning_tokenizer = None
        self._reasoning_load_attempted = False

    def _ensure_model_loaded(self) -> bool:
        """임베딩 모델 로드 (최초 1회)"""
        if self.model is not None:
            return True
        if self._load_attempted:
            return False

        self._load_attempted = True

        if not self.model_path.exists():
            logger.warning(f"Embedding model not found: {self.model_path}")
            return False

        try:
            import torch
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
            config = checkpoint["config"]
            self.vocab = checkpoint["vocab"]

            self.model = _build_embedding_model(
                embed_dim=config["embed_dim"],
                hidden_dim=config["hidden_dim"],
                n_ages=len(self.vocab["age_groups"]),
                n_genders=len(self.vocab["genders"]),
                n_occupations=len(self.vocab["occupations"]),
                n_traits=len(self.vocab["traits"]),
                n_categories=len(self.vocab["categories"]),
                n_axes=len(self.vocab["axes"]),
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            logger.info(
                f"Embedding model loaded: {config['total_params']:,} params, "
                f"avg_error: {config['best_avg_error']:.1f}점"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False

    @property
    def available(self) -> bool:
        return self._ensure_model_loaded()

    def _ensure_rag_loaded(self) -> bool:
        """RAG 검색기 로드 (최초 1회). ChromaDB 없으면 False."""
        if self._rag is not None:
            return True
        if self._rag_load_attempted:
            return False

        self._rag_load_attempted = True
        try:
            # rag/ 폴더를 sys.path에 추가해서 import
            import sys
            rag_dir = str(_RAG_DIR)
            if rag_dir not in sys.path:
                sys.path.insert(0, rag_dir)
            from query import AIPARetriever
            self._rag = AIPARetriever(n_results=3)
            logger.info("RAG retriever loaded")
            return True
        except Exception as e:
            logger.warning(f"RAG not available: {e}")
            return False

    def _get_trend_adjustment(self, request: EvaluationRequest) -> dict[str, float]:
        """
        RAG에서 트렌드 검색 → 축별 보정값 계산

        트렌드가 긍정적이면 관련 축 점수를 올리고,
        트렌드가 없으면 보정 없음 (0).

        반환: {"호감도": +5.0, "구매의향": +3.0, ...}
        """
        if not self._ensure_rag_loaded():
            return {}

        try:
            # 자극물 키워드로 트렌드 검색
            query = f"{request.stimulus_type.value} {request.stimulus[:50]}"
            results = self._rag.search_trends(query, n_results=3)

            documents = results.get("documents", [[]])[0]
            if not documents:
                return {}

            # 트렌드 문서 수에 따라 보정값 결정
            # 관련 트렌드가 많을수록 → 시장 관심 높음 → 호감도/구매의향 상승
            trend_count = len(documents)
            base_boost = min(trend_count * 3, 10)  # 최대 +10점

            # 트렌드 내용에서 긍정/부정 키워드 체크
            trend_text = " ".join(documents).lower()
            positive_keywords = ["증가", "상승", "인기", "급등", "성장", "확대", "호조"]
            negative_keywords = ["감소", "하락", "위축", "둔화", "축소", "부진"]

            positive_count = sum(1 for kw in positive_keywords if kw in trend_text)
            negative_count = sum(1 for kw in negative_keywords if kw in trend_text)

            sentiment = (positive_count - negative_count) * 2  # -10 ~ +10 범위

            # 축별 보정값 (트렌드 관련 축만 보정)
            adjustments = {}
            trend_sensitive_axes = ["호감도", "구매의향", "관심도", "사용의향",
                                    "참여의향", "클릭의향", "추천의향"]
            for axis in trend_sensitive_axes:
                adjustments[axis] = base_boost + sentiment

            logger.info(f"Trend adjustment: {trend_count} docs, sentiment={sentiment}, boost={base_boost}")
            return adjustments

        except Exception as e:
            logger.warning(f"Trend adjustment failed: {e}")
            return {}

    def _ensure_reasoning_model_loaded(self) -> bool:
        """0.5B 이유 생성 모델 로드 (최초 1회)"""
        if self._reasoning_model is not None:
            return True
        if self._reasoning_load_attempted:
            return False

        self._reasoning_load_attempted = True
        merged_path = _PROJECT_ROOT / "training" / "models" / "reasoning" / "merged"

        if not merged_path.exists():
            logger.warning(f"Reasoning model not found: {merged_path}")
            return False

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading reasoning model: {merged_path}")
            self._reasoning_tokenizer = AutoTokenizer.from_pretrained(
                str(merged_path), trust_remote_code=True
            )
            self._reasoning_model = AutoModelForCausalLM.from_pretrained(
                str(merged_path),
                torch_dtype=torch.float32,  # CPU는 float32
                device_map="cpu",
                trust_remote_code=True,
            )
            self._reasoning_model.eval()
            logger.info("Reasoning model loaded successfully")
            return True

        except Exception as e:
            logger.warning(f"Failed to load reasoning model: {e}")
            return False

    def _generate_reasoning_local(self, request: EvaluationRequest, evaluations: list) -> list[str]:
        """자체 0.5B 모델로 이유 생성"""
        import torch

        scores_text = "\n".join([f"- {e.name}: {e.score}점" for e in evaluations])
        traits_str = ", ".join(request.persona_traits) if request.persona_traits else "없음"

        prompt = f"""당신은 소비자 반응 평가 전문가입니다.
다음 평가 결과의 이유를 페르소나 관점에서 설명해주세요.

[자극물]
유형: {request.stimulus_type.value}
내용: {request.stimulus[:200]}

[페르소나]
연령대: {request.persona_age_group}
성별: {request.persona_gender}
직업: {request.persona_occupation}
특성: {traits_str}

[평가 점수]
{scores_text}

각 점수의 이유를 1문장씩 설명하고 한줄평을 작성하세요."""

        try:
            inputs = self._reasoning_tokenizer(
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self._reasoning_model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                )

            raw = self._reasoning_tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )

            # JSON 파싱 시도
            import json
            raw = raw.strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
                reasonings = data.get("reasonings", [])
                if isinstance(reasonings, list) and len(reasonings) >= len(evaluations):
                    return [str(r) for r in reasonings[:len(evaluations)]]

            # JSON 실패하면 줄 단위로 파싱
            lines = [l.strip() for l in raw.split("\n") if l.strip() and len(l.strip()) > 5]
            if len(lines) >= len(evaluations):
                return lines[:len(evaluations)]

        except Exception as e:
            logger.warning(f"Local reasoning generation failed: {e}")

        return []

    def _safe_index(self, vocab_key: str, value: str) -> int:
        """vocab에서 인덱스 찾기, 없으면 마지막(기타)"""
        vocab_list = self.vocab[vocab_key]
        try:
            return vocab_list.index(value)
        except ValueError:
            return len(vocab_list) - 1

    def _encode_traits(self, traits: list[str], max_traits: int = 5) -> list[int]:
        """특성 리스트를 인덱스 배열로"""
        indices = []
        for t in traits[:max_traits]:
            indices.append(self._safe_index("traits", t))
        while len(indices) < max_traits:
            indices.append(0)
        return indices

    def _predict_scores(self, request: EvaluationRequest) -> list[EvaluationAxis]:
        """임베딩 모델로 축별 점수 예측"""
        import torch
        axes = request.get_axes()

        age_idx = torch.tensor([self._safe_index("age_groups", request.persona_age_group)])
        gender_idx = torch.tensor([self._safe_index("genders", request.persona_gender)])
        occ_idx = torch.tensor([self._safe_index("occupations", request.persona_occupation)])
        trait_idx = torch.tensor([self._encode_traits(request.persona_traits)])
        cat_idx = torch.tensor([self._safe_index("categories", request.stimulus_type.value)])

        evaluations = []
        with torch.no_grad():
            for axis_name in axes:
                axis_idx = torch.tensor([self._safe_index("axes", axis_name)])
                score = self.model(age_idx, gender_idx, occ_idx, trait_idx, cat_idx, axis_idx)
                score_100 = int(score.item() * 100)
                score_100 = max(0, min(100, score_100))

                evaluations.append(EvaluationAxis(
                    name=axis_name,
                    score=score_100,
                    reasoning="",  # 이유는 나중에 LLM이 채움
                ))

        return evaluations

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        """
        평가 실행

        1. 임베딩 모델 있으면 → 점수 예측 + (선택) Claude로 이유
        2. 없으면 → Claude 전체 처리
        3. 둘 다 없으면 → Mock
        """
        if self._ensure_model_loaded():
            return await self._evaluate_with_embedding(request)

        # 임베딩 모델 없으면 Claude fallback
        settings = get_settings()
        if settings.anthropic_api_key:
            return await self._evaluate_claude_full(request)

        return self._generate_mock(request)

    async def _evaluate_with_embedding(self, request: EvaluationRequest) -> EvaluationResponse:
        """임베딩 모델로 점수 + 트렌드 보정 + (선택) Claude로 이유"""
        # ① 점수 예측 (임베딩 모델, 0.01초)
        evaluations = self._predict_scores(request)

        # ② 트렌드 보정 (RAG, 요즘 관심도에 따라 점수 조정)
        trend_adj = self._get_trend_adjustment(request)
        if trend_adj:
            for ev in evaluations:
                adj = trend_adj.get(ev.name, 0)
                if adj != 0:
                    ev.score = max(0, min(100, ev.score + int(adj)))

        # ③ 이유 생성 (자체 모델 우선 → Claude fallback → 템플릿)
        reasoning_done = False

        # 우선순위 1: 자체 0.5B 모델
        if self._ensure_reasoning_model_loaded():
            try:
                reasonings = self._generate_reasoning_local(request, evaluations)
                if reasonings and len(reasonings) >= len(evaluations):
                    for ev, reason in zip(evaluations, reasonings):
                        ev.reasoning = reason
                    reasoning_done = True
                    logger.info("Reasoning generated by local 0.5B model")
            except Exception as e:
                logger.warning(f"Local reasoning failed: {e}")

        # 우선순위 2: Claude API
        if not reasoning_done:
            settings = get_settings()
            if settings.anthropic_api_key:
                try:
                    reasonings = await self._generate_reasonings(request, evaluations)
                    for ev, reason in zip(evaluations, reasonings):
                        ev.reasoning = reason
                    reasoning_done = True
                except Exception as e:
                    logger.warning(f"Claude reasoning failed: {e}")

        # 우선순위 3: 템플릿
        if not reasoning_done:
            self._fill_template_reasonings(request, evaluations)

        # 한줄평 생성
        top_axis = max(evaluations, key=lambda e: e.score)
        low_axis = min(evaluations, key=lambda e: e.score)
        open_response = f"{top_axis.name}이(가) {top_axis.score}점으로 높고, {low_axis.name}이(가) {low_axis.score}점으로 낮습니다."

        return EvaluationResponse(
            evaluations=evaluations,
            open_response=open_response,
            confidence=0.85,
        )

    async def _generate_reasonings(
        self, request: EvaluationRequest, evaluations: list[EvaluationAxis]
    ) -> list[str]:
        """Claude API로 각 축의 이유만 생성"""
        from .llm_service import LLMService

        llm = LLMService()
        if not llm.client:
            return [""] * len(evaluations)

        scores_summary = ", ".join([f"{e.name}: {e.score}점" for e in evaluations])
        traits_str = ", ".join(request.persona_traits) if request.persona_traits else "없음"

        prompt = f"""다음 페르소나가 자극물을 평가한 결과입니다. 각 점수에 대한 이유를 페르소나 관점에서 1문장씩 작성하세요.

[페르소나] {request.persona_age_group} {request.persona_gender} {request.persona_occupation} (특성: {traits_str})
[자극물] {request.stimulus_type.value}: {request.stimulus[:100]}
[점수] {scores_summary}

JSON 배열로만 응답하세요. 예: ["{evaluations[0].name}에 대한 이유", "{evaluations[1].name}에 대한 이유", ...]"""

        try:
            raw = llm._call_api(prompt, max_tokens=400)
            raw = raw.strip()
            # JSON 배열 파싱
            if raw.startswith("["):
                reasons = json.loads(raw)
                if isinstance(reasons, list) and len(reasons) >= len(evaluations):
                    return [str(r) for r in reasons[:len(evaluations)]]
        except Exception:
            pass

        return [""] * len(evaluations)

    def _fill_template_reasonings(self, request: EvaluationRequest, evaluations: list[EvaluationAxis]):
        """Claude 없을 때 템플릿 기반 이유 생성"""
        persona_desc = f"{request.persona_age_group} {request.persona_occupation}"
        for ev in evaluations:
            if ev.score >= 70:
                ev.reasoning = f"{persona_desc}의 관점에서 {ev.name}이(가) 긍정적으로 평가됩니다."
            elif ev.score >= 40:
                ev.reasoning = f"{persona_desc}의 관점에서 {ev.name}은(는) 보통 수준입니다."
            else:
                ev.reasoning = f"{persona_desc}의 관점에서 {ev.name}이(가) 부정적으로 평가됩니다."

    async def _evaluate_claude_full(self, request: EvaluationRequest) -> EvaluationResponse:
        """임베딩 모델 없을 때 Claude로 전체 처리 (fallback)"""
        from .llm_service import LLMService

        llm = LLMService()
        if not llm.client:
            return self._generate_mock(request)

        axes = request.get_axes()
        traits_str = ", ".join(request.persona_traits) if request.persona_traits else "없음"

        prompt = f"""당신은 소비자 반응 평가 전문가입니다.

[자극물] 유형: {request.stimulus_type.value} / 내용: {request.stimulus}
[페르소나] {request.persona_age_group} {request.persona_gender} {request.persona_occupation} (특성: {traits_str})
[평가 축] {', '.join(axes)}

각 축에 대해 0-100점과 근거를 JSON으로 응답하세요.
{{"evaluations": [{{"name": "축이름", "score": 점수, "reasoning": "이유"}}], "open_response": "한줄평", "confidence": 0.75}}"""

        try:
            raw = llm._call_api(prompt, max_tokens=600)
            return self._parse_json_response(raw, axes)
        except Exception as e:
            logger.error(f"Claude evaluation failed: {e}")
            return self._generate_mock(request)

    def _parse_json_response(self, raw: str, axes: list[str]) -> EvaluationResponse:
        """JSON 응답 파싱"""
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return self._generate_mock_from_axes(axes)

        evaluations = []
        for item in data.get("evaluations", []):
            score = item.get("score", 50)
            if isinstance(score, (int, float)) and score <= 10:
                score = int(score * 10)
            evaluations.append(EvaluationAxis(
                name=item.get("name", ""),
                score=max(0, min(100, int(score))),
                reasoning=item.get("reasoning", ""),
            ))

        found = {e.name for e in evaluations}
        for axis in axes:
            if axis not in found:
                evaluations.append(EvaluationAxis(name=axis, score=50, reasoning=""))

        return EvaluationResponse(
            evaluations=evaluations,
            open_response=data.get("open_response", ""),
            confidence=data.get("confidence", 0.7),
        )

    def _generate_mock(self, request: EvaluationRequest) -> EvaluationResponse:
        return self._generate_mock_from_axes(request.get_axes())

    def _generate_mock_from_axes(self, axes: list[str]) -> EvaluationResponse:
        import random
        evaluations = [
            EvaluationAxis(name=axis, score=random.randint(40, 80), reasoning=f"{axis} (mock)")
            for axis in axes
        ]
        return EvaluationResponse(
            evaluations=evaluations,
            open_response="Mock 데이터입니다.",
            confidence=0.0,
        )
