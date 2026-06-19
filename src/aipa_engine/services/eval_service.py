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

# ─────────────────────────────────────────────────────────────
# 트렌드 보정 파라미터
# ─────────────────────────────────────────────────────────────
# 검색량 증감률(±50%)을 점수로 환산할 때의 진폭. 0.5 * 30 = ±15점.
TREND_MAX_POINTS = 30.0

# 축별 트렌드 민감도 가중치 (트렌드에 따라 더/덜 흔들리는 정도)
# 구매의향·관심도처럼 시장 분위기에 직접 반응하는 축은 높게,
# 가격적절성·신뢰도처럼 트렌드와 무관한 축은 낮게.
AXIS_TREND_WEIGHT = {
    "트렌드부합": 1.2, "트렌드부합도": 1.2,
    "구매의향": 1.0, "클릭의향": 1.0, "관심도": 1.0,
    "사용의향": 0.9, "참여의향": 0.9,
    "호감도": 0.8, "추천의향": 0.8, "디자인호감도": 0.7,
    "신뢰도": 0.3, "가격적절성": 0.2,
}
DEFAULT_AXIS_TREND_WEIGHT = 0.4  # 매핑에 없는 축은 약하게만 반영

# 페르소나 트렌드 추종 성향 (연령 + 특성)
AGE_TREND_MULT = {
    "10대": 1.2, "20대": 1.2, "30대": 1.0,
    "40대": 0.9, "50대": 0.8, "60대+": 0.7,
}
TREND_FOLLOWING_KW = ["트렌디", "트렌드", "유행", "SNS", "인스타", "틱톡",
                      "얼리어답터", "유튜브", "핫"]
TREND_RESISTANT_KW = ["보수", "전통", "검소", "절약", "알뜰", "실용"]


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

    def _compute_trend_signal(self, query: str, n_results: int = 5) -> tuple[float, int]:
        """
        RAG에서 트렌드를 검색해 '관련도 가중 검색량 증감'을 점수로 환산.

        문제 1 해결: 문서 '개수'가 아니라 거리(distance) 기반 관련도로 가중.
        문제 2 해결: 키워드 매칭이 아니라 메타데이터의 latest_ratio vs avg_ratio
                     (실제 검색량 증감)을 사용.

        반환: (base_points, doc_count)
          base_points > 0 : 검색량 상승 추세, < 0 : 하락 추세, 0 : 신호 없음/무관
        """
        if not self._ensure_rag_loaded():
            return 0.0, 0
        try:
            results = self._rag.search_trends(query, n_results=n_results)
            docs = (results.get("documents") or [[]])[0]
            if not docs:
                return 0.0, 0

            metas = (results.get("metadatas") or [[]])[0]
            dists = (results.get("distances") or [[]])[0]
            threshold = getattr(self._rag, "distance_threshold", 1.5) or 1.5

            weighted_sum = 0.0
            weight_total = 0.0
            for i, meta in enumerate(metas):
                dist = dists[i] if i < len(dists) else threshold
                # 관련도: 가까울수록 1, 임계값에서 0
                relevance = max(0.0, 1.0 - (dist / threshold))
                if relevance <= 0:
                    continue
                latest = meta.get("latest_ratio")
                avg = meta.get("avg_ratio")
                if latest is None or not avg:
                    continue
                pct = (float(latest) - float(avg)) / float(avg)  # 검색량 증감률
                pct = max(-0.5, min(0.5, pct))                    # ±50%로 클램프
                weighted_sum += relevance * pct
                weight_total += relevance

            if weight_total == 0:
                return 0.0, len(docs)

            weighted_trend = weighted_sum / weight_total          # [-0.5, 0.5]
            base_points = weighted_trend * TREND_MAX_POINTS       # ≈ [-15, +15]
            return base_points, len(docs)

        except Exception as e:
            logger.debug(f"trend signal failed: {e}")
            return 0.0, 0

    def _persona_trend_sensitivity(self, traits, age_group: str) -> float:
        """
        문제 3-a 해결: 페르소나의 트렌드 추종 성향 배수 (0.4~1.6).
        트렌디·젊을수록 ↑, 보수적·고령일수록 ↓.
        """
        mult = AGE_TREND_MULT.get(age_group, 1.0)
        for t in (traits or []):
            if any(kw in t for kw in TREND_FOLLOWING_KW):
                mult += 0.15
            elif any(kw in t for kw in TREND_RESISTANT_KW):
                mult -= 0.15
        return max(0.4, min(1.6, mult))

    def _get_trend_adjustment(self, request: EvaluationRequest) -> dict[str, float]:
        """
        RAG 트렌드 → 축별 점수 보정값 계산.

        문제 3-b 해결: 축마다 트렌드 민감도(AXIS_TREND_WEIGHT)를 달리 적용하고,
                      페르소나 성향(sensitivity)으로 한 번 더 스케일.
        반환: {"구매의향": +6.2, "가격적절성": +1.1, ...} (음수 가능)
        """
        base_points, doc_count = self._compute_trend_signal(
            f"{request.stimulus_type.value} {request.stimulus[:50]}"
        )
        if base_points == 0.0:
            return {}

        sensitivity = self._persona_trend_sensitivity(
            request.persona_traits, request.persona_age_group
        )

        adjustments: dict[str, float] = {}
        for axis in request.get_axes():
            w = AXIS_TREND_WEIGHT.get(axis, DEFAULT_AXIS_TREND_WEIGHT)
            if w <= 0:
                continue
            adj = base_points * sensitivity * w
            if abs(adj) >= 0.5:
                adjustments[axis] = round(adj, 1)

        if adjustments:
            logger.info(
                f"Trend adj: base={base_points:.1f}, sens={sensitivity:.2f}, "
                f"docs={doc_count}, axes={len(adjustments)}"
            )
        return adjustments

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
