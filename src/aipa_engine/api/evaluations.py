"""
평가 API 엔드포인트 (Evaluation API Endpoints)

AIPA-Eval 파인튜닝 모델을 통한 자극물 평가 API.
C#으로 비유하면 EvaluationController.cs 역할.

주요 엔드포인트:
- POST /api/v1/evaluations/ → 단건 평가
- POST /api/v1/evaluations/batch → 여러 페르소나로 일괄 평가
- GET  /api/v1/evaluations/axes/{stimulus_type} → 카테고리별 기본 평가 축 조회
- GET  /api/v1/evaluations/status → 모델 상태 확인
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..models.evaluation import (
    EvaluationRequest,
    EvaluationResponse,
    StimulusType,
    DEFAULT_AXES,
)
from ..services.eval_service import EvalService

logger = logging.getLogger(__name__)

router = APIRouter()

# 서비스 싱글톤 (모델 로드는 첫 요청 시 1회만)
_eval_service: EvalService | None = None


def _get_service() -> EvalService:
    global _eval_service
    if _eval_service is None:
        _eval_service = EvalService()
    return _eval_service


# === 요청/응답 DTO ===

class BatchEvaluationRequest(BaseModel):
    """여러 페르소나로 일괄 평가 요청"""
    stimulus: str = Field(description="평가할 자극물")
    stimulus_type: StimulusType = StimulusType.GENERAL
    axes: list[str] = Field(default_factory=list)
    personas: list[dict] = Field(
        min_length=1,
        description="페르소나 목록 [{age_group, gender, occupation, income, traits}]",
    )


class BatchEvaluationResponse(BaseModel):
    """일괄 평가 결과"""
    results: list[EvaluationResponse]
    summary: dict[str, float] = Field(description="축별 평균 점수")
    model_type: str = Field(description="사용된 모델 (local/claude/mock)")


# === 엔드포인트 ===

@router.post("/", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    """단건 자극물 평가 - 하나의 페르소나가 자극물을 평가"""
    service = _get_service()
    return await service.evaluate(request)


@router.post("/batch", response_model=BatchEvaluationResponse)
async def batch_evaluate(request: BatchEvaluationRequest):
    """
    일괄 평가 - 여러 페르소나가 동일 자극물을 평가

    예: 20대 여성, 30대 남성, 50대 여성이 같은 광고를 각각 평가
    → 연령/성별별 반응 차이를 한눈에 비교 가능
    """
    service = _get_service()
    results = []

    for p in request.personas:
        eval_req = EvaluationRequest(
            stimulus=request.stimulus,
            stimulus_type=request.stimulus_type,
            axes=request.axes,
            persona_age_group=p.get("age_group", "30대"),
            persona_gender=p.get("gender", "male"),
            persona_occupation=p.get("occupation", ""),
            persona_income=p.get("income", ""),
            persona_traits=p.get("traits", []),
        )
        result = await service.evaluate(eval_req)
        results.append(result)

    # 축별 평균 점수 계산
    summary = _calculate_summary(results)

    # 모델 타입 판별
    if service.available:
        model_type = "local"
    elif results and results[0].confidence > 0:
        model_type = "claude"
    else:
        model_type = "mock"

    return BatchEvaluationResponse(
        results=results,
        summary=summary,
        model_type=model_type,
    )


@router.get("/axes/{stimulus_type}")
async def get_default_axes(stimulus_type: StimulusType):
    """카테고리별 기본 평가 축 조회"""
    axes = DEFAULT_AXES.get(stimulus_type, DEFAULT_AXES[StimulusType.GENERAL])
    return {
        "stimulus_type": stimulus_type.value,
        "axes": axes,
    }


@router.get("/axes")
async def get_all_axes():
    """전체 카테고리별 평가 축 목록"""
    return {
        st.value: axes
        for st, axes in DEFAULT_AXES.items()
    }


@router.get("/status")
async def get_eval_status():
    """평가 모델 상태 확인"""
    service = _get_service()
    return {
        "embedding_model_available": service.available,
        "model_path": str(service.model_path),
        "rag_available": service._ensure_rag_loaded() if service.available else False,
    }


def _calculate_summary(results: list[EvaluationResponse]) -> dict[str, float]:
    """결과 목록에서 축별 평균 점수 계산"""
    if not results:
        return {}

    axis_scores: dict[str, list[float]] = {}
    for result in results:
        for ev in result.evaluations:
            axis_scores.setdefault(ev.name, []).append(ev.score)

    return {
        name: round(sum(scores) / len(scores), 1)
        for name, scores in axis_scores.items()
    }
