"""API routes"""

from fastapi import APIRouter

from .simulations import router as simulations_router
from .personas import router as personas_router
from .statistics import router as statistics_router
from .monitoring import router as monitoring_router
from .chat import router as chat_router
from .upload import router as upload_router
from .evaluations import router as evaluations_router
from .platforms import router as platforms_router
from .users import router as users_router

router = APIRouter()

router.include_router(users_router, prefix="/users", tags=["users"])
router.include_router(simulations_router, prefix="/simulations", tags=["simulations"])
router.include_router(personas_router, prefix="/personas", tags=["personas"])
router.include_router(statistics_router, prefix="/statistics", tags=["statistics"])
router.include_router(monitoring_router)
router.include_router(chat_router, prefix="/chat", tags=["chat"])
router.include_router(upload_router, prefix="/upload", tags=["upload"])
router.include_router(evaluations_router, prefix="/evaluations", tags=["evaluations"])
router.include_router(platforms_router)  # 이미 prefix='/platforms' 내장
