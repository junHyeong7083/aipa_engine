"""
파일 업로드 API 엔드포인트 (Upload API Endpoints)

PDF, CSV, 이미지 파일 업로드 및 텍스트 추출.
Flutter 앱의 upload_screen에서 호출됨.
"""

import csv
import io
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# 업로드 파일 저장 경로
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class UploadResponse(BaseModel):
    """파일 업로드 응답"""
    success: bool
    filename: str
    file_id: str
    extracted_text: Optional[str] = None
    parsed_questions: Optional[list] = None  # Claude가 파싱한 질문/보기
    message: str = ""


# 허용 파일 타입
ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".csv"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


@router.post("/document", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    문서 파일 업로드 (PDF, CSV)

    POST /api/v1/upload/document
    - PDF: 텍스트 추출 (PyMuPDF 사용)
    - CSV: 내용 읽기 및 요약
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 없습니다.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다: {ext}. 지원 형식: {', '.join(ALLOWED_DOCUMENT_EXTENSIONS)}"
        )

    try:
        # 파일 읽기
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="파일 크기가 20MB를 초과합니다.")

        # 파일 저장
        file_id = str(uuid.uuid4())[:8]
        save_filename = f"{file_id}_{file.filename}"
        save_path = UPLOAD_DIR / save_filename
        save_path.write_bytes(content)

        # 텍스트 추출
        extracted_text = ""
        if ext == ".pdf":
            extracted_text = _extract_pdf_text(content)
        elif ext == ".csv":
            extracted_text = _extract_csv_text(content)

        logger.info(f"Document uploaded: {file.filename} ({len(content)} bytes, extracted {len(extracted_text)} chars)")

        # PDF면 Claude로 설문 질문/보기 파싱
        parsed_questions = None
        if ext == ".pdf" and extracted_text:
            parsed_questions = _parse_survey_with_claude(extracted_text)

        return UploadResponse(
            success=True,
            filename=file.filename,
            file_id=file_id,
            extracted_text=extracted_text if extracted_text else None,
            parsed_questions=parsed_questions,
            message=f"문서 업로드 완료 ({len(extracted_text)}자 추출, {len(parsed_questions) if parsed_questions else 0}개 질문 파싱)" if extracted_text else "문서 업로드 완료",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"파일 업로드 실패: {str(e)}")


@router.post("/image", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    이미지 파일 업로드 (JPG, PNG 등)

    POST /api/v1/upload/image
    - 이미지 저장
    - 텍스트 추출은 하지 않음 (향후 OCR 추가 가능)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 없습니다.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 이미지 형식입니다: {ext}. 지원 형식: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
        )

    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="파일 크기가 20MB를 초과합니다.")

        # 파일 저장
        file_id = str(uuid.uuid4())[:8]
        save_filename = f"{file_id}_{file.filename}"
        save_path = UPLOAD_DIR / save_filename
        save_path.write_bytes(content)

        logger.info(f"Image uploaded: {file.filename} ({len(content)} bytes)")

        return UploadResponse(
            success=True,
            filename=file.filename,
            file_id=file_id,
            extracted_text=None,
            message="이미지 업로드 완료",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 업로드 실패: {str(e)}")


def _extract_pdf_text(content: bytes) -> str:
    """
    PDF 파일에서 텍스트 + 테이블 추출

    우선순위:
    1. pdfplumber (텍스트 + 테이블 구조 추출) ← 설문지 파싱에 최적
    2. PyMuPDF (텍스트만) ← fallback
    """
    # 방법 1: pdfplumber (테이블 포함)
    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_parts = [f"[페이지 {i + 1}]"]

                # 텍스트 추출
                text = page.extract_text() or ""
                if text.strip():
                    page_parts.append(text.strip())

                # 테이블 추출 (설문지의 표형 질문 대응)
                try:
                    tables = page.extract_tables()
                    for t_idx, table in enumerate(tables):
                        if not table:
                            continue
                        page_parts.append(f"\n[표 {t_idx + 1}]")
                        for row in table:
                            cells = [cell.strip() if cell else "" for cell in row]
                            if any(cells):
                                page_parts.append(" | ".join(cells))
                except Exception:
                    pass

                text_parts.append("\n".join(page_parts))

        full_text = "\n\n".join(text_parts)
        if len(full_text) > 10000:
            full_text = full_text[:10000] + "\n\n... (이하 생략)"

        logger.info(f"PDF extracted with pdfplumber: {len(full_text)} chars")
        return full_text

    except ImportError:
        logger.info("pdfplumber not available, trying PyMuPDF")

    # 방법 2: PyMuPDF (fallback)
    try:
        import fitz

        doc = fitz.open(stream=content, filetype="pdf")
        text_parts = []
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(f"[페이지 {page_num + 1}]\n{page_text.strip()}")
        doc.close()

        full_text = "\n\n".join(text_parts)
        if len(full_text) > 10000:
            full_text = full_text[:10000] + "\n\n... (이하 생략)"
        return full_text

    except ImportError:
        logger.warning("No PDF library available (pdfplumber or PyMuPDF)")
        return "[PDF 텍스트 추출 불가: PDF 라이브러리가 설치되지 않았습니다]"
    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        return f"[PDF 텍스트 추출 실패: {str(e)}]"


def _parse_survey_with_claude(extracted_text: str) -> Optional[list]:
    """
    Claude API로 설문 텍스트에서 질문/보기를 구조화 추출.
    어떤 형식의 설문지든 범용적으로 파싱.
    """
    try:
        from ..services.llm_service import LLMService
        llm = LLMService()
        if not llm.client:
            logger.warning("Claude API 없음 - 설문 파싱 스킵")
            return None

        prompt = f"""다음은 PDF에서 추출한 설문지 텍스트입니다. 여기서 설문 질문과 보기(선택지)만 추출해주세요.

규칙:
- 안내문, 인사말, 설명, 페이지 번호, 참고 표(채널 목록 등) 무시
- 실제 설문 질문만 추출
- 각 질문의 보기(선택지)도 함께 추출
- 보기가 없는 질문은 주관식으로 처리 (choices를 빈 배열로)
- ★ 중요: 표(매트릭스) 형태 질문은 각 행(문항)을 별도 질문으로 분리하세요.
  예: "Q4. 다음 기능을 어느 정도 이용하십니까? [행별 1개씩 선택]" 아래에 행이 3개 있으면
  → Q4-1, Q4-2, Q4-3 으로 분리하고, 각각 동일한 척도(전혀 그렇지 않다~매우 그렇다)를 choices로 넣으세요.

반드시 아래 JSON 배열 형식으로만 응답하세요. 다른 텍스트 없이 JSON만:
[
  {{"id": "q1", "text": "질문 텍스트", "choices": ["보기1", "보기2", "보기3"]}},
  {{"id": "q2", "text": "질문 텍스트", "choices": []}}
]

설문지 텍스트:
---
{extracted_text[:6000]}
---

JSON:"""

        raw = llm._call_api(prompt, max_tokens=2000)
        raw = raw.strip()

        # JSON 배열 파싱
        import json
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            questions = json.loads(raw[start:end])
            if isinstance(questions, list) and len(questions) > 0:
                # id 재정렬
                for i, q in enumerate(questions):
                    q["id"] = f"q{i+1}"
                logger.info(f"Claude 설문 파싱 성공: {len(questions)}개 질문")
                return questions

        logger.warning(f"Claude 설문 파싱 실패: JSON 파싱 불가")
        return None

    except Exception as e:
        logger.warning(f"Claude 설문 파싱 실패: {e}")
        return None


def _extract_csv_text(content: bytes) -> str:
    """CSV 파일 내용을 텍스트로 변환"""
    try:
        # UTF-8로 시도, 실패하면 CP949(한국어) 시도
        text = None
        for encoding in ["utf-8", "cp949", "euc-kr", "latin-1"]:
            try:
                text = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            return "[CSV 파일 인코딩을 인식할 수 없습니다]"

        reader = csv.reader(io.StringIO(text))
        rows = list(reader)

        if not rows:
            return "[빈 CSV 파일]"

        # 헤더
        header = rows[0]
        total_rows = len(rows) - 1  # 헤더 제외

        # 요약 텍스트 생성
        parts = [
            f"CSV 파일 요약: {total_rows}행 x {len(header)}열",
            f"컬럼: {', '.join(header)}",
            "",
            "처음 10행 데이터:",
        ]

        for i, row in enumerate(rows[1:11], 1):
            row_text = " | ".join(f"{header[j]}: {cell}" for j, cell in enumerate(row) if j < len(header))
            parts.append(f"  {i}. {row_text}")

        if total_rows > 10:
            parts.append(f"  ... (총 {total_rows}행 중 10행만 표시)")

        return "\n".join(parts)

    except Exception as e:
        logger.error(f"CSV text extraction failed: {e}")
        return f"[CSV 파일 읽기 실패: {str(e)}]"
