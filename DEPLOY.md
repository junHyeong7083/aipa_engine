# AIPA Engine - GCP Cloud Run 배포 가이드

## 1. 로컬 테스트

### Python 가상환경에서 실행
```bash
cd c:\Users\user\Git\AIPA_Engine
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -e .
uvicorn aipa_engine.main:app --host 0.0.0.0 --port 8080 --reload
# http://localhost:8080/health 에서 확인
# http://localhost:8080/docs 에서 Swagger UI (FastAPI 자동 생성)
```

### Docker로 로컬 테스트
```bash
docker build -t aipa-engine .
docker run -p 8080:8080 \
  -e ANTHROPIC_API_KEY=your_key \
  -e KOSIS_API_KEY=your_key \
  aipa-engine
# http://localhost:8080/health 에서 확인
```

## 2. GCP 배포

### 사전 준비

#### Google Cloud CLI 설치
https://cloud.google.com/sdk/docs/install 에서 다운로드 후 설치

#### GCP 프로젝트 설정
```bash
gcloud auth login
gcloud config set project aipa-ceca3
```

#### 필요한 API 활성화
```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

### 배포 (한 줄로 끝)

프로젝트 루트에서 실행:

```bash
gcloud run deploy aipa-engine \
  --source . \
  --region asia-northeast3 \
  --allow-unauthenticated \
  --set-env-vars="ANTHROPIC_API_KEY=your_key,KOSIS_API_KEY=your_key,ANTHROPIC_MODEL=claude-sonnet-4-20250514"
```

이 명령어가 자동으로:
1. Docker 이미지 빌드 (Dockerfile 사용)
2. Artifact Registry에 이미지 푸시
3. Cloud Run에 배포

## 3. 배포 완료 후

배포가 끝나면 URL이 출력됨:
```
Service URL: https://aipa-engine-xxxxx-du.a.run.app
```

이 URL이 API 서버 주소!

## 4. API 테스트

```bash
# 헬스체크
curl https://aipa-engine-xxxxx-du.a.run.app/health

# 페르소나 생성
curl -X POST https://aipa-engine-xxxxx-du.a.run.app/api/v1/personas/generate \
  -H "Content-Type: application/json" \
  -d '{"panel_count": 5}'

# 시뮬레이션 생성
curl -X POST https://aipa-engine-xxxxx-du.a.run.app/api/v1/simulations \
  -H "Content-Type: application/json" \
  -d '{"config": {"panel_count": 5}, "questions": [{"text": "테스트 질문", "question_type": "single_choice", "choices": ["A", "B", "C"]}]}'
```

## 5. 환경변수 업데이트 (배포 후)

```bash
gcloud run services update aipa-engine \
  --region asia-northeast3 \
  --set-env-vars="ANTHROPIC_API_KEY=new_key"
```

## 6. Flutter 앱 연결

Flutter `.env` 파일에 Cloud Run URL 설정:
```
API_BASE_URL=https://aipa-engine-xxxxx-du.a.run.app
```
