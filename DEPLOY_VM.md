# AIPA Engine — VM 배포 가이드 (PostgreSQL + Docker Compose)

aishort 백엔드와 동일한 패턴: **자체 VM + PostgreSQL + nip.io HTTPS**.
Cloud Run / Firestore 의존성을 제거했으므로 GCP 프로젝트 권한이 없어도 배포 가능.

> 사전 준비물
> - 공인 IP 가 있는 VM (GCP Compute Engine / 기타). RAM **최소 4GB** (reasoning 0.5B 모델 로딩).
> - VM 방화벽에서 80/443(HTTPS), 그리고 임시로 8080 허용.
> - Docker + Docker Compose plugin 설치.

---

## 1. 소스 업로드

```bash
# 로컬에서 (모델 포함 — training/models 2.8GB 도 함께 전송)
tar --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
    -czf aipa-engine.tgz -C /c/Users/user/Git AIPA_Engine
scp aipa-engine.tgz <user>@<VM_IP>:~/

# VM에서
mkdir -p /opt/aipa-engine && tar -xzf ~/aipa-engine.tgz -C /opt/aipa-engine --strip-components=1
cd /opt/aipa-engine
```

## 2. 환경 변수

```bash
cp .env.example .env
# 아래 값 채우기:
#   DB_PASSWORD=<강한 랜덤>            (openssl rand -hex 16)
#   API_BEARER_TOKEN=<강한 랜덤>       (openssl rand -hex 32)  ← 클라이언트와 동일 값 사용
#   ANTHROPIC_API_KEY / KOSIS_API_KEY / NAVER_* (있으면)
#   ALLOWED_ORIGINS=https://<VM_IP를 -로>.nip.io
nano .env
```

## 3. 기동

```bash
docker compose up -d --build
docker compose logs -f app          # "PostgreSQL 연결 + 스키마 준비 완료" 확인
curl -s http://127.0.0.1:8080/health   # {"status":"healthy",...}
```

테이블(users, survey_history, simulations, pipeline_*, training_data)은 앱 부팅 시 자동 생성됨.

## 4. HTTPS (nip.io + Caddy) — 도메인 없이 TLS

`<VM_IP>` 가 `34.64.36.56` 이면 도메인은 `34-64-36-56.nip.io`.

```bash
# Caddy 설치 (자동 Let's Encrypt)
sudo apt install -y caddy
sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
34-64-36-56.nip.io {
    reverse_proxy 127.0.0.1:8080
}
EOF
sudo systemctl restart caddy
# 이제 https://34-64-36-56.nip.io/health 접속 가능
```

`.env` 의 `ALLOWED_ORIGINS` 에도 같은 nip.io 도메인을 넣어줄 것.

## 5. 클라이언트 연결

flutter_app `.env` 의 `API_BASE_URL` 을 새 주소로 변경 후 앱 리빌드:
```
API_BASE_URL=https://34-64-36-56.nip.io
API_BEARER_TOKEN=<위 3번에서 만든 동일 토큰>
```

---

## 업데이트 배포

```bash
cd /opt/aipa-engine
# (새 소스 업로드 후)
docker compose up -d --build app
```

## 트러블슈팅

| 증상 | 원인/조치 |
|------|-----------|
| `app` 로그에 `PostgreSQL 연결 실패` | `.env` DB_PASSWORD 와 db 서비스 비번 불일치, 또는 db 헬스체크 대기 |
| 쓰기 라우트 503 `인증이 구성되지 않았습니다` | `API_BEARER_TOKEN` 미설정 |
| 모델 로드 실패 → 템플릿 응답 | `training/models` 마운트 경로 확인, VM RAM 부족(4GB+) |
| OOM 으로 컨테이너 재시작 | RAM 증설 또는 reasoning 모델 비활성화 |
