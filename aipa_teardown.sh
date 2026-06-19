#!/usr/bin/env bash
# ============================================================
# AIPA 시연 후 VM에서 AIPA 관련 흔적을 한 번에 제거.
# (aishort 등 다른 서비스에는 영향 없음 — 'aipa' 네임스페이스/전용 파일만 건드림)
#
# VM에서 실행:  bash ~/aipa/aipa_teardown.sh
# ============================================================
set -uo pipefail

APP_DIR="${APP_DIR:-$HOME/aipa}"
NGINX_CONF="/etc/nginx/sites-enabled/aipa.conf"
NGINX_AVAIL="/etc/nginx/sites-available/aipa.conf"

echo "[teardown] 1) docker compose 프로젝트 'aipa' 제거 (컨테이너+볼륨+이미지+네트워크)"
if [ -f "$APP_DIR/docker-compose.yml" ]; then
  (cd "$APP_DIR" && sudo docker compose -p aipa down -v --rmi local) || true
fi
# 혹시 남은 aipa_* 리소스 강제 정리
sudo docker ps -aq --filter "name=aipa" | xargs -r sudo docker rm -f 2>/dev/null || true
sudo docker volume ls -q --filter "name=aipa" | xargs -r sudo docker volume rm 2>/dev/null || true
sudo docker network ls -q --filter "name=aipa" | xargs -r sudo docker network rm 2>/dev/null || true

echo "[teardown] 2) nginx AIPA 블록 제거 + reload"
sudo rm -f "$NGINX_CONF" "$NGINX_AVAIL"
sudo nginx -t 2>/dev/null && sudo systemctl reload nginx || true

echo "[teardown] 3) 앱 디렉터리 삭제: $APP_DIR"
rm -rf "$APP_DIR"

echo "[teardown] 완료. (GCP 방화벽 규칙은 로컬에서 별도 삭제:"
echo "    gcloud compute firewall-rules delete aipa-allow-8443 --quiet )"
echo "[teardown] Docker 엔진 자체는 남겨둠 (필요시: sudo apt-get remove -y docker-ce 등)"
