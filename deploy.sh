#!/bin/bash
# LicensePlateControl - Clean Deploy Script v1.2.0
# Usage: ./deploy.sh [--clean]
#   --clean  = Full rebuild (no Docker cache), use if packages changed

set -e
cd "$(dirname "$0")"

echo "========================================="
echo " LicensePlateControl Deploy v1.2.0"
echo "========================================="
echo ""

# Check we're on the right branch
BRANCH=$(git branch --show-current)
echo "Git Branch: $BRANCH"
echo "Git Commit: $(git rev-parse --short HEAD)"
echo ""

# Ensure volumes directory exists
mkdir -p volumes/models volumes/events

# Check if --clean flag is set
if [ "$1" = "--clean" ]; then
    echo "[1/4] Stopping all containers..."
    docker compose down
    echo ""
    echo "[2/4] Rebuilding ALL images (no cache)..."
    docker compose build --no-cache backend engine frontend
else
    echo "[1/4] Stopping all containers..."
    docker compose down
    echo ""
    echo "[2/4] Rebuilding images..."
    docker compose build backend engine frontend
fi

echo ""
echo "[3/4] Starting all services..."
docker compose up -d

echo ""
echo "[4/4] Waiting for services to start..."
sleep 5

# Health checks
echo ""
echo "========================================="
echo " Health Checks"
echo "========================================="

# Engine health
echo -n "Engine:   "
ENGINE_HEALTH=$(curl -s http://localhost:8001/health 2>/dev/null || echo '{"error":"not reachable"}')
echo "$ENGINE_HEALTH"

# Backend health (check via RTSP status)
echo -n "Backend:  "
BACKEND_HEALTH=$(curl -s http://localhost:8002/api/rtsp/status 2>/dev/null || echo '{"error":"not reachable"}')
if echo "$BACKEND_HEALTH" | grep -q "state"; then
    echo "OK"
else
    echo "ERROR - $BACKEND_HEALTH"
fi

# Frontend health
echo -n "Frontend: "
FRONTEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:80/ 2>/dev/null || echo "000")
if [ "$FRONTEND_STATUS" = "200" ]; then
    echo "OK (HTTP $FRONTEND_STATUS)"
else
    echo "ERROR (HTTP $FRONTEND_STATUS)"
fi

# Check if ALPR is loaded
echo ""
if echo "$ENGINE_HEALTH" | grep -q '"alpr_loaded":true'; then
    echo "ALPR: LOADED - Kennzeichenerkennung aktiv"
elif echo "$ENGINE_HEALTH" | grep -q '"mock_mode":true'; then
    echo "ALPR: MOCK MODE - Kennzeichenerkennung NICHT aktiv!"
    echo ""
    echo "Moegliche Ursachen:"
    echo "  1. Modell-Download fehlgeschlagen (Internet im Container?)"
    echo "  2. GPU-Passthrough Problem (/dev/dri)"
    echo "  3. Package-Fehler"
    echo ""
    echo "Diagnostik:"
    echo "  docker logs licenseplatecontrol-engine"
    echo "  curl http://localhost:8001/health"
    echo "  curl -X POST http://localhost:8001/reload  (retry)"
fi

echo ""
echo "========================================="
echo " Deploy abgeschlossen!"
echo "========================================="
echo " Frontend: http://localhost"
echo " Backend:  http://localhost:8002"
echo " Engine:   http://localhost:8001"
echo "========================================="
