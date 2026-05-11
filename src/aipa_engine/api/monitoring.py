"""
AIPA Monitoring Dashboard API

Lightweight in-memory metrics collection and health-check endpoints.
No external dependencies (no Prometheus/Grafana) required.
"""
from __future__ import annotations

import json
import logging
import time
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

logger = logging.getLogger("aipa.monitoring")

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAG_DIR = Path(__file__).resolve().parents[3] / "rag"
RAG_DB_PATH = str(RAG_DIR / "chroma_db")
RAG_METADATA_PATH = RAG_DIR / "index_metadata.json"

_APP_START_TIME: float = time.time()

# ---------------------------------------------------------------------------
# Ring-buffer log handler  (captures structured log entries in memory)
# ---------------------------------------------------------------------------
_MAX_LOG_ENTRIES = 500


class _LogEntry:
    __slots__ = ("timestamp", "level", "message", "logger_name")

    def __init__(self, timestamp: str, level: str, message: str, logger_name: str):
        self.timestamp = timestamp
        self.level = level
        self.message = message
        self.logger_name = logger_name

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "logger": self.logger_name,
            "message": self.message,
        }


class RingBufferLogHandler(logging.Handler):
    """Logging handler that stores the last N log records in a deque."""

    def __init__(self, capacity: int = _MAX_LOG_ENTRIES):
        super().__init__()
        self._buffer: deque[_LogEntry] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        entry = _LogEntry(
            timestamp=datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            level=record.levelname,
            message=self.format(record),
            logger_name=record.name,
        )
        with self._lock:
            self._buffer.append(entry)

    def get_entries(self, level: Optional[str] = None, limit: int = 100) -> list[dict]:
        with self._lock:
            entries = list(self._buffer)
        if level:
            level_upper = level.upper()
            entries = [e for e in entries if e.level == level_upper]
        # Return most recent first, capped by limit
        return [e.to_dict() for e in reversed(entries)][:limit]


# Singleton handler - attach to root logger on import
ring_buffer_handler = RingBufferLogHandler()
ring_buffer_handler.setLevel(logging.DEBUG)
ring_buffer_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(ring_buffer_handler)


# ---------------------------------------------------------------------------
# MetricsCollector (singleton)
# ---------------------------------------------------------------------------

class MetricsCollector:
    """In-memory request / simulation metrics tracker.

    Thread-safe via a simple lock. Designed to be hooked into FastAPI
    middleware so that every request is automatically tracked.
    """

    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsCollector":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_state()
            return cls._instance

    # noinspection PyAttributeOutsideInit
    def _init_state(self) -> None:
        self._req_lock = threading.Lock()
        self.total_requests: int = 0
        self.by_endpoint: dict[str, int] = {}
        self.by_status: dict[str, int] = {}  # "2xx", "4xx", "5xx"
        self._latencies: deque[float] = deque(maxlen=5000)

        self.simulations_total: int = 0
        self.simulations_completed: int = 0
        self.simulations_failed: int = 0
        self._sim_durations: deque[float] = deque(maxlen=1000)

    # -- request tracking --

    def record_request(self, path: str, status_code: int, latency_ms: float) -> None:
        bucket = f"{status_code // 100}xx"
        with self._req_lock:
            self.total_requests += 1
            self.by_endpoint[path] = self.by_endpoint.get(path, 0) + 1
            self.by_status[bucket] = self.by_status.get(bucket, 0) + 1
            self._latencies.append(latency_ms)

    @property
    def avg_latency_ms(self) -> float:
        with self._req_lock:
            if not self._latencies:
                return 0.0
            return sum(self._latencies) / len(self._latencies)

    # -- simulation tracking --

    def record_simulation_start(self) -> None:
        with self._req_lock:
            self.simulations_total += 1

    def record_simulation_end(self, success: bool, duration_seconds: float) -> None:
        with self._req_lock:
            if success:
                self.simulations_completed += 1
            else:
                self.simulations_failed += 1
            self._sim_durations.append(duration_seconds)

    @property
    def avg_simulation_duration(self) -> float:
        with self._req_lock:
            if not self._sim_durations:
                return 0.0
            return sum(self._sim_durations) / len(self._sim_durations)

    # -- snapshot for API --

    def snapshot(self) -> dict:
        with self._req_lock:
            return {
                "requests": {
                    "total": self.total_requests,
                    "by_endpoint": dict(self.by_endpoint),
                    "by_status": dict(self.by_status),
                    "avg_latency_ms": round(self.avg_latency_ms, 1),
                },
                "simulations": {
                    "total_run": self.simulations_total,
                    "completed": self.simulations_completed,
                    "failed": self.simulations_failed,
                    "avg_duration_seconds": round(self.avg_simulation_duration, 2),
                },
            }


# Module-level convenience accessor
metrics = MetricsCollector()


# ---------------------------------------------------------------------------
# RAG stats helper
# ---------------------------------------------------------------------------

def _get_rag_stats() -> dict:
    """Return RAG document counts + last-updated timestamp.

    Tries to read from ChromaDB directly; falls back to metadata file.
    """
    result: dict = {
        "total_documents": 0,
        "last_updated": None,
        "collections": {},
    }

    # Try loading metadata file first (always available, even if chroma is not)
    if RAG_METADATA_PATH.exists():
        try:
            with open(RAG_METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            result["last_updated"] = meta.get("last_indexed_iso")
            stats = meta.get("stats", {})
            result["total_documents"] = stats.get("naver_total", 0) + stats.get("kosis_total", 0)
            result["collections"] = {
                "naver_trends": stats.get("naver_total", 0),
                "kosis_stats": stats.get("kosis_total", 0),
            }
            return result
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback: query ChromaDB directly
    try:
        import chromadb
        client = chromadb.PersistentClient(path=RAG_DB_PATH)
        collections = {c.name: c.count() for c in client.list_collections()}
        result["collections"] = collections
        result["total_documents"] = sum(collections.values())
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Health-check helpers
# ---------------------------------------------------------------------------

async def _check_anthropic() -> dict:
    """Quick connectivity check for the Anthropic API."""
    from ..config import get_settings
    settings = get_settings()
    if not settings.anthropic_api_key:
        return {"status": "error", "latency_ms": 0, "detail": "API key not configured"}
    try:
        import httpx
        start = time.time()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": settings.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
        latency = (time.time() - start) * 1000
        status = "ok" if resp.status_code < 400 else "error"
        return {"status": status, "latency_ms": round(latency)}
    except Exception as e:
        return {"status": "error", "latency_ms": 0, "detail": str(e)}


async def _check_kosis() -> dict:
    """Quick connectivity check for the KOSIS API."""
    from ..config import get_settings
    settings = get_settings()
    if not settings.kosis_api_key:
        return {"status": "error", "latency_ms": 0, "detail": "API key not configured"}
    try:
        import httpx
        start = time.time()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://kosis.kr/openapi/Param/statisticsParameterData.do",
                params={"method": "getList", "apiKey": settings.kosis_api_key, "format": "json"},
            )
        latency = (time.time() - start) * 1000
        status = "ok" if resp.status_code < 500 else "error"
        return {"status": status, "latency_ms": round(latency)}
    except Exception as e:
        return {"status": "error", "latency_ms": 0, "detail": str(e)}


def _check_firestore() -> dict:
    """Check if Firestore credentials are available."""
    try:
        import firebase_admin
        from firebase_admin import firestore
        app = firebase_admin.get_app()
        return {"status": "ok"}
    except Exception:
        return {"status": "unavailable"}


def _check_rag_index() -> dict:
    """Check RAG index status."""
    stats = _get_rag_stats()
    doc_count = stats.get("total_documents", 0)
    status = "ok" if doc_count > 0 else "empty"
    return {"status": status, "document_count": doc_count}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Visual monitoring dashboard."""
    return _DASHBOARD_HTML


@router.get("/health")
async def health():
    """Detailed health check with external dependency status."""
    anthropic_check = await _check_anthropic()
    kosis_check = await _check_kosis()
    firestore_check = _check_firestore()
    rag_check = _check_rag_index()

    checks = {
        "anthropic_api": anthropic_check,
        "kosis_api": kosis_check,
        "firestore": firestore_check,
        "rag_index": rag_check,
    }

    # Determine overall status
    error_count = sum(
        1 for c in checks.values()
        if c.get("status") in ("error", "unavailable", "empty")
    )
    if error_count == 0:
        overall = "healthy"
    elif error_count <= 2:
        overall = "degraded"
    else:
        overall = "unhealthy"

    return {
        "status": overall,
        "uptime_seconds": round(time.time() - _APP_START_TIME),
        "checks": checks,
    }


@router.get("/metrics")
async def get_metrics():
    """Operational metrics snapshot."""
    snap = metrics.snapshot()
    snap["rag"] = _get_rag_stats()
    return snap


@router.get("/logs")
async def get_logs(
    level: Optional[str] = Query(None, description="Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
    limit: int = Query(100, ge=1, le=500, description="Max number of log entries to return"),
):
    """Recent structured log entries from the in-memory ring buffer."""
    entries = ring_buffer_handler.get_entries(level=level, limit=limit)
    return {"count": len(entries), "entries": entries}


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------
_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIPA Engine Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; }
  h1 { font-size: 1.5rem; margin-bottom: 20px; color: #38bdf8; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }
  .card h2 { font-size: 0.875rem; text-transform: uppercase; color: #94a3b8; margin-bottom: 12px; letter-spacing: 0.05em; }
  .status-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
  .status-healthy, .status-ok { background: #064e3b; color: #34d399; }
  .status-degraded { background: #78350f; color: #fbbf24; }
  .status-unhealthy, .status-error, .status-unavailable { background: #7f1d1d; color: #f87171; }
  .status-empty { background: #3b3b3b; color: #a1a1aa; }
  .check-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #334155; }
  .check-row:last-child { border-bottom: none; }
  .check-name { font-weight: 500; }
  .check-latency { color: #94a3b8; font-size: 0.8rem; margin-left: 8px; }
  .metric-big { font-size: 2rem; font-weight: 700; color: #f1f5f9; }
  .metric-label { font-size: 0.8rem; color: #64748b; }
  .metric-row { display: flex; gap: 24px; flex-wrap: wrap; }
  .metric-item { text-align: center; min-width: 80px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  th { text-align: left; color: #94a3b8; padding: 6px 8px; border-bottom: 1px solid #334155; }
  td { padding: 6px 8px; border-bottom: 1px solid #1e293b; }
  .log-error { color: #f87171; }
  .log-warning { color: #fbbf24; }
  .log-info { color: #38bdf8; }
  .log-debug { color: #94a3b8; }
  .topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
  .refresh-btn { background: #334155; color: #e2e8f0; border: 1px solid #475569; padding: 8px 16px; border-radius: 8px; cursor: pointer; font-size: 0.85rem; }
  .refresh-btn:hover { background: #475569; }
  #last-update { font-size: 0.8rem; color: #64748b; }
  .endpoint-bar { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
  .endpoint-name { font-size: 0.75rem; color: #94a3b8; width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .bar-bg { flex: 1; height: 8px; background: #334155; border-radius: 4px; }
  .bar-fill { height: 100%; background: #38bdf8; border-radius: 4px; transition: width 0.3s; }
  .bar-count { font-size: 0.75rem; color: #64748b; width: 40px; text-align: right; }
</style>
</head>
<body>
<div class="topbar">
  <h1>AIPA Engine Monitor</h1>
  <div>
    <span id="last-update"></span>
    <button class="refresh-btn" onclick="loadAll()">Refresh</button>
  </div>
</div>

<div class="grid" id="health-section">
  <div class="card">
    <h2>System Status</h2>
    <div id="health-content">Loading...</div>
  </div>
  <div class="card">
    <h2>Request Metrics</h2>
    <div id="metrics-content">Loading...</div>
  </div>
  <div class="card">
    <h2>RAG Index</h2>
    <div id="rag-content">Loading...</div>
  </div>
  <div class="card">
    <h2>Simulations</h2>
    <div id="sim-content">Loading...</div>
  </div>
</div>

<div class="grid">
  <div class="card" style="grid-column: 1 / -1;">
    <h2>Top Endpoints</h2>
    <div id="endpoints-content">Loading...</div>
  </div>
</div>

<div class="card">
  <h2>Recent Logs</h2>
  <div id="logs-content" style="max-height:400px;overflow-y:auto;">Loading...</div>
</div>

<script>
const BASE = window.location.origin + '/api/v1/monitoring';

function badge(status) {
  return `<span class="status-badge status-${status}">${status}</span>`;
}

async function loadHealth() {
  try {
    const r = await fetch(BASE + '/health');
    const d = await r.json();
    let html = `<div style="margin-bottom:12px">${badge(d.status)} <span style="margin-left:8px;color:#94a3b8;font-size:0.85rem">Uptime: ${formatUptime(d.uptime_seconds)}</span></div>`;
    for (const [name, check] of Object.entries(d.checks)) {
      html += `<div class="check-row">
        <span class="check-name">${name.replace(/_/g, ' ')}</span>
        <span>${badge(check.status)}${check.latency_ms ? `<span class="check-latency">${check.latency_ms}ms</span>` : ''}${check.document_count !== undefined ? `<span class="check-latency">${check.document_count} docs</span>` : ''}</span>
      </div>`;
    }
    document.getElementById('health-content').innerHTML = html;
  } catch(e) { document.getElementById('health-content').innerHTML = 'Error loading'; }
}

async function loadMetrics() {
  try {
    const r = await fetch(BASE + '/metrics');
    const d = await r.json();
    const req = d.requests;
    document.getElementById('metrics-content').innerHTML = `
      <div class="metric-row">
        <div class="metric-item"><div class="metric-big">${req.total}</div><div class="metric-label">Total Requests</div></div>
        <div class="metric-item"><div class="metric-big">${req.avg_latency_ms}</div><div class="metric-label">Avg Latency (ms)</div></div>
        <div class="metric-item"><div class="metric-big" style="color:#34d399">${req.by_status['2xx']||0}</div><div class="metric-label">2xx</div></div>
        <div class="metric-item"><div class="metric-big" style="color:#fbbf24">${req.by_status['4xx']||0}</div><div class="metric-label">4xx</div></div>
        <div class="metric-item"><div class="metric-big" style="color:#f87171">${req.by_status['5xx']||0}</div><div class="metric-label">5xx</div></div>
      </div>`;

    // Endpoints
    const eps = Object.entries(req.by_endpoint).sort((a,b) => b[1]-a[1]).slice(0,10);
    const maxCount = eps.length ? eps[0][1] : 1;
    let epHtml = '';
    for (const [path, count] of eps) {
      const pct = (count / maxCount * 100).toFixed(0);
      epHtml += `<div class="endpoint-bar">
        <span class="endpoint-name" title="${path}">${path}</span>
        <div class="bar-bg"><div class="bar-fill" style="width:${pct}%"></div></div>
        <span class="bar-count">${count}</span>
      </div>`;
    }
    document.getElementById('endpoints-content').innerHTML = epHtml || '<span style="color:#64748b">No data yet</span>';

    // RAG
    const rag = d.rag;
    document.getElementById('rag-content').innerHTML = `
      <div class="metric-row">
        <div class="metric-item"><div class="metric-big">${rag.total_documents}</div><div class="metric-label">Total Documents</div></div>
      </div>
      <div style="margin-top:12px;font-size:0.85rem;color:#94a3b8">
        ${Object.entries(rag.collections||{}).map(([k,v]) => `${k}: ${v}`).join(' &middot; ')}
        ${rag.last_updated ? `<br>Updated: ${rag.last_updated.slice(0,19)}` : ''}
      </div>`;

    // Simulations
    const sim = d.simulations;
    document.getElementById('sim-content').innerHTML = `
      <div class="metric-row">
        <div class="metric-item"><div class="metric-big">${sim.total_run}</div><div class="metric-label">Total</div></div>
        <div class="metric-item"><div class="metric-big" style="color:#34d399">${sim.completed}</div><div class="metric-label">Completed</div></div>
        <div class="metric-item"><div class="metric-big" style="color:#f87171">${sim.failed}</div><div class="metric-label">Failed</div></div>
        <div class="metric-item"><div class="metric-big">${sim.avg_duration_seconds}s</div><div class="metric-label">Avg Duration</div></div>
      </div>`;
  } catch(e) { document.getElementById('metrics-content').innerHTML = 'Error loading'; }
}

async function loadLogs() {
  try {
    const r = await fetch(BASE + '/logs?limit=50');
    const d = await r.json();
    if (!d.entries.length) { document.getElementById('logs-content').innerHTML = '<span style="color:#64748b">No logs</span>'; return; }
    let html = '<table><tr><th>Time</th><th>Level</th><th>Logger</th><th>Message</th></tr>';
    for (const e of d.entries) {
      const cls = 'log-' + e.level.toLowerCase();
      const time = e.timestamp.slice(11, 23);
      const msg = e.message.length > 120 ? e.message.slice(0,120) + '...' : e.message;
      html += `<tr><td>${time}</td><td class="${cls}">${e.level}</td><td style="color:#64748b">${e.logger}</td><td>${escapeHtml(msg)}</td></tr>`;
    }
    html += '</table>';
    document.getElementById('logs-content').innerHTML = html;
  } catch(e) { document.getElementById('logs-content').innerHTML = 'Error loading'; }
}

function escapeHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function formatUptime(s) {
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60);
  return h > 0 ? `${h}h ${m}m` : `${m}m ${s%60}s`;
}

function loadAll() {
  loadHealth(); loadMetrics(); loadLogs();
  document.getElementById('last-update').textContent = 'Updated: ' + new Date().toLocaleTimeString();
}

loadAll();
setInterval(loadAll, 15000);
</script>
</body>
</html>
"""
