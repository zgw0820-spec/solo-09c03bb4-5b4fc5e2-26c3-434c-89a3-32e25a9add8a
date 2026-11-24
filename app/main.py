"""FastAPI 入口：RPC 双程耗时分析的网页与接口。

接口
----
* ``POST /api/analyze``：保留原用法，请求体 ``{"limit_us": 200, "csv_text": "..."}``，
  服务端解析整份 CSV（兼容旧客户端与测试）。
* ``POST /api/analyze_summary``：网页新用法。调用号与原始时刻留在浏览器，只提交按
  有向主机对聚合的约束摘要 ``{"limit_us": 200, "hosts": [...], "constraints": [...]}``，
  约束条数按主机对计上限，不随每对主机的调用数增长。
"""

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import solver

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "static" / "index.html"

app = FastAPI(title="RPC 双程耗时分析", version="2.0.0")


class AnalyzeRequest(BaseModel):
    limit_us: int
    csv_text: str


class SummaryRequest(BaseModel):
    limit_us: int
    hosts: list[str]
    constraints: list[dict[str, Any]]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX_HTML)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
def analyze_api(req: AnalyzeRequest):
    """原用法：整份 CSV 上送，服务端解析、分析，响应结构不变。"""
    if req.limit_us < 0:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "单程耗时上限必须是非负整数（微秒）"},
        )
    try:
        calls = solver.parse_csv(req.csv_text)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(exc)})
    return {"ok": True, "result": solver.analyze(calls, req.limit_us)}


@app.post("/api/analyze_summary")
def analyze_summary_api(req: SummaryRequest):
    """新用法：只接收约束摘要（主机名 + 差值 + 原表行号），不接收调用号与原始时刻。"""
    if req.limit_us < 0:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "单程耗时上限必须是非负整数（微秒）"},
        )
    try:
        solver.validate_summary(req.hosts, req.constraints)
        result = solver.analyze_constraints(req.hosts, req.constraints, req.limit_us)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(exc)})
    return {"ok": True, "result": result}
