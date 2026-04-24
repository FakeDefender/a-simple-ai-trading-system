"""内置 Web 控制台入口。"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import sys
import threading

if __package__ in {None, ''}:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
import traceback
import uuid
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, unquote, urlparse

from src.app_service import (
    get_config_path,
    get_file_preview,
    get_project_root,
    get_results_root,
    get_run_chart_data,
    get_run_details,
    list_result_runs,
    load_config_text,
    load_runtime_config,
    parse_config_text,
    parse_runtime_config_text,
    patch_config_text,
    render_config_text,
    run_live_pipeline,
    run_main_pipeline,
    run_paper_pipeline,
    save_config_text,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui")


class _ThreadLogFilter(logging.Filter):
    def __init__(self, thread_id: int):
        super().__init__()
        self.thread_id = thread_id

    def filter(self, record: logging.LogRecord) -> bool:
        return int(getattr(record, "thread", -1)) == self.thread_id


class JobLogBuffer:
    def __init__(self, max_lines: int = 400):
        self._lock = threading.Lock()
        self._lines = deque(maxlen=max_lines)
        self._partial = ""

    def write(self, text: str):
        if not text:
            return 0
        normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
        with self._lock:
            self._partial += normalized
            while "\n" in self._partial:
                line, self._partial = self._partial.split("\n", 1)
                self._lines.append(line)
        return len(text)

    def flush(self):
        return None

    def emit(self, message: str):
        timestamp = datetime.now().isoformat(timespec="seconds")
        self.write(f"{timestamp} | {message}\n")

    def snapshot(self, limit: int = 200) -> Dict[str, Any]:
        with self._lock:
            lines = list(self._lines)
            partial = self._partial
        if partial:
            lines = lines + [partial]
        limit = max(int(limit), 1)
        return {
            "logs": lines[-limit:],
            "line_count": len(lines),
            "truncated": len(lines) > limit,
        }


class JobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._job_logs: Dict[str, JobLogBuffer] = {}

    def list_jobs(self):
        with self._lock:
            jobs = [dict(item) for item in self._jobs.values()]
        jobs.sort(key=lambda item: item["created_at"], reverse=True)
        return jobs

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            log_buffer = self._job_logs.get(job_id)
        if job is None:
            return None
        payload = dict(job)
        if log_buffer is not None:
            payload["logs_tail"] = log_buffer.snapshot(limit=20)["logs"]
        return payload

    def get_job_logs(self, job_id: str, limit: int = 200) -> Optional[Dict[str, Any]]:
        with self._lock:
            log_buffer = self._job_logs.get(job_id)
        if log_buffer is None:
            return None
        payload = log_buffer.snapshot(limit=limit)
        payload["job_id"] = job_id
        return payload

    def start_job(
        self,
        mode: str,
        config_text: str,
        save_config: bool = False,
        force_update: Optional[bool] = None,
    ) -> Dict[str, Any]:
        config = parse_runtime_config_text(config_text)
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "mode": mode,
            "status": "queued",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "started_at": None,
            "ended_at": None,
            "error": None,
            "result": None,
            "save_config": bool(save_config),
        }
        log_buffer = JobLogBuffer()
        log_buffer.emit(f"任务已进入队列，模式={mode}")
        with self._lock:
            self._jobs[job_id] = job
            self._job_logs[job_id] = log_buffer

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, mode, config_text, config, bool(save_config), force_update),
            daemon=True,
        )
        thread.start()
        return dict(job)

    def _update_job(self, job_id: str, **kwargs):
        with self._lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id].update(kwargs)

    def _log(self, job_id: str, message: str):
        with self._lock:
            log_buffer = self._job_logs.get(job_id)
        if log_buffer is not None:
            log_buffer.emit(message)

    def _run_job(
        self,
        job_id: str,
        mode: str,
        config_text: str,
        config: Dict[str, Any],
        save_config: bool,
        force_update: Optional[bool],
    ):
        self._update_job(job_id, status="running", started_at=datetime.now().isoformat(timespec="seconds"))
        self._log(job_id, f"任务开始执行，模式={mode}")

        root_logger = logging.getLogger()
        thread_handler = logging.StreamHandler(self._job_logs[job_id])
        thread_handler.setLevel(logging.INFO)
        thread_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        thread_handler.addFilter(_ThreadLogFilter(threading.get_ident()))
        root_logger.addHandler(thread_handler)

        try:
            if save_config:
                save_config_text(config_text)
                self._log(job_id, f"当前 YAML 已写回磁盘: {get_config_path()}")

            if mode == "main":
                self._log(job_id, "开始执行研究主流程")
                result = run_main_pipeline(config=config)
            elif mode == "paper":
                self._log(job_id, "开始执行 Paper Trading")
                result = run_paper_pipeline(config=config)
            elif mode == "live":
                self._log(job_id, "开始执行 Live Dry Run")
                result = run_live_pipeline(config=config, force_update=force_update)
            else:
                raise ValueError(f"不支持的任务类型: {mode}")

            summary = result.get("summary", {}) or {}
            preview = ", ".join(
                f"{key}={value}"
                for key, value in list(summary.items())[:3]
            )
            if result.get("relative_output_dir"):
                self._log(job_id, f"结果目录: {result['relative_output_dir']}")
            if preview:
                self._log(job_id, f"任务摘要: {preview}")
            self._log(job_id, "任务执行完成")
            self._update_job(
                job_id,
                status="completed",
                ended_at=datetime.now().isoformat(timespec="seconds"),
                result=result,
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Web 控制台任务失败: %s", exc)
            self._log(job_id, f"任务失败: {exc}")
            self._update_job(
                job_id,
                status="failed",
                ended_at=datetime.now().isoformat(timespec="seconds"),
                error={
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        finally:
            root_logger.removeHandler(thread_handler)


JOB_MANAGER = JobManager()


def _sanitize_config_for_response(value):
    sensitive_keys = {"api_key", "secret", "secret_key", "api_secret", "password", "access_token", "refresh_token"}
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in sensitive_keys or lowered.endswith("_api_key"):
                sanitized[key] = "***已配置***" if item else ""
            else:
                sanitized[key] = _sanitize_config_for_response(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_config_for_response(item) for item in value]
    return value


class TradingConsoleHandler(BaseHTTPRequestHandler):
    server_version = "TradingConsole/0.2"

    def log_message(self, format, *args):  # noqa: A003
        logger.debug("%s - %s", self.address_string(), format % args)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            return self._write_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "config_path": get_config_path(),
                    "results_root": get_results_root(),
                    "server_time": datetime.now().isoformat(timespec="seconds"),
                },
            )
        if parsed.path == "/api/config":
            return self._write_json(
                HTTPStatus.OK,
                {
                    "path": get_config_path(),
                    "text": load_config_text(),
                    "config": _sanitize_config_for_response(load_runtime_config()),
                },
            )
        if parsed.path == "/api/jobs":
            return self._write_json(HTTPStatus.OK, {"jobs": JOB_MANAGER.list_jobs()})
        if parsed.path.startswith("/api/jobs/"):
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "logs":
                limit = self._read_int_query(parsed.query, "limit", default=160, minimum=10, maximum=500)
                job = JOB_MANAGER.get_job_logs(parts[2], limit=limit)
                if job is None:
                    return self._write_json(HTTPStatus.NOT_FOUND, {"error": "任务不存在"})
                return self._write_json(HTTPStatus.OK, job)
            if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                job = JOB_MANAGER.get_job(parts[2])
                if job is None:
                    return self._write_json(HTTPStatus.NOT_FOUND, {"error": "任务不存在"})
                return self._write_json(HTTPStatus.OK, job)
        if parsed.path == "/api/results":
            limit = self._read_int_query(parsed.query, "limit", default=20, minimum=1, maximum=200)
            return self._write_json(HTTPStatus.OK, {"runs": list_result_runs(limit=limit)})
        if parsed.path == "/api/result":
            query = parse_qs(parsed.query)
            relative_path = query.get("path", [""])[0]
            if not relative_path:
                return self._write_json(HTTPStatus.BAD_REQUEST, {"error": "缺少 path 参数"})
            try:
                return self._write_json(HTTPStatus.OK, get_run_details(relative_path))
            except Exception as exc:
                return self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        if parsed.path == "/api/result/chart":
            query = parse_qs(parsed.query)
            relative_path = query.get("path", [""])[0]
            if not relative_path:
                return self._write_json(HTTPStatus.BAD_REQUEST, {"error": "缺少 path 参数"})
            try:
                return self._write_json(HTTPStatus.OK, get_run_chart_data(relative_path))
            except Exception as exc:
                return self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        if parsed.path == "/api/file-preview":
            query = parse_qs(parsed.query)
            relative_path = query.get("path", [""])[0]
            if not relative_path:
                return self._write_json(HTTPStatus.BAD_REQUEST, {"error": "缺少 path 参数"})
            limit = self._read_int_query(parsed.query, "limit", default=50, minimum=1, maximum=200)
            try:
                return self._write_json(HTTPStatus.OK, get_file_preview(relative_path, limit=limit))
            except Exception as exc:
                return self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        if parsed.path.startswith("/files/"):
            relative_path = unquote(parsed.path[len("/files/") :])
            return self._serve_project_file(relative_path)
        return self._serve_static(parsed.path)

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        payload = self._read_json_body()
        if payload is None:
            return self._write_json(HTTPStatus.BAD_REQUEST, {"error": "请求体必须是 JSON"})

        if parsed.path == "/api/config":
            try:
                text = str(payload.get("text", ""))
                config = save_config_text(text)
                return self._write_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "path": get_config_path(),
                        "text": render_config_text(config),
                        "config": config,
                    },
                )
            except Exception as exc:
                return self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        if parsed.path == "/api/config/patch":
            try:
                text = str(payload.get("text", ""))
                patch = payload.get("patch", {}) or {}
                return self._write_json(HTTPStatus.OK, patch_config_text(text, patch))
            except Exception as exc:
                return self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        if parsed.path == "/api/jobs":
            try:
                mode = str(payload.get("mode", "")).strip().lower()
                if mode not in {"main", "paper", "live"}:
                    raise ValueError("mode 只支持 main、paper、live")
                config_text = str(payload.get("config_text", ""))
                save_config = bool(payload.get("save_config", False))
                force_update = payload.get("force_update")
                job = JOB_MANAGER.start_job(
                    mode=mode,
                    config_text=config_text,
                    save_config=save_config,
                    force_update=force_update,
                )
                return self._write_json(HTTPStatus.ACCEPTED, job)
            except Exception as exc:
                return self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        return self._write_json(HTTPStatus.NOT_FOUND, {"error": "未找到接口"})

    def _read_int_query(
        self,
        query_string: str,
        key: str,
        default: int,
        minimum: int = 1,
        maximum: int = 200,
    ) -> int:
        query = parse_qs(query_string)
        raw_value = query.get(key, [default])[0]
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(value, maximum))

    def _read_json_body(self) -> Optional[Dict[str, Any]]:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            return json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def _write_json(self, status: HTTPStatus, payload: Dict[str, Any]):
        body = json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, request_path: str):
        relative_path = "index.html" if request_path in {"/", ""} else request_path.lstrip("/")
        file_path = os.path.normpath(os.path.join(STATIC_ROOT, relative_path))
        if not file_path.startswith(STATIC_ROOT):
            return self._write_json(HTTPStatus.FORBIDDEN, {"error": "非法静态资源路径"})
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            file_path = os.path.join(STATIC_ROOT, "index.html")
        return self._serve_file(file_path)

    def _serve_project_file(self, relative_path: str):
        project_root = get_project_root()
        file_path = os.path.normpath(os.path.join(project_root, relative_path))
        if not file_path.startswith(project_root):
            return self._write_json(HTTPStatus.FORBIDDEN, {"error": "非法文件路径"})
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return self._write_json(HTTPStatus.NOT_FOUND, {"error": "文件不存在"})
        return self._serve_file(file_path)

    def _serve_file(self, file_path: str):
        content_type, _encoding = mimetypes.guess_type(file_path)
        content_type = content_type or "application/octet-stream"
        with open(file_path, "rb") as file:
            body = file.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") or content_type in {"application/json", "application/javascript"} else content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class TradingConsoleServer(ThreadingHTTPServer):
    daemon_threads = True


def create_server(host: str = "127.0.0.1", port: int = 8800) -> TradingConsoleServer:
    return TradingConsoleServer((host, int(port)), TradingConsoleHandler)


def main():
    parser = argparse.ArgumentParser(description="启动 AI 交易研究系统 Web 控制台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8800)
    args = parser.parse_args()

    server = create_server(host=args.host, port=args.port)
    logger.info("Web 控制台启动: http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭 Web 控制台")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
