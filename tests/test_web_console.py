import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from unittest import mock

from src.app_service import get_project_root, get_results_root
from src.web_app import JOB_MANAGER, create_server


class TestWebConsole(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(host="127.0.0.1", port=0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.temp_dirs = []
        with JOB_MANAGER._lock:
            JOB_MANAGER._jobs.clear()
            JOB_MANAGER._job_logs.clear()

    def tearDown(self):
        for directory in self.temp_dirs:
            shutil.rmtree(directory, ignore_errors=True)

    def _json_request(self, path, method="GET", payload=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read().decode("utf-8"))
            finally:
                exc.close()

    def _text_request(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=10) as response:
            return response.status, response.read().decode("utf-8")

    def _create_result_dir(self):
        os.makedirs(get_results_root(), exist_ok=True)
        directory = tempfile.mkdtemp(prefix="web_console_", dir=get_results_root())
        self.temp_dirs.append(directory)
        return directory

    def _write_text(self, path, content):
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)

    def test_index_page_is_served(self):
        status, body = self._text_request("/")
        self.assertEqual(status, 200)
        self.assertIn("AI 交易工作台", body)
        self.assertIn("我要交易什么", body)

    def test_web_app_supports_direct_script_startup(self):
        completed = subprocess.run(
            [sys.executable, os.path.join("src", "web_app.py"), "--help"],
            cwd=get_project_root(),
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr.decode(errors="replace"))
        self.assertIn(b"usage:", completed.stdout.lower())

    def test_api_config_and_patch(self):
        status, payload = self._json_request("/api/config")
        self.assertEqual(status, 200)
        self.assertIn("text", payload)
        self.assertIn("config", payload)

        status, patched = self._json_request(
            "/api/config/patch",
            method="POST",
            payload={
                "text": payload["text"],
                "patch": {
                    "data": {"symbol": "demo.us", "symbols": ["demo.us", "600000.sh"]},
                    "market": {"profile": "us_equity"},
                },
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(patched["config"]["data"]["symbol"], "demo.us")
        self.assertEqual(patched["config"]["market"]["profile"], "us_equity")
        self.assertIn("demo.us", patched["text"])

    def test_api_jobs_rejects_invalid_mode(self):
        status, payload = self._json_request(
            "/api/jobs",
            method="POST",
            payload={
                "mode": "invalid",
                "config_text": "data:\n  symbol: aapl.us\n",
            },
        )
        self.assertEqual(status, 400)
        self.assertIn("mode", payload["error"])

    def test_api_job_merges_runtime_secrets_from_environment(self):
        captured = {}

        def fake_run_pipeline(config=None):
            captured["config"] = config or {}
            return {
                "run_type": "research",
                "summary": {"llm_enabled": bool((config or {}).get("llm", {}).get("enabled"))},
                "artifacts": [],
            }

        config_text = (
            "data:\n"
            "  symbol: demo.us\n"
            "llm:\n"
            "  enabled: true\n"
            "  provider: deepseek\n"
            "  model: deepseek-chat\n"
        )
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "unit-test-key"}), mock.patch(
            "src.web_app.run_main_pipeline", side_effect=fake_run_pipeline
        ):
            status, payload = self._json_request(
                "/api/jobs",
                method="POST",
                payload={"mode": "main", "config_text": config_text},
            )
            self.assertEqual(status, 202)
            job_id = payload["id"]

            job_payload = None
            for _ in range(30):
                time.sleep(0.1)
                status, job_payload = self._json_request(f"/api/jobs/{job_id}")
                self.assertEqual(status, 200)
                if job_payload["status"] == "completed":
                    break

        self.assertIsNotNone(job_payload)
        self.assertEqual(job_payload["status"], "completed")
        self.assertEqual(captured["config"]["api_keys"]["deepseek"]["api_key"], "unit-test-key")
        self.assertTrue(captured["config"]["llm"]["enabled"])

    def test_api_job_logs(self):
        def fake_run_pipeline(config=None):
            logging.getLogger("test.web_console").info("模拟 pipeline 已执行")
            return {
                "run_type": "paper",
                "summary": {"total_return": 0.12, "max_drawdown": 0.03},
                "artifacts": [],
            }

        with mock.patch("src.web_app.run_paper_pipeline", side_effect=fake_run_pipeline):
            status, payload = self._json_request(
                "/api/jobs",
                method="POST",
                payload={
                    "mode": "paper",
                    "config_text": "data:\n  symbol: demo.us\n",
                },
            )
            self.assertEqual(status, 202)
            job_id = payload["id"]

            job_payload = None
            for _ in range(30):
                time.sleep(0.1)
                status, job_payload = self._json_request(f"/api/jobs/{job_id}")
                self.assertEqual(status, 200)
                if job_payload["status"] == "completed":
                    break
            self.assertIsNotNone(job_payload)
            self.assertEqual(job_payload["status"], "completed")

            status, logs_payload = self._json_request(f"/api/jobs/{job_id}/logs?limit=50")
            self.assertEqual(status, 200)
            self.assertEqual(logs_payload["job_id"], job_id)
            joined_logs = "\n".join(logs_payload["logs"])
            self.assertIn("开始执行 Paper Trading", joined_logs)
            self.assertIn("任务执行完成", joined_logs)

    def test_result_chart_and_file_preview(self):
        result_dir = self._create_result_dir()
        self._write_text(
            os.path.join(result_dir, "paper_summary.json"),
            json.dumps({"final_equity": 105000, "total_return": 0.05}, ensure_ascii=False, indent=2),
        )
        self._write_text(
            os.path.join(result_dir, "equity_curve.csv"),
            "timestamp,equity\n2026-01-01,100000\n2026-01-02,101500\n2026-01-03,103200\n",
        )
        self._write_text(
            os.path.join(result_dir, "paper_account_history.csv"),
            "timestamp,equity,cash,drawdown,realized_pnl,unrealized_pnl\n"
            "2026-01-01,100000,100000,0,0,0\n"
            "2026-01-02,101000,58000,0.01,500,500\n"
            "2026-01-03,105000,56000,0.02,2500,1500\n",
        )
        self._write_text(
            os.path.join(result_dir, "notes.log"),
            "first line\nsecond line\nthird line\n",
        )
        self._write_text(
            os.path.join(result_dir, "ai_review.json"),
            json.dumps({"headline": "本地复盘", "llm_used": False}, ensure_ascii=False, indent=2),
        )

        relative_dir = os.path.relpath(result_dir, get_project_root()).replace("\\", "/")
        status, detail_payload = self._json_request(f"/api/result?path={urllib.parse.quote(relative_dir)}")
        self.assertEqual(status, 200)
        self.assertEqual(detail_payload["run_type"], "paper")
        self.assertEqual(detail_payload["ai_review"]["headline"], "本地复盘")

        status, chart_payload = self._json_request(f"/api/result/chart?path={urllib.parse.quote(relative_dir)}")
        self.assertEqual(status, 200)
        chart_names = {chart["file_name"] for chart in chart_payload["charts"]}
        self.assertIn("equity_curve.csv", chart_names)
        self.assertIn("paper_account_history.csv", chart_names)

        relative_csv = f"{relative_dir}/paper_account_history.csv"
        status, csv_preview = self._json_request(
            f"/api/file-preview?path={urllib.parse.quote(relative_csv)}&limit=2"
        )
        self.assertEqual(status, 200)
        self.assertEqual(csv_preview["kind"], "table")
        self.assertEqual(csv_preview["row_count"], 3)
        self.assertEqual(len(csv_preview["rows"]), 2)
        self.assertIn("equity", csv_preview["columns"])

        relative_json = f"{relative_dir}/paper_summary.json"
        status, json_preview = self._json_request(
            f"/api/file-preview?path={urllib.parse.quote(relative_json)}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(json_preview["kind"], "json")
        self.assertEqual(json_preview["content"]["final_equity"], 105000)

        relative_log = f"{relative_dir}/notes.log"
        status, log_preview = self._json_request(
            f"/api/file-preview?path={urllib.parse.quote(relative_log)}&limit=2"
        )
        self.assertEqual(status, 200)
        self.assertEqual(log_preview["kind"], "text")
        self.assertIn("first line", log_preview["text"])
        self.assertTrue(log_preview["truncated"])


if __name__ == "__main__":
    unittest.main()
