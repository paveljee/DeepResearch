import json
import os
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base_stats() -> dict:
    return {
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_time_sec": 0.0,
    }


class MetricsTracker:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self._thread_local = threading.local()
        self._initialized = False
        self._metrics_dir = ""
        self._calls_path = ""
        self._summary_path = ""
        self._global_call_index = 0
        self._run_call_index = defaultdict(int)
        self._summary = {}

    @classmethod
    def instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure_initialized(self):
        if self._initialized:
            return

        metrics_dir = os.getenv("DEEPRESEARCH_METRICS_DIR", "./outputs/metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        self._metrics_dir = metrics_dir
        self._calls_path = os.path.join(metrics_dir, "llm_call_metrics.jsonl")
        self._summary_path = os.path.join(metrics_dir, "llm_metrics_summary.json")

        self._summary = {
            "generated_at_utc": _utc_now(),
            "metrics_dir": self._metrics_dir,
            "totals": _base_stats(),
            "by_model": {},
            "by_run": {},
            "tool_invocations_total": {},
            "notes": {
                "run_id": "One run_id corresponds to one question-rollout execution.",
                "model_key": "model_key is '<llm_role>::<model_name>'.",
            },
        }
        self._initialized = True
        self._flush_summary()

    def _flush_summary(self):
        self._summary["generated_at_utc"] = _utc_now()
        with open(self._summary_path, "w", encoding="utf-8") as f:
            json.dump(self._summary, f, ensure_ascii=False, indent=2)

    def set_llm_context(
        self,
        *,
        run_id: str,
        rollout_idx: Optional[int],
        question: str,
        main_llm_call_index: int,
    ):
        self._thread_local.llm_context = {
            "run_id": run_id,
            "rollout_idx": rollout_idx,
            "question": question,
            "main_llm_call_index": main_llm_call_index,
        }

    def get_llm_context(self) -> Optional[dict]:
        return getattr(self._thread_local, "llm_context", None)

    def clear_llm_context(self):
        if hasattr(self._thread_local, "llm_context"):
            del self._thread_local.llm_context

    def record_llm_call(
        self,
        *,
        run_id: str,
        rollout_idx: Optional[int],
        question: str,
        llm_role: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        latency_sec: float,
        tools_invoked: Optional[Dict[str, int]] = None,
        parent_main_llm_call_index: Optional[int] = None,
        extra: Optional[dict] = None,
    ) -> dict:
        with self._lock:
            self._ensure_initialized()
            self._global_call_index += 1
            self._run_call_index[run_id] += 1
            run_llm_call_index = self._run_call_index[run_id]

            tools_invoked = tools_invoked or {}
            total_tokens = int(input_tokens) + int(output_tokens)
            latency_sec = float(latency_sec)
            model_key = f"{llm_role}::{model_name}"

            totals = self._summary["totals"]
            totals["llm_calls"] += 1
            totals["input_tokens"] += int(input_tokens)
            totals["output_tokens"] += int(output_tokens)
            totals["total_tokens"] += int(total_tokens)
            totals["total_time_sec"] += latency_sec

            by_model = self._summary["by_model"].setdefault(model_key, _base_stats())
            by_model["llm_calls"] += 1
            by_model["input_tokens"] += int(input_tokens)
            by_model["output_tokens"] += int(output_tokens)
            by_model["total_tokens"] += int(total_tokens)
            by_model["total_time_sec"] += latency_sec

            by_run = self._summary["by_run"].setdefault(
                run_id,
                {
                    "question": question,
                    "rollout_idx": rollout_idx,
                    "stats": _base_stats(),
                    "by_model": {},
                    "tool_invocations_total": {},
                },
            )
            by_run_stats = by_run["stats"]
            by_run_stats["llm_calls"] += 1
            by_run_stats["input_tokens"] += int(input_tokens)
            by_run_stats["output_tokens"] += int(output_tokens)
            by_run_stats["total_tokens"] += int(total_tokens)
            by_run_stats["total_time_sec"] += latency_sec

            by_run_model = by_run["by_model"].setdefault(model_key, _base_stats())
            by_run_model["llm_calls"] += 1
            by_run_model["input_tokens"] += int(input_tokens)
            by_run_model["output_tokens"] += int(output_tokens)
            by_run_model["total_tokens"] += int(total_tokens)
            by_run_model["total_time_sec"] += latency_sec

            global_tools = Counter(self._summary["tool_invocations_total"])
            run_tools = Counter(by_run["tool_invocations_total"])
            for tool_name, count in tools_invoked.items():
                global_tools[tool_name] += int(count)
                run_tools[tool_name] += int(count)
            self._summary["tool_invocations_total"] = dict(global_tools)
            by_run["tool_invocations_total"] = dict(run_tools)

            event = {
                "timestamp_utc": _utc_now(),
                "global_llm_call_index": self._global_call_index,
                "run_llm_call_index": run_llm_call_index,
                "run_id": run_id,
                "rollout_idx": rollout_idx,
                "question": question,
                "llm_role": llm_role,
                "model_name": model_name,
                "model_key": model_key,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "total_tokens": int(total_tokens),
                "latency_sec": latency_sec,
                "tools_invoked_this_call": tools_invoked,
                "parent_main_llm_call_index": parent_main_llm_call_index,
                "running_totals": dict(self._summary["totals"]),
                "running_model_totals": dict(self._summary["by_model"][model_key]),
                "running_tool_totals": dict(self._summary["tool_invocations_total"]),
                "running_run_totals": dict(by_run_stats),
            }
            if extra:
                event["extra"] = extra

            with open(self._calls_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

            self._flush_summary()
            return event


def set_llm_context(*, run_id: str, rollout_idx: Optional[int], question: str, main_llm_call_index: int):
    MetricsTracker.instance().set_llm_context(
        run_id=run_id,
        rollout_idx=rollout_idx,
        question=question,
        main_llm_call_index=main_llm_call_index,
    )


def get_llm_context() -> Optional[dict]:
    return MetricsTracker.instance().get_llm_context()


def clear_llm_context():
    MetricsTracker.instance().clear_llm_context()


def record_llm_call(
    *,
    run_id: str,
    rollout_idx: Optional[int],
    question: str,
    llm_role: str,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    latency_sec: float,
    tools_invoked: Optional[Dict[str, int]] = None,
    parent_main_llm_call_index: Optional[int] = None,
    extra: Optional[dict] = None,
) -> dict:
    return MetricsTracker.instance().record_llm_call(
        run_id=run_id,
        rollout_idx=rollout_idx,
        question=question,
        llm_role=llm_role,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_sec=latency_sec,
        tools_invoked=tools_invoked,
        parent_main_llm_call_index=parent_main_llm_call_index,
        extra=extra,
    )

