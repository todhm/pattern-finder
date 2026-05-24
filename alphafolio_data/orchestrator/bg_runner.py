"""백그라운드 long-running task 진단 wrapper.

docker compose exec -d 의 stream 미정리로 SIGPIPE / BrokenPipeError 가 발생해
child 프로세스가 silent death 하는 문제를 진단하기 위함.

사용:
    docker compose exec -d alphafolio_data \\
        python -m orchestrator.bg_runner us_weekly /tmp/weekly_run.log

향후 silent death 시 /tmp/bg_runner_diag.log 에 정확한 signal / exception 캡쳐.
"""
from __future__ import annotations

import asyncio
import faulthandler
import importlib
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Callable


DIAG_LOG = Path("/tmp/bg_runner_diag.log")


def _diag(msg: str) -> None:
    """진단 로그 — flush + fsync 로 즉시 디스크 기록."""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [pid={os.getpid()}] {msg}\n"
    with DIAG_LOG.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _install_signal_handlers() -> None:
    """SIGTERM/HUP/INT/PIPE 잡아서 어느 signal 로 죽었는지 기록."""

    def handler(signum: int, frame) -> None:
        sig_name = signal.Signals(signum).name
        _diag(f"SIGNAL received: {sig_name} ({signum})")
        if frame is not None:
            stack = "".join(traceback.format_stack(frame))
            _diag(f"Stack at signal:\n{stack}")
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT, signal.SIGPIPE):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError) as e:
            _diag(f"signal.signal({sig}) failed: {e}")


def _install_faulthandler() -> None:
    """C-level crash (segfault 등) 도 stack trace 로 기록."""
    fh_log = Path("/tmp/bg_runner_faulthandler.log").open("a", encoding="utf-8")
    faulthandler.enable(file=fh_log, all_threads=True)


def _install_excepthook() -> None:
    """잡히지 않은 예외 (asyncio task crash 포함) 도 진단 로그에 기록."""

    def hook(exc_type, exc_value, exc_tb) -> None:
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        _diag(f"UNCAUGHT EXCEPTION:\n{tb_str}")
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = hook


def _redirect_streams_if_needed() -> None:
    """detach mode 에서 stdout/stderr 가 broken pipe 가 되어도 죽지 않게.
    이미 redirect 되어 있으면 (nohup ... > file) skip.
    """
    try:
        sys.stdout.write("")
        sys.stdout.flush()
    except (BrokenPipeError, OSError) as e:
        _diag(f"stdout broken at startup ({e}); redirecting to DIAG_LOG")
        f = DIAG_LOG.open("a", encoding="utf-8")
        sys.stdout = f
        sys.stderr = f


TASKS: dict[str, Callable[[], object]] = {}


def _register_us_weekly() -> None:
    async def _run():
        sys.path.insert(0, "/app")
        from us.alphavantage import WeeklyCollector

        api_key = os.environ["ALPHAVANTAGE_API_KEY"]
        db_url = os.environ["DATABASE_URL"]
        col = WeeklyCollector(api_key, db_url, max_concurrent=20)
        await col.run_collection()

    TASKS["us_weekly"] = _run


_register_us_weekly()


def _register_dummy_stream_test() -> None:
    """진단 전용: SIGPIPE/BrokenPipe 재현 테스트."""

    async def _run():
        for i in range(600):
            print(f"dummy tick {i}", flush=True)
            await asyncio.sleep(1)

    TASKS["dummy_stream_test"] = _run


_register_dummy_stream_test()


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m orchestrator.bg_runner <task_name>", file=sys.stderr)
        return 2

    task_name = sys.argv[1]
    if task_name not in TASKS:
        _diag(f"unknown task {task_name!r}; available: {list(TASKS)}")
        return 2

    _diag(f"=== START task={task_name} argv={sys.argv} ===")
    _redirect_streams_if_needed()
    _install_signal_handlers()
    _install_faulthandler()
    _install_excepthook()

    try:
        runner = TASKS[task_name]
        result = runner()
        if asyncio.iscoroutine(result):
            asyncio.run(result)
        _diag(f"=== EXIT OK task={task_name} ===")
        return 0
    except SystemExit as e:
        _diag(f"SystemExit code={e.code}")
        raise
    except BaseException as e:
        _diag(f"FATAL {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
