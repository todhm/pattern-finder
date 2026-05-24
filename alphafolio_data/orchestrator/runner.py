"""DAG executor: dependency-aware async scheduler.

각 task 는 자기 자신의 depends_on 의 모든 task 의 완료 (success) 이벤트가
set 될 때까지 asyncio.Event 로 대기한다. 모든 task 가 동시에 spawn 되지만
의존성이 만족될 때까지 await 으로 멈춤 → 순서 의존 없는 진짜 DAG 실행.

기존의 sequential for-loop 한계 (task 정렬 순서대로 처리하다가 깊은 곳의
deps 가 같은 list 안에서 아직 success 가 아니면 'skipped' 처리되던 버그)
를 근본적으로 제거.

진행 시점에 새 task 가 추가되더라도 다음 resume 에서 정상 동작.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Optional
from uuid import uuid4

from orchestrator import storage
from orchestrator.tasks import Ctx, TASK_REGISTRY, build_dag

logger = logging.getLogger(__name__)

# Track running tasks so cancel can interrupt
_RUNNING: dict[str, asyncio.Task] = {}


async def create_pipeline_run(country: str, start_date: date, end_date: date,
                              params: Optional[dict] = None) -> str:
    """Create a new run and persist initial task list. Returns run_id."""
    run_id = f"{uuid4().hex[:8]}_{country.lower()}_{start_date.isoformat()}"
    params = params or {}
    dag = build_dag(country)
    await storage.create_run(run_id, country, start_date, end_date, params)
    await storage.init_tasks(run_id, dag)
    await storage.log(run_id, "info",
                      f"Created run {run_id}: {country} {start_date}~{end_date} "
                      f"with {len(dag)} tasks")
    return run_id


async def execute(run_id: str):
    """DAG 의 모든 task 를 동시에 spawn — 각자 deps Event 만 충족되면 즉시 실행.

    병렬 실행 가능한 branch 는 진짜로 병렬 진행 (financials + macros + us_calculator
    + em8_pre_filter 가 동시에). critical path = 가장 긴 chain 의 합.
    """
    run = await storage.get_run(run_id)
    if not run:
        return
    if run["status"] == "cancelled":
        return

    await storage.update_run(run_id, "running")
    await storage.log(run_id, "info", f"Run start ({run['country']})")

    tasks_db = await storage.list_tasks(run_id)
    task_state = {t["task_id"]: dict(t) for t in tasks_db}

    # 각 task 별 완료 이벤트 — success 시 set, 실패/skip 시 set 하되 _failed 플래그
    done_events: dict[str, asyncio.Event] = {tid: asyncio.Event() for tid in task_state}
    task_succeeded: dict[str, bool] = {}

    # 이미 success 인 것은 즉시 event set + succeeded=True
    for tid, t in task_state.items():
        if t["status"] == "success":
            done_events[tid].set()
            task_succeeded[tid] = True

    run_failed = False

    async def run_one(tid: str):
        nonlocal run_failed
        t = task_state[tid]

        # 이미 success 면 skip
        if t["status"] == "success":
            return

        # dependencies 의 done_event 모두 대기
        deps = list(t.get("depends_on") or [])
        for dep in deps:
            if dep not in done_events:
                # DAG 에 없는 dep — 코드 버그
                await storage.update_task(run_id, tid, "failed",
                                          error=f"unknown dep: {dep}",
                                          mark_start=True, mark_end=True)
                done_events[tid].set()
                task_succeeded[tid] = False
                run_failed = True
                return
            await done_events[dep].wait()

        # 모든 dep 가 success 였는지 확인 (실패한 dep 가 있으면 skip)
        for dep in deps:
            if not task_succeeded.get(dep, False):
                await storage.update_task(run_id, tid, "skipped",
                                          error=f"upstream {dep} not satisfied")
                await storage.log(run_id, "warn",
                                  f"{tid}: upstream {dep} not satisfied", tid)
                done_events[tid].set()
                task_succeeded[tid] = False
                return

        # run cancel 체크
        cur = await storage.get_run(run_id)
        if cur and cur["status"] == "cancelled":
            done_events[tid].set()
            task_succeeded[tid] = False
            return

        fn = TASK_REGISTRY.get(tid)
        if not fn:
            await storage.update_task(run_id, tid, "failed",
                                      error=f"no implementation: {tid}",
                                      mark_start=True, mark_end=True)
            await storage.log(run_id, "error", f"unknown task {tid}", tid)
            done_events[tid].set()
            task_succeeded[tid] = False
            run_failed = True
            return

        ctx = Ctx(run_id=run_id, country=run["country"],
                  start_date=run["start_date"], end_date=run["end_date"],
                  params=run["params"] if isinstance(run["params"], dict)
                        else _parse_json(run["params"]),
                  storage=storage, _task_id=tid)

        await storage.update_task(run_id, tid, "running", mark_start=True)
        await storage.log(run_id, "run", f"▶ {tid}", tid)

        try:
            inner = asyncio.create_task(fn(ctx))
            _RUNNING[f"{run_id}:{tid}"] = inner
            output = await inner
            await storage.update_task(run_id, tid, "success",
                                      output=output, mark_end=True)
            await storage.log(run_id, "ok", f"✓ {tid}", tid)
            task_succeeded[tid] = True
        except asyncio.CancelledError:
            await storage.update_task(run_id, tid, "failed",
                                      error="cancelled", mark_end=True)
            await storage.log(run_id, "warn", f"⊘ {tid} cancelled", tid)
            task_succeeded[tid] = False
            raise   # downstream 들이 wait 풀고 skip 처리
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            await storage.update_task(run_id, tid, "failed",
                                      error=err[:1000], mark_end=True)
            await storage.log(run_id, "error", f"✗ {tid}: {err}", tid)
            task_succeeded[tid] = False
            run_failed = True
        finally:
            _RUNNING.pop(f"{run_id}:{tid}", None)
            done_events[tid].set()   # 성공/실패 모두 downstream 깨움

    try:
        # 모든 task spawn 후 gather — 의존성은 각 task 가 자체 await
        coros = [run_one(tid) for tid in task_state.keys()]
        await asyncio.gather(*coros, return_exceptions=True)

        # 결과 판정
        cur = await storage.get_run(run_id)
        if cur and cur["status"] == "cancelled":
            return
        if run_failed:
            await storage.update_run(run_id, "failed",
                                     error="one or more tasks failed",
                                     completed=True)
            await storage.log(run_id, "error", "Run finished with failures")
        else:
            await storage.update_run(run_id, "completed", completed=True)
            await storage.log(run_id, "ok", "Run completed successfully")
    except Exception as e:
        await storage.update_run(run_id, "failed", error=str(e)[:500], completed=True)
        await storage.log(run_id, "error", f"runner crash: {e}")


async def resume(run_id: str):
    """Mark failed/skipped tasks back to pending, then execute."""
    tasks = await storage.list_tasks(run_id)
    for t in tasks:
        if t["status"] in ("failed", "skipped"):
            await storage.reset_task(run_id, t["task_id"])
            await storage.log(run_id, "info", f"Reset {t['task_id']} → pending")
    await storage.log(run_id, "info", "Resuming from first non-success task")
    asyncio.create_task(execute(run_id))


async def cancel(run_id: str):
    await storage.update_run(run_id, "cancelled")
    task = _RUNNING.get(run_id)
    if task:
        task.cancel()
    await storage.log(run_id, "warn", "Cancellation requested")


def _parse_json(v):
    if isinstance(v, dict):
        return v
    import json
    try:
        return json.loads(v)
    except Exception:
        return {}
