"""Orchestrator — wires grade generation, simulation, metrics, storage."""
import logging
from datetime import date, timedelta
from uuid import uuid4

from backtest import storage, metrics
from backtest.simulator import PortfolioSimulator

logger = logging.getLogger(__name__)


async def run_backtest(params: dict) -> dict:
    """Run portfolio simulation. Returns full result dict including run_id + metrics.

    Assumes {us,kr}_stock_grade has data for the date range. To populate it,
    call generate_grades_for_range() first.
    """
    run_id = f"{uuid4().hex[:8]}_{params['country'].lower()}"
    await storage.create_run(run_id, params)

    try:
        pool = await storage.get_pool()
        sim = PortfolioSimulator(
            pool=pool,
            country=params["country"],
            start_date=params["start_date"],
            end_date=params["end_date"],
            initial_cash=params["initial_cash"],
            top_n=params["top_n"],
            rebal_freq_days=params["rebal_freq_days"],
            grades_filter=params.get("grades_filter") or ["STRONG_BUY", "BUY", "강력 매수", "매수"],
            commission_rate=params.get("commission_rate", 0.0025),
            slippage_rate=params.get("slippage_rate", 0.001),
        )

        nav_history, trades = await sim.run()
        await storage.save_nav(run_id, nav_history)
        await storage.save_trades(run_id, trades)

        m = metrics.calculate_metrics(nav_history, float(params["initial_cash"]))
        await storage.update_run_complete(run_id, m)

        logger.info(f"[backtest {run_id}] completed: {m}")
        return {
            "run_id": run_id,
            "status": "completed",
            "metrics": m,
            "num_trades": len(trades),
            "num_nav_points": len(nav_history),
        }

    except Exception as e:
        logger.exception(f"[backtest {run_id}] failed")
        await storage.update_run_failed(run_id, str(e))
        return {"run_id": run_id, "status": "failed", "error": str(e)}


async def generate_grades_for_range(
    country: str,
    start_date: date,
    end_date: date,
    skip_existing: bool = True,
    use_prefilter: bool = False,
    prefilter_top_n: int = 500,
    with_event_modifier: bool = True,
) -> dict:
    """Loop call run_option1(target_date) to populate stock_grade for past dates.

    If use_prefilter=True and daily_top_symbols table has data for the date,
    only those symbols are analyzed (about 10× speedup). Falls back to full
    universe if daily_top_symbols is empty for the date.

    Slow: 5~15min per trading day × N days for full universe.
          ~6min/day for top-500 pre-filtered.
    """
    pool = await storage.get_pool()

    country_u = country.upper()
    if country_u == "US":
        from us.us_main import run_option1
        grade_table = "us_stock_grade"
    elif country_u == "KR":
        from kr.kr_main import run_option1
        grade_table = "kr_stock_grade"
    else:
        raise ValueError(f"country must be KR or US, got {country}")

    existing_dates = set()
    if skip_existing:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT DISTINCT date FROM {grade_table} "
                f"WHERE date BETWEEN $1 AND $2",
                start_date, end_date,
            )
            existing_dates = {r["date"] for r in rows}
        logger.info(f"[grade-gen] {len(existing_dates)} dates already populated, will skip")

    processed, skipped, failed = [], [], []
    d = start_date
    while d <= end_date:
        if d.weekday() >= 5:
            d += timedelta(days=1)
            continue
        if d in existing_dates:
            skipped.append(str(d))
            d += timedelta(days=1)
            continue
        try:
            # ----- EM8 pre-filter symbols -------------------------
            symbols_filter = None
            if use_prefilter and country_u == "US":
                async with pool.acquire() as conn:
                    pf_rows = await conn.fetch(
                        """SELECT symbol FROM daily_top_symbols
                           WHERE date = $1 ORDER BY rank LIMIT $2""",
                        d, prefilter_top_n,
                    )
                if pf_rows:
                    symbols_filter = [r["symbol"] for r in pf_rows]
                    logger.info(
                        f"[grade-gen] {d} using EM8 pre-filter "
                        f"({len(symbols_filter)} symbols)")
                else:
                    logger.warning(
                        f"[grade-gen] {d} use_prefilter=True but "
                        f"daily_top_symbols empty — using full universe")
            # -------------------------------------------------------

            logger.info(f"[grade-gen] {country_u} analyzing {d}")
            # Pass symbols_filter to run_option1 if it accepts the kwarg
            try:
                if symbols_filter is not None:
                    await run_option1(target_date=d, symbols=symbols_filter,
                                      with_event_modifier=with_event_modifier)
                else:
                    await run_option1(target_date=d,
                                      with_event_modifier=with_event_modifier)
            except TypeError:
                # Older run_option1 doesn't accept 'symbols' kwarg yet —
                # fall back to full universe (no speedup, but doesn't break)
                logger.warning(
                    f"[grade-gen] run_option1 doesn't accept symbols kwarg; "
                    f"running full universe for {d}")
                await run_option1(target_date=d)
            processed.append(str(d))
        except Exception as e:
            logger.exception(f"[grade-gen] {d} failed: {e}")
            failed.append({"date": str(d), "error": str(e)})
        d += timedelta(days=1)

    return {
        "country": country_u,
        "processed_count": len(processed),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "processed": processed[-10:],  # tail only to keep response small
        "skipped": skipped[-10:],
        "failed": failed,
    }
