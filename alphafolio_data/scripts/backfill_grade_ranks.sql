-- Backfill us_stock_grade rank/percentile columns from already-stored scores.
-- No grade re-generation needed — these columns are deterministic functions of
-- rs_value / final_score / value|quality|momentum|growth_score, all already
-- persisted. Mirrors the per-date post-processing in quant us_main.py but
-- partitioned BY date so it covers the whole history in one pass.
--
-- Scope:
--   rs_rank, *_rank/_percentile (value/quality/momentum/growth): backfill dates
--     where they are NULL (graded before migration 0019). Forward dates already
--     have correct values (those queries have no join).
--   industry_rank/industry_percentile: recompute for ALL dates — both the NULL
--     historical dates AND the forward dates whose values were INFLATED ~1300x by
--     the old buggy us_stock_basic join (now fixed via DISTINCT ON dedupe).
-- Run with the DAG idle (quant restarted) to avoid write contention.

\timing on

-- 1) RS Rank (no join; NTILE(100) per date over rs_value DESC)
WITH rs_ranked AS (
    SELECT symbol, date,
           NTILE(100) OVER (PARTITION BY date ORDER BY rs_value DESC) AS pct
    FROM us_stock_grade
    WHERE rs_value IS NOT NULL
      AND date IN (SELECT DISTINCT date FROM us_stock_grade WHERE rs_rank IS NULL)
)
UPDATE us_stock_grade g
SET rs_rank = CASE
    WHEN r.pct <= 10 THEN '매우강함 (Top 10%)'
    WHEN r.pct <= 20 THEN '강함 (Top 20%)'
    WHEN r.pct <= 40 THEN '보통 (Top 40%)'
    WHEN r.pct <= 60 THEN '약함 (Top 60%)'
    WHEN r.pct <= 80 THEN '매우약함 (Top 80%)'
    ELSE '최하위 (Bottom 20%)'
END
FROM rs_ranked r
WHERE g.symbol = r.symbol AND g.date = r.date AND g.rs_rank IS NULL;

-- 2) Industry Rank + percentile (ALL dates; deduped us_stock_basic join)
WITH basic AS (
    SELECT DISTINCT ON (symbol) symbol, industry
    FROM us_stock_basic
    WHERE industry IS NOT NULL AND industry != ''
    ORDER BY symbol, date DESC
),
ir AS (
    SELECT g.symbol, g.date,
        RANK() OVER (PARTITION BY g.date, b.industry ORDER BY g.final_score DESC) AS irank,
        ROUND((PERCENT_RANK() OVER (PARTITION BY g.date, b.industry ORDER BY g.final_score DESC) * 100)::NUMERIC, 1) AS ipct
    FROM us_stock_grade g
    JOIN basic b ON g.symbol = b.symbol
)
UPDATE us_stock_grade g
SET industry_rank = ir.irank, industry_percentile = ir.ipct
FROM ir
WHERE g.symbol = ir.symbol AND g.date = ir.date
  AND (g.industry_rank IS DISTINCT FROM ir.irank
       OR g.industry_percentile IS DISTINCT FROM ir.ipct);

-- 3) Factor rankings (no join; per date). Label '공동 N위' on ties, else 'N위'.
-- value
WITH r AS (
    SELECT symbol, date,
        RANK() OVER (PARTITION BY date ORDER BY value_score DESC) AS rk,
        COUNT(*) OVER (PARTITION BY date, value_score) AS tie,
        ROUND((PERCENT_RANK() OVER (PARTITION BY date ORDER BY value_score DESC) * 100)::NUMERIC, 1) AS pct
    FROM us_stock_grade
    WHERE final_grade != '평가 불가' AND value_score IS NOT NULL
      AND date IN (SELECT DISTINCT date FROM us_stock_grade WHERE value_rank IS NULL)
)
UPDATE us_stock_grade g
SET value_rank = CASE WHEN r.tie > 1 THEN '공동 ' || r.rk || '위' ELSE r.rk || '위' END,
    value_percentile = r.pct
FROM r WHERE g.symbol = r.symbol AND g.date = r.date AND g.value_rank IS NULL;

-- quality
WITH r AS (
    SELECT symbol, date,
        RANK() OVER (PARTITION BY date ORDER BY quality_score DESC) AS rk,
        COUNT(*) OVER (PARTITION BY date, quality_score) AS tie,
        ROUND((PERCENT_RANK() OVER (PARTITION BY date ORDER BY quality_score DESC) * 100)::NUMERIC, 1) AS pct
    FROM us_stock_grade
    WHERE final_grade != '평가 불가' AND quality_score IS NOT NULL
      AND date IN (SELECT DISTINCT date FROM us_stock_grade WHERE quality_rank IS NULL)
)
UPDATE us_stock_grade g
SET quality_rank = CASE WHEN r.tie > 1 THEN '공동 ' || r.rk || '위' ELSE r.rk || '위' END,
    quality_percentile = r.pct
FROM r WHERE g.symbol = r.symbol AND g.date = r.date AND g.quality_rank IS NULL;

-- momentum
WITH r AS (
    SELECT symbol, date,
        RANK() OVER (PARTITION BY date ORDER BY momentum_score DESC) AS rk,
        COUNT(*) OVER (PARTITION BY date, momentum_score) AS tie,
        ROUND((PERCENT_RANK() OVER (PARTITION BY date ORDER BY momentum_score DESC) * 100)::NUMERIC, 1) AS pct
    FROM us_stock_grade
    WHERE final_grade != '평가 불가' AND momentum_score IS NOT NULL
      AND date IN (SELECT DISTINCT date FROM us_stock_grade WHERE momentum_rank IS NULL)
)
UPDATE us_stock_grade g
SET momentum_rank = CASE WHEN r.tie > 1 THEN '공동 ' || r.rk || '위' ELSE r.rk || '위' END,
    momentum_percentile = r.pct
FROM r WHERE g.symbol = r.symbol AND g.date = r.date AND g.momentum_rank IS NULL;

-- growth
WITH r AS (
    SELECT symbol, date,
        RANK() OVER (PARTITION BY date ORDER BY growth_score DESC) AS rk,
        COUNT(*) OVER (PARTITION BY date, growth_score) AS tie,
        ROUND((PERCENT_RANK() OVER (PARTITION BY date ORDER BY growth_score DESC) * 100)::NUMERIC, 1) AS pct
    FROM us_stock_grade
    WHERE final_grade != '평가 불가' AND growth_score IS NOT NULL
      AND date IN (SELECT DISTINCT date FROM us_stock_grade WHERE growth_rank IS NULL)
)
UPDATE us_stock_grade g
SET growth_rank = CASE WHEN r.tie > 1 THEN '공동 ' || r.rk || '위' ELSE r.rk || '위' END,
    growth_percentile = r.pct
FROM r WHERE g.symbol = r.symbol AND g.date = r.date AND g.growth_rank IS NULL;
