"""quant grade tables: us_stock_grade + kr_stock_grade

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-20

These are the output tables that alphafolio_quant writes to via INSERT
ON CONFLICT (date, symbol). Schema reverse-engineered from quant's
INSERT statements in us_main.py:1736 (us_stock_grade, 81 cols) and
kr/db_async.py (kr_stock_grade — union of the comprehensive INSERT at
line 483 plus the legacy batch_save INSERT at line 786 which still
references support/resistance/supertrend/expected_range columns).

JSONB columns identified by ``::jsonb`` casts in the INSERT statements
(value_v2_detail / quality_v2_detail / momentum_v2_detail /
growth_v2_detail / buy_triggers / sell_triggers / hold_triggers /
insider_signal).

Idempotent: every CREATE / CREATE INDEX uses IF NOT EXISTS so it can
be applied on databases that previously had these tables created by
quant's ad-hoc scripts.
"""
from alembic import op


revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ==================================================================
    # us_stock_grade — 81 columns from us/us_main.py:1736
    # PK (date, symbol) per ON CONFLICT clause
    # ==================================================================
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_stock_grade (
            date                        DATE NOT NULL,
            symbol                      TEXT NOT NULL,
            stock_name                  TEXT,
            beta                        DOUBLE PRECISION,
            confidence_score            DOUBLE PRECISION,
            factor_combination_bonus    DOUBLE PRECISION,
            final_score                 DOUBLE PRECISION,
            final_grade                 TEXT,
            value_score                 DOUBLE PRECISION,
            quality_score               DOUBLE PRECISION,
            momentum_score              DOUBLE PRECISION,
            growth_score                DOUBLE PRECISION,
            value_v2_detail             JSONB,
            quality_v2_detail           JSONB,
            momentum_v2_detail          JSONB,
            growth_v2_detail            JSONB,
            var_95                      DOUBLE PRECISION,
            cvar_95                     DOUBLE PRECISION,
            hurst_exponent              DOUBLE PRECISION,
            var_95_5d                   DOUBLE PRECISION,
            var_95_20d                  DOUBLE PRECISION,
            var_95_60d                  DOUBLE PRECISION,
            var_95_90d                  DOUBLE PRECISION,
            var_99                      DOUBLE PRECISION,
            var_99_90d                  DOUBLE PRECISION,
            rs_value                    DOUBLE PRECISION,
            interaction_score           DOUBLE PRECISION,
            conviction_score            DOUBLE PRECISION,
            entry_timing_score          DOUBLE PRECISION,
            score_trend_2w              DOUBLE PRECISION,
            price_position_52w          DOUBLE PRECISION,
            atr_pct                     DOUBLE PRECISION,
            stop_loss_pct               DOUBLE PRECISION,
            take_profit_pct             DOUBLE PRECISION,
            risk_reward_ratio           DOUBLE PRECISION,
            position_size_pct           DOUBLE PRECISION,
            scenario_bullish_prob       DOUBLE PRECISION,
            scenario_sideways_prob      DOUBLE PRECISION,
            scenario_bearish_prob       DOUBLE PRECISION,
            scenario_bullish_return     DOUBLE PRECISION,
            scenario_sideways_return    DOUBLE PRECISION,
            scenario_bearish_return     DOUBLE PRECISION,
            scenario_sample_count       INTEGER,
            buy_triggers                JSONB,
            sell_triggers               JSONB,
            hold_triggers               JSONB,
            iv_percentile               DOUBLE PRECISION,
            insider_signal              JSONB,
            outlier_risk_score          DOUBLE PRECISION,
            risk_flag                   TEXT,
            weight_growth               DOUBLE PRECISION,
            weight_momentum             DOUBLE PRECISION,
            weight_quality              DOUBLE PRECISION,
            weight_value                DOUBLE PRECISION,
            volatility_annual           DOUBLE PRECISION,
            max_drawdown_1y             DOUBLE PRECISION,
            sharpe_ratio                DOUBLE PRECISION,
            sortino_ratio               DOUBLE PRECISION,
            calmar_ratio                DOUBLE PRECISION,
            value_momentum              DOUBLE PRECISION,
            quality_momentum            DOUBLE PRECISION,
            momentum_momentum           DOUBLE PRECISION,
            growth_momentum             DOUBLE PRECISION,
            sector_rotation_score       DOUBLE PRECISION,
            sector_momentum             DOUBLE PRECISION,
            sector_rank                 INTEGER,
            sector_percentile           DOUBLE PRECISION,
            risk_profile_text           TEXT,
            risk_recommendation         TEXT,
            time_series_text            TEXT,
            signal_overall              TEXT,
            market_state                TEXT,
            downside_vol                DOUBLE PRECISION,
            tail_beta                   DOUBLE PRECISION,
            corr_spy                    DOUBLE PRECISION,
            cvar_99                     DOUBLE PRECISION,
            var_95_ewma                 DOUBLE PRECISION,
            inv_vol_weight              DOUBLE PRECISION,
            vol_percentile              DOUBLE PRECISION,
            corr_sector_avg             DOUBLE PRECISION,
            drawdown_duration_avg       DOUBLE PRECISION,
            created_at                  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at                  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, symbol)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_stock_grade_symbol      ON us_stock_grade(symbol);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_stock_grade_final_grade ON us_stock_grade(final_grade);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_stock_grade_final_score ON us_stock_grade(final_score DESC);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_stock_grade_date_desc   ON us_stock_grade(date DESC);")

    # ==================================================================
    # kr_stock_grade — union of all INSERTs in kr/db_async.py + kr_main.py
    #
    # Comprehensive INSERT (db_async.py:483, 81 params) covers the modern
    # column set. The legacy batch_save INSERT (db_async.py:786) still
    # references expected_range_*/support_*/resistance_*/supertrend_value/
    # trend/signal — kept here so the legacy path also succeeds.
    #
    # PK (date, symbol) per ON CONFLICT (symbol, date) clauses.
    # ==================================================================
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_stock_grade (
            date                        DATE NOT NULL,
            symbol                      TEXT NOT NULL,
            stock_name                  TEXT,
            final_grade                 TEXT,
            final_score                 DOUBLE PRECISION,
            value_score                 DOUBLE PRECISION,
            quality_score               DOUBLE PRECISION,
            momentum_score              DOUBLE PRECISION,
            growth_score                DOUBLE PRECISION,
            confidence_score            DOUBLE PRECISION,
            var_95                      DOUBLE PRECISION,
            cvar_95                     DOUBLE PRECISION,
            cvar_99                     DOUBLE PRECISION,
            hurst_exponent              DOUBLE PRECISION,
            var_95_ewma                 DOUBLE PRECISION,
            var_95_5d                   DOUBLE PRECISION,
            var_95_20d                  DOUBLE PRECISION,
            var_95_60d                  DOUBLE PRECISION,
            var_99                      DOUBLE PRECISION,
            var_99_60d                  DOUBLE PRECISION,
            inv_vol_weight              DOUBLE PRECISION,
            downside_vol                DOUBLE PRECISION,
            vol_percentile              DOUBLE PRECISION,
            atr_20d                     DOUBLE PRECISION,
            atr_pct_20d                 DOUBLE PRECISION,
            volatility_annual           DOUBLE PRECISION,
            max_drawdown_1y             DOUBLE PRECISION,
            beta                        DOUBLE PRECISION,
            tail_beta                   DOUBLE PRECISION,
            corr_kospi                  DOUBLE PRECISION,
            corr_sector_avg             DOUBLE PRECISION,
            drawdown_duration_avg       DOUBLE PRECISION,
            inst_net_30d                INTEGER,
            foreign_net_30d             INTEGER,
            value_momentum              DOUBLE PRECISION,
            quality_momentum            DOUBLE PRECISION,
            momentum_momentum           DOUBLE PRECISION,
            growth_momentum             DOUBLE PRECISION,
            industry_rank               INTEGER,
            industry_percentile         DOUBLE PRECISION,
            rs_value                    DOUBLE PRECISION,
            rs_rank                     INTEGER,
            factor_combination_bonus    DOUBLE PRECISION,
            sector_rotation_score       DOUBLE PRECISION,
            sector_momentum             DOUBLE PRECISION,
            sector_rank                 INTEGER,
            sector_percentile           DOUBLE PRECISION,
            entry_timing_score          DOUBLE PRECISION,
            score_trend_2w              DOUBLE PRECISION,
            price_position_52w          DOUBLE PRECISION,
            atr_pct                     DOUBLE PRECISION,
            stop_loss_pct               DOUBLE PRECISION,
            take_profit_pct             DOUBLE PRECISION,
            risk_reward_ratio           DOUBLE PRECISION,
            position_size_pct           DOUBLE PRECISION,
            scenario_bullish_prob       DOUBLE PRECISION,
            scenario_sideways_prob      DOUBLE PRECISION,
            scenario_bearish_prob       DOUBLE PRECISION,
            scenario_bullish_return     DOUBLE PRECISION,
            scenario_sideways_return    DOUBLE PRECISION,
            scenario_bearish_return     DOUBLE PRECISION,
            scenario_sample_count       INTEGER,
            buy_triggers                JSONB,
            sell_triggers               JSONB,
            hold_triggers               JSONB,
            value_v2_detail             JSONB,
            quality_v2_detail           JSONB,
            momentum_v2_detail          JSONB,
            growth_v2_detail            JSONB,
            sharpe_ratio                DOUBLE PRECISION,
            sortino_ratio               DOUBLE PRECISION,
            calmar_ratio                DOUBLE PRECISION,
            conviction_score            DOUBLE PRECISION,
            outlier_risk_score          DOUBLE PRECISION,
            risk_flag                   TEXT,
            risk_profile_text           TEXT,
            risk_recommendation         TEXT,
            time_series_text            TEXT,
            signal_overall              TEXT,
            market_state                TEXT,
            -- legacy columns from batch_save INSERT (db_async.py:786)
            expected_range_3m_min       INTEGER,
            expected_range_3m_max       INTEGER,
            expected_range_1y_min       INTEGER,
            expected_range_1y_max       INTEGER,
            support_1                   DOUBLE PRECISION,
            support_2                   DOUBLE PRECISION,
            resistance_1                DOUBLE PRECISION,
            resistance_2                DOUBLE PRECISION,
            supertrend_value            DOUBLE PRECISION,
            trend                       TEXT,
            signal                      TEXT,
            created_at                  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at                  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, symbol)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_kr_stock_grade_symbol      ON kr_stock_grade(symbol);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_kr_stock_grade_final_grade ON kr_stock_grade(final_grade);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_kr_stock_grade_final_score ON kr_stock_grade(final_score DESC);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_kr_stock_grade_date_desc   ON kr_stock_grade(date DESC);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS kr_stock_grade;")
    op.execute("DROP TABLE IF EXISTS us_stock_grade;")
