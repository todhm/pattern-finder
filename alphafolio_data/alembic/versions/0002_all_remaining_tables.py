"""all remaining tables for alphafolio_data

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-19

Reverse-engineered from collector code in us/, kr/, index/, utils/.
Schemas derived from `CREATE TEMP TABLE temp_<name>` blocks (asyncpg COPY
optimization pattern) and `INSERT INTO ...` statements, with PK inferred
from `ON CONFLICT (...)` clauses.

All tables use `CREATE TABLE IF NOT EXISTS` so this migration is safe to
apply on databases where these tables were created by ad-hoc scripts
without Alembic tracking. Downgrade drops them unconditionally.

This migration also ALTERs us_symbol (created in 0001 with only `symbol`
+ timestamps) to add columns used by us/finnhub_symbol.py.
"""
from alembic import op


revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Extend us_symbol (0001 only created symbol + timestamps; finnhub
    # collector populates display_symbol, description, figi, mic, currency, type)
    # ------------------------------------------------------------------
    op.execute("ALTER TABLE us_symbol ADD COLUMN IF NOT EXISTS display_symbol VARCHAR(20);")
    op.execute("ALTER TABLE us_symbol ADD COLUMN IF NOT EXISTS description VARCHAR(255);")
    op.execute("ALTER TABLE us_symbol ADD COLUMN IF NOT EXISTS figi VARCHAR(20);")
    op.execute("ALTER TABLE us_symbol ADD COLUMN IF NOT EXISTS mic VARCHAR(10);")
    op.execute("ALTER TABLE us_symbol ADD COLUMN IF NOT EXISTS currency VARCHAR(10);")
    op.execute("ALTER TABLE us_symbol ADD COLUMN IF NOT EXISTS type VARCHAR(50);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_symbol_mic  ON us_symbol(mic);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_symbol_type ON us_symbol(type);")

    # ==================================================================
    # US Daily Pipeline (Steps 1-13)
    # ==================================================================

    # 1) us_daily — temp table at us/alphavantage.py:887
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_daily (
            date            DATE NOT NULL,
            symbol          VARCHAR(20) NOT NULL,
            open            DECIMAL(12,4),
            high            DECIMAL(12,4),
            low             DECIMAL(12,4),
            close           DECIMAL(12,4),
            volume          BIGINT,
            change_amount   DECIMAL(12,4),
            change_rate     DECIMAL(8,4),
            per             DECIMAL(6,2),
            pbr             DECIMAL(8,2),
            avg_volume_5d         BIGINT,
            avg_volume_20d        BIGINT,
            avg_volume_50d        BIGINT,
            avg_volume_200d       BIGINT,
            avg_trading_value_5d  BIGINT,
            avg_trading_value_20d BIGINT,
            updated_at      TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_daily_date ON us_daily(date);")

    # 2) us_vwap_base — temp table at us/alphavantage.py:1893
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_vwap_base (
            symbol          VARCHAR(20) NOT NULL,
            indicator       VARCHAR(100),
            last_refreshed  DATE,
            interval        VARCHAR(20),
            time_zone       VARCHAR(50),
            date            DATE,
            datetime        TIMESTAMP NOT NULL,
            vwap            DECIMAL(12,4),
            created_at      TIMESTAMP,
            PRIMARY KEY (symbol, datetime)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_vwap_base_date   ON us_vwap_base(date);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_vwap_base_symbol ON us_vwap_base(symbol);")

    # 3) us_news — INSERT at us/us_news.py:409, PK from ON CONFLICT (url, ticker)
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_news (
            title                   TEXT,
            url                     TEXT NOT NULL,
            time_published          TIMESTAMP,
            authors                 TEXT,
            summary                 TEXT,
            banner_image            TEXT,
            source                  TEXT,
            category_within_source  TEXT,
            source_domain           TEXT,
            topics                  JSONB,
            overall_sentiment_score DOUBLE PRECISION,
            overall_sentiment_label TEXT,
            ticker                  VARCHAR(20) NOT NULL,
            relevance_score_t       DOUBLE PRECISION,
            ticker_sentiment_score  DOUBLE PRECISION,
            ticker_sentiment_label  TEXT,
            created_at              TIMESTAMP,
            PRIMARY KEY (url, ticker)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_news_ticker         ON us_news(ticker);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_news_time_published ON us_news(time_published);")

    # 4) us_daily_etf — INSERT at us/us_etf.py:356
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_daily_etf (
            symbol     VARCHAR(20) NOT NULL,
            date       DATE NOT NULL,
            open       DECIMAL(12,4),
            high       DECIMAL(12,4),
            low        DECIMAL(12,4),
            close      DECIMAL(12,4),
            volume     BIGINT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_daily_etf_date ON us_daily_etf(date);")

    # 5) market_index — INSERT at index/index.py:43
    op.execute("""
        CREATE TABLE IF NOT EXISTS market_index (
            exchange      VARCHAR(50) NOT NULL,
            date          DATE NOT NULL,
            close         DOUBLE PRECISION,
            change_amount DOUBLE PRECISION,
            change_rate   DOUBLE PRECISION,
            open          DOUBLE PRECISION,
            high          DOUBLE PRECISION,
            low           DOUBLE PRECISION,
            volume        BIGINT,
            trading_value BIGINT,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (exchange, date)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_market_index_date ON market_index(date);")

    # 6) us_indicators — aggregated indicator table, INSERT at us/us_calculator.py:1555
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_indicators (
            symbol            VARCHAR(20) NOT NULL,
            stock_name        TEXT,
            date              DATE NOT NULL,
            rsi               DOUBLE PRECISION,
            macd              DOUBLE PRECISION,
            macd_signal       DOUBLE PRECISION,
            macd_hist         DOUBLE PRECISION,
            real_upper_band   DOUBLE PRECISION,
            real_middle_band  DOUBLE PRECISION,
            real_lower_band   DOUBLE PRECISION,
            vwap              DOUBLE PRECISION,
            eod_vwap          DOUBLE PRECISION,
            avg_vwap          DOUBLE PRECISION,
            price_vs_vwap     DOUBLE PRECISION,
            atr               DOUBLE PRECISION,
            slowk             DOUBLE PRECISION,
            slowd             DOUBLE PRECISION,
            mfi               DOUBLE PRECISION,
            roc               DOUBLE PRECISION,
            sma               DOUBLE PRECISION,
            ema               DOUBLE PRECISION,
            adx               DOUBLE PRECISION,
            wma               DOUBLE PRECISION,
            aroon             DOUBLE PRECISION,
            cci               DOUBLE PRECISION,
            obv               DOUBLE PRECISION,
            created_at        TIMESTAMP,
            PRIMARY KEY (date, symbol)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_indicators_symbol ON us_indicators(symbol);")

    # 7) Individual indicator tables — us/us_calculator.py
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_rsi (
            symbol         VARCHAR(20) NOT NULL,
            indicator      TEXT,
            last_refreshed DATE,
            interval       VARCHAR(20),
            time_period    INTEGER,
            series_type    VARCHAR(20),
            time_zone      VARCHAR(50),
            date           DATE NOT NULL,
            rsi            DOUBLE PRECISION,
            created_at     TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS us_macd (
            symbol         VARCHAR(20) NOT NULL,
            indicator      TEXT,
            last_refreshed DATE,
            interval       VARCHAR(20),
            fast_period    INTEGER,
            slow_period    INTEGER,
            signal_period  INTEGER,
            series_type    VARCHAR(20),
            date           DATE NOT NULL,
            macd           DOUBLE PRECISION,
            macd_signal    DOUBLE PRECISION,
            macd_hist      DOUBLE PRECISION,
            created_at     TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS us_bbands (
            symbol                     VARCHAR(20) NOT NULL,
            indicator                  TEXT,
            last_refreshed             DATE,
            interval                   VARCHAR(20),
            time_period                INTEGER,
            deviation_multiplier_upper DOUBLE PRECISION,
            deviation_multiplier_lower DOUBLE PRECISION,
            ma_type                    INTEGER,
            series_type                VARCHAR(20),
            time_zone                  VARCHAR(50),
            date                       DATE NOT NULL,
            real_upper_band            DOUBLE PRECISION,
            real_middle_band           DOUBLE PRECISION,
            real_lower_band            DOUBLE PRECISION,
            created_at                 TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS us_stoch (
            symbol         VARCHAR(20) NOT NULL,
            indicator      TEXT,
            last_refreshed DATE,
            interval       VARCHAR(20),
            fastk_period   INTEGER,
            slowk_period   INTEGER,
            slowk_ma_type  INTEGER,
            slowd_period   INTEGER,
            slowd_ma_type  INTEGER,
            time_zone      VARCHAR(50),
            date           DATE NOT NULL,
            slowk          DOUBLE PRECISION,
            slowd          DOUBLE PRECISION,
            created_at     TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS us_mfi (
            symbol         VARCHAR(20) NOT NULL,
            indicator      TEXT,
            last_refreshed DATE,
            interval       VARCHAR(20),
            time_period    INTEGER,
            time_zone      VARCHAR(50),
            date           DATE NOT NULL,
            mfi            DOUBLE PRECISION,
            created_at     TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS us_roc (
            symbol         VARCHAR(20) NOT NULL,
            indicator      TEXT,
            last_refreshed DATE,
            interval       VARCHAR(20),
            time_period    INTEGER,
            series_type    VARCHAR(20),
            time_zone      VARCHAR(50),
            date           DATE NOT NULL,
            roc            DOUBLE PRECISION,
            created_at     TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS us_sma (
            symbol         VARCHAR(20) NOT NULL,
            indicator      TEXT,
            last_refreshed DATE,
            interval       VARCHAR(20),
            time_period    INTEGER,
            series_type    VARCHAR(20),
            time_zone      VARCHAR(50),
            date           DATE NOT NULL,
            sma            DOUBLE PRECISION,
            created_at     TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS us_ema (
            symbol         VARCHAR(20) NOT NULL,
            indicator      TEXT,
            last_refreshed DATE,
            interval       VARCHAR(20),
            time_period    INTEGER,
            series_type    VARCHAR(20),
            time_zone      VARCHAR(50),
            date           DATE NOT NULL,
            ema            DOUBLE PRECISION,
            created_at     TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS us_adx (
            symbol         VARCHAR(20) NOT NULL,
            indicator      TEXT,
            last_refreshed DATE,
            interval       VARCHAR(20),
            time_period    INTEGER,
            time_zone      VARCHAR(50),
            date           DATE NOT NULL,
            adx            DOUBLE PRECISION,
            created_at     TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS us_wma (
            symbol         VARCHAR(20) NOT NULL,
            indicator      TEXT,
            last_refreshed DATE,
            interval       VARCHAR(20),
            time_period    INTEGER,
            series_type    VARCHAR(20),
            time_zone      VARCHAR(50),
            date           DATE NOT NULL,
            wma            DOUBLE PRECISION,
            created_at     TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS us_aroon (
            symbol         VARCHAR(20) NOT NULL,
            indicator      TEXT,
            last_refreshed DATE,
            interval       VARCHAR(20),
            time_period    INTEGER,
            time_zone      VARCHAR(50),
            date           DATE NOT NULL,
            aroon          DOUBLE PRECISION,
            created_at     TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS us_cci (
            symbol         VARCHAR(20) NOT NULL,
            indicator      TEXT,
            last_refreshed DATE,
            interval       VARCHAR(20),
            time_period    INTEGER,
            time_zone      VARCHAR(50),
            date           DATE NOT NULL,
            cci            DOUBLE PRECISION,
            created_at     TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS us_obv (
            symbol         VARCHAR(20) NOT NULL,
            indicator      TEXT,
            last_refreshed DATE,
            interval       VARCHAR(20),
            time_zone      VARCHAR(50),
            date           DATE NOT NULL,
            obv            DOUBLE PRECISION,
            created_at     TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS us_atr (
            symbol         VARCHAR(20) NOT NULL,
            indicator      TEXT,
            last_refreshed DATE,
            interval       VARCHAR(20),
            time_period    INTEGER,
            time_zone      VARCHAR(50),
            date           DATE NOT NULL,
            atr            DOUBLE PRECISION,
            created_at     TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)

    # us_vwap (calculated VWAP per-day) — INSERT at us/us_calculator.py:845
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_vwap (
            symbol        VARCHAR(20) NOT NULL,
            close         DOUBLE PRECISION,
            volume        BIGINT,
            eod_vwap      DOUBLE PRECISION,
            avg_vwap      DOUBLE PRECISION,
            price_vs_vwap DOUBLE PRECISION,
            date          DATE NOT NULL,
            created_at    TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)

    # 8) us_option — temp table at us/us_option.py:305. Production table is
    # range-partitioned by `date` (see ensure_daily_partition_exists). We
    # create a non-partitioned table here; partitioning is a runtime
    # optimization layered on top.
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_option (
            contract_id        VARCHAR(30) NOT NULL,
            symbol             VARCHAR(10),
            expiration         DATE,
            strike             NUMERIC(10, 2),
            type               VARCHAR(10),
            last               NUMERIC(10, 2),
            mark               NUMERIC(10, 2),
            bid                NUMERIC(10, 2),
            bid_size           INTEGER,
            ask                NUMERIC(10, 2),
            ask_size           INTEGER,
            volume             INTEGER,
            open_interest      INTEGER,
            date               DATE NOT NULL,
            implied_volatility NUMERIC(10, 5),
            delta              NUMERIC(10, 5),
            gamma              NUMERIC(10, 5),
            theta              NUMERIC(10, 5),
            vega               NUMERIC(10, 5),
            rho                NUMERIC(10, 5),
            PRIMARY KEY (contract_id, date)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_option_symbol     ON us_option(symbol);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_option_date       ON us_option(date);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_option_expiration ON us_option(expiration);")

    # 9) us_option_daily_summary — INSERTs at us/us_option.py:542, populate_option_summary.py:168
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_option_daily_summary (
            symbol                 VARCHAR(10) NOT NULL,
            date                   DATE NOT NULL,
            total_call_volume      BIGINT,
            total_put_volume       BIGINT,
            avg_implied_volatility NUMERIC(12, 6),
            min_implied_volatility NUMERIC(12, 6),
            max_implied_volatility NUMERIC(12, 6),
            avg_call_iv            NUMERIC(12, 6),
            avg_put_iv             NUMERIC(12, 6),
            call_option_count      INTEGER,
            put_option_count       INTEGER,
            call_gex               DOUBLE PRECISION,
            put_gex                DOUBLE PRECISION,
            net_gex                DOUBLE PRECISION,
            gex_ratio              DOUBLE PRECISION,
            gamma_flip_distance    DOUBLE PRECISION,
            PRIMARY KEY (symbol, date)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_option_daily_summary_date ON us_option_daily_summary(date);")

    # 10) us_move_index — us/us_move_index_collector.py
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_move_index (
            name       TEXT,
            interval   VARCHAR(20),
            unit       VARCHAR(50),
            date       DATE PRIMARY KEY,
            value      DOUBLE PRECISION,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 11) FRED macro tables — all share same shape (name, interval, unit, date, value)
    for tname in (
        "us_dollar_index",
        "us_credit_spread",
        "us_vix",
        "us_fed_rrp",
        "us_gdp",
        "us_pmi",
    ):
        op.execute(f"""
            CREATE TABLE IF NOT EXISTS {tname} (
                name       TEXT,
                interval   VARCHAR(20),
                unit       VARCHAR(50),
                date       DATE PRIMARY KEY,
                value      DOUBLE PRECISION,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)

    # 12) us_earnings_calendar — temp table at us/finance_data.py:2829
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_earnings_calendar (
            symbol           VARCHAR(10) NOT NULL,
            name             VARCHAR(100),
            reportdate       DATE,
            fiscaldateending DATE,
            estimate         NUMERIC(18,4),
            currency         VARCHAR(10),
            created_at       TIMESTAMP,
            PRIMARY KEY (symbol, reportdate, fiscaldateending)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_earnings_calendar_reportdate ON us_earnings_calendar(reportdate);")

    # 13) us_insider_transactions — temp table at us/finance_data.py:3429
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_insider_transactions (
            date                    DATE NOT NULL,
            symbol                  VARCHAR(10) NOT NULL,
            executive               TEXT[] NOT NULL,
            executive_title         TEXT[] NOT NULL,
            security_type           VARCHAR(255),
            acquisition_or_disposal VARCHAR(255),
            shares                  NUMERIC,
            share_price             NUMERIC,
            created_at              TIMESTAMP,
            PRIMARY KEY (date, symbol, executive, executive_title)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_insider_transactions_symbol ON us_insider_transactions(symbol);")

    # ==================================================================
    # Additional US tables (used by non-daily endpoints)
    # ==================================================================

    # us_weekly — INSERT at us/alphavantage.py:2368
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_weekly (
            date       DATE NOT NULL,
            symbol     VARCHAR(20) NOT NULL,
            open       DECIMAL(12,4),
            high       DECIMAL(12,4),
            low        DECIMAL(12,4),
            close      DECIMAL(12,4),
            volume     BIGINT,
            updated_at TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_weekly_date ON us_weekly(date);")

    # us_monthly — INSERT at us/alphavantage.py:1454
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_monthly (
            date       DATE NOT NULL,
            symbol     VARCHAR(20) NOT NULL,
            open       DECIMAL(12,4),
            high       DECIMAL(12,4),
            low        DECIMAL(12,4),
            close      DECIMAL(12,4),
            volume     BIGINT,
            updated_at TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)

    # us_income_statement — temp at us/finance_data.py:398
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_income_statement (
            symbol                                 VARCHAR(20) NOT NULL,
            fiscal_date_ending                     DATE NOT NULL,
            reported_currency                      VARCHAR(10),
            gross_profit                           BIGINT,
            total_revenue                          BIGINT,
            cost_of_revenue                        BIGINT,
            cost_of_goods_and_services_sold        BIGINT,
            operating_income                       BIGINT,
            selling_general_and_administrative     BIGINT,
            research_and_development               BIGINT,
            operating_expenses                     BIGINT,
            investment_income_net                  BIGINT,
            net_interest_income                    BIGINT,
            interest_income                        BIGINT,
            interest_expense                       BIGINT,
            non_interest_income                    BIGINT,
            other_non_operating_income             BIGINT,
            depreciation                           BIGINT,
            depreciation_and_amortization          BIGINT,
            income_before_tax                      BIGINT,
            income_tax_expense                     BIGINT,
            interest_and_debt_expense              BIGINT,
            net_income_from_continuing_operations  BIGINT,
            comprehensive_income_net_of_tax        BIGINT,
            ebit                                   BIGINT,
            ebitda                                 BIGINT,
            net_income                             BIGINT,
            created_at                             TIMESTAMP,
            PRIMARY KEY (symbol, fiscal_date_ending)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_us_income_statement_fiscal ON us_income_statement(fiscal_date_ending);")

    # us_balance_sheet — temp at us/finance_data.py:1045
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_balance_sheet (
            symbol                                       VARCHAR(20) NOT NULL,
            fiscal_date_ending                           DATE NOT NULL,
            reported_currency                            VARCHAR(10),
            total_assets                                 BIGINT,
            total_current_assets                         BIGINT,
            cash_and_cash_equivalents_at_carrying_value  BIGINT,
            cash_and_short_term_investments              BIGINT,
            inventory                                    BIGINT,
            current_net_receivables                      BIGINT,
            total_non_current_assets                     BIGINT,
            property_plant_equipment                     BIGINT,
            accumulated_depreciation_amortization_ppe    BIGINT,
            intangible_assets                            BIGINT,
            intangible_assets_excluding_goodwill         BIGINT,
            goodwill                                     BIGINT,
            investments                                  BIGINT,
            long_term_investments                        BIGINT,
            short_term_investments                       BIGINT,
            other_current_assets                         BIGINT,
            other_non_current_assets                     BIGINT,
            total_liabilities                            BIGINT,
            total_current_liabilities                    BIGINT,
            current_accounts_payable                     BIGINT,
            deferred_revenue                             BIGINT,
            current_debt                                 BIGINT,
            short_term_debt                              BIGINT,
            total_non_current_liabilities                BIGINT,
            capital_lease_obligations                    BIGINT,
            long_term_debt                               BIGINT,
            current_long_term_debt                       BIGINT,
            long_term_debt_noncurrent                    BIGINT,
            short_long_term_debt_total                   BIGINT,
            other_current_liabilities                    BIGINT,
            other_non_current_liabilities                BIGINT,
            total_shareholder_equity                     BIGINT,
            treasury_stock                               BIGINT,
            retained_earnings                            BIGINT,
            common_stock                                 BIGINT,
            common_stock_shares_outstanding              BIGINT,
            created_at                                   TIMESTAMP,
            PRIMARY KEY (symbol, fiscal_date_ending)
        );
    """)

    # us_cash_flow — temp at us/finance_data.py:1706
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_cash_flow (
            symbol                                                           VARCHAR(20) NOT NULL,
            fiscal_date_ending                                               DATE NOT NULL,
            reported_currency                                                VARCHAR(10),
            operating_cashflow                                               BIGINT,
            payments_for_operating_activities                                BIGINT,
            proceeds_from_operating_activities                               BIGINT,
            change_in_operating_liabilities                                  BIGINT,
            change_in_operating_assets                                       BIGINT,
            depreciation_depletion_and_amortization                          BIGINT,
            capital_expenditures                                             BIGINT,
            change_in_receivables                                            BIGINT,
            change_in_inventory                                              BIGINT,
            profit_loss                                                      BIGINT,
            cashflow_from_investment                                         BIGINT,
            cashflow_from_financing                                          BIGINT,
            proceeds_from_repayments_of_short_term_debt                      BIGINT,
            payments_for_repurchase_of_common_stock                          BIGINT,
            payments_for_repurchase_of_equity                                BIGINT,
            payments_for_repurchase_of_preferred_stock                       BIGINT,
            dividend_payout                                                  BIGINT,
            dividend_payout_common_stock                                     BIGINT,
            dividend_payout_preferred_stock                                  BIGINT,
            proceeds_from_issuance_of_common_stock                           BIGINT,
            proceeds_from_issuance_of_long_term_debt_and_capital_securities  BIGINT,
            proceeds_from_issuance_of_preferred_stock                        BIGINT,
            proceeds_from_repurchase_of_equity                               BIGINT,
            proceeds_from_sale_of_treasury_stock                             BIGINT,
            change_in_cash_and_cash_equivalents                              BIGINT,
            change_in_exchange_rate                                          BIGINT,
            net_income                                                       BIGINT,
            created_at                                                       TIMESTAMP,
            PRIMARY KEY (symbol, fiscal_date_ending)
        );
    """)

    # us_earnings_estimates — temp at us/finance_data.py:2292
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_earnings_estimates (
            symbol                                       VARCHAR(20) NOT NULL,
            estimate_date                                DATE NOT NULL,
            horizon                                      VARCHAR(50) NOT NULL,
            eps_estimate_average                         NUMERIC(18,4),
            eps_estimate_high                            NUMERIC(18,4),
            eps_estimate_low                             NUMERIC(18,4),
            eps_estimate_analyst_count                   INTEGER,
            eps_estimate_average_7_days_ago              NUMERIC(18,4),
            eps_estimate_average_30_days_ago             NUMERIC(18,4),
            eps_estimate_average_60_days_ago             NUMERIC(18,4),
            eps_estimate_average_90_days_ago             NUMERIC(18,4),
            eps_estimate_revision_up_trailing_7_days     INTEGER,
            eps_estimate_revision_down_trailing_7_days   INTEGER,
            eps_estimate_revision_up_trailing_30_days    INTEGER,
            eps_estimate_revision_down_trailing_30_days  INTEGER,
            revenue_estimate_average                     NUMERIC(18,2),
            revenue_estimate_high                        NUMERIC(18,2),
            revenue_estimate_low                         NUMERIC(18,2),
            revenue_estimate_analyst_count               INTEGER,
            created_at                                   TIMESTAMP,
            PRIMARY KEY (symbol, estimate_date, horizon)
        );
    """)

    # us_dividends — temp at us/finance_data.py:3839
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_dividends (
            symbol           VARCHAR(10) NOT NULL,
            ex_dividend_date DATE NOT NULL,
            declaration_date DATE,
            record_date      DATE,
            payment_date     DATE,
            amount           NUMERIC(18, 4),
            created_at       TIMESTAMP,
            PRIMARY KEY (symbol, ex_dividend_date)
        );
    """)

    # us_splits — temp at us/finance_data.py:4285
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_splits (
            symbol         VARCHAR(10) NOT NULL,
            effective_date DATE NOT NULL,
            split_factor   DECIMAL(18, 4),
            created_at     TIMESTAMP,
            PRIMARY KEY (symbol, effective_date)
        );
    """)

    # us_fed_funds_rate / us_treasury_yield / us_cpi / us_unemployment_rate
    # — all share same shape (date, interval, value, name, unit, created_at)
    for tname in ("us_fed_funds_rate", "us_treasury_yield", "us_cpi", "us_unemployment_rate"):
        op.execute(f"""
            CREATE TABLE IF NOT EXISTS {tname} (
                date       DATE NOT NULL,
                interval   VARCHAR(20) NOT NULL,
                value      DECIMAL(8,4),
                name       TEXT,
                unit       VARCHAR(50),
                created_at TIMESTAMP,
                PRIMARY KEY (date, interval)
            );
        """)

    # us_ipo_calendar — INSERT at us/alphavantage.py:2775
    op.execute("""
        CREATE TABLE IF NOT EXISTS us_ipo_calendar (
            symbol         VARCHAR(20) NOT NULL,
            name           TEXT,
            ipodate        DATE NOT NULL,
            pricerangelow  NUMERIC(18,4),
            pricerangehigh NUMERIC(18,4),
            currency       VARCHAR(10),
            exchange       VARCHAR(50),
            PRIMARY KEY (symbol, ipodate)
        );
    """)

    # ==================================================================
    # KR (Korea) tables — kr/krx.py, kr/krx_index.py, kr/bok.py
    # ==================================================================

    # kr_intraday — INSERT at kr/krx.py:634. PK from ON CONFLICT (symbol).
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_intraday (
            symbol                 VARCHAR(10) PRIMARY KEY,
            stock_name             VARCHAR(200),
            exchange               VARCHAR(50),
            close                  NUMERIC(18,2),
            change_amount          NUMERIC(18,2),
            change_rate            NUMERIC(8,2),
            open                   NUMERIC(18,2),
            high                   NUMERIC(18,2),
            low                    NUMERIC(18,2),
            volume                 BIGINT,
            trading_value          BIGINT,
            market_cap             BIGINT,
            listed_shares          BIGINT,
            avg_volume_5d          BIGINT,
            avg_volume_20d         BIGINT,
            avg_volume_60d         BIGINT,
            avg_volume_200d        BIGINT,
            avg_trading_value_5d   BIGINT,
            avg_trading_value_20d  BIGINT,
            avg_trading_value_60d  BIGINT,
            avg_trading_value_200d BIGINT,
            created_at             TIMESTAMP,
            updated_at             TIMESTAMP
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_kr_intraday_exchange ON kr_intraday(exchange);")

    # kr_intraday_detail — INSERT at kr/krx.py:811. PK (symbol).
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_intraday_detail (
            symbol         VARCHAR(10) PRIMARY KEY,
            stock_name     VARCHAR(200),
            close          NUMERIC(18,2),
            change_amount  NUMERIC(18,2),
            change_rate    NUMERIC(8,2),
            eps            NUMERIC(18,2),
            per            NUMERIC(8,2),
            bps            NUMERIC(18,2),
            pbr            NUMERIC(8,2),
            dps            NUMERIC(18,2),
            dividend_yield NUMERIC(8,2),
            created_at     TIMESTAMP,
            updated_at     TIMESTAMP
        );
    """)

    # kr_intraday_total — INSERT at kr/krx.py:871. PK (symbol, date).
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_intraday_total (
            symbol                 VARCHAR(10) NOT NULL,
            stock_name             VARCHAR(200),
            exchange               VARCHAR(50),
            close                  NUMERIC(18,2),
            change_amount          NUMERIC(18,2),
            change_rate            NUMERIC(8,2),
            open                   NUMERIC(18,2),
            high                   NUMERIC(18,2),
            low                    NUMERIC(18,2),
            volume                 BIGINT,
            trading_value          BIGINT,
            market_cap             BIGINT,
            listed_shares          BIGINT,
            eps                    NUMERIC(18,2),
            per                    NUMERIC(8,2),
            roe                    NUMERIC(8,2),
            bps                    NUMERIC(18,2),
            pbr                    NUMERIC(8,2),
            dps                    NUMERIC(18,2),
            dividend_yield         NUMERIC(8,2),
            avg_trading_value_5d   BIGINT,
            avg_trading_value_20d  BIGINT,
            avg_trading_value_60d  BIGINT,
            avg_trading_value_200d BIGINT,
            avg_volume_5d          BIGINT,
            avg_volume_20d         BIGINT,
            avg_volume_60d         BIGINT,
            avg_volume_200d        BIGINT,
            date                   DATE NOT NULL,
            PRIMARY KEY (symbol, date)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_kr_intraday_total_date ON kr_intraday_total(date);")

    # kr_investor_daily_trading — INSERT at kr/krx.py:1029
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_investor_daily_trading (
            investor_type   VARCHAR(100) NOT NULL,
            sell_volume     BIGINT,
            buy_volume      BIGINT,
            net_buy_volume  BIGINT,
            sell_value      BIGINT,
            buy_value       BIGINT,
            net_buy_value   BIGINT,
            date            DATE NOT NULL,
            PRIMARY KEY (investor_type, date)
        );
    """)

    # kr_individual_investor_daily_trading — INSERT at kr/krx.py:2170
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_individual_investor_daily_trading (
            date                  DATE NOT NULL,
            symbol                VARCHAR(10) NOT NULL,
            inst_buy_volume       BIGINT,
            inst_sell_volume      BIGINT,
            inst_net_volume       BIGINT,
            inst_buy_value        BIGINT,
            inst_sell_value       BIGINT,
            inst_net_value        BIGINT,
            inst_buy_ratio        NUMERIC(8,2),
            retail_buy_volume     BIGINT,
            retail_sell_volume    BIGINT,
            retail_net_volume     BIGINT,
            retail_buy_value      BIGINT,
            retail_sell_value     BIGINT,
            retail_net_value      BIGINT,
            retail_buy_ratio      NUMERIC(8,2),
            foreign_buy_volume    BIGINT,
            foreign_sell_volume   BIGINT,
            foreign_net_volume    BIGINT,
            foreign_buy_value     BIGINT,
            foreign_sell_value    BIGINT,
            foreign_net_value     BIGINT,
            foreign_buy_ratio     NUMERIC(8,2),
            total_buy_volume      BIGINT,
            total_sell_volume     BIGINT,
            total_net_volume      BIGINT,
            total_buy_value       BIGINT,
            total_sell_value      BIGINT,
            total_net_value       BIGINT,
            PRIMARY KEY (date, symbol)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_kr_indiv_inv_symbol ON kr_individual_investor_daily_trading(symbol);")

    # kr_program_daily_trading — INSERT at kr/krx.py:1147
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_program_daily_trading (
            category        VARCHAR(100) NOT NULL,
            sell_volume     BIGINT,
            buy_volume      BIGINT,
            net_buy_volume  BIGINT,
            sell_value      BIGINT,
            buy_value       BIGINT,
            net_buy_value   BIGINT,
            date            DATE NOT NULL,
            PRIMARY KEY (category, date)
        );
    """)

    # kr_blocktrades — INSERT at kr/krx.py:1278
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_blocktrades (
            symbol            VARCHAR(10) NOT NULL,
            stock_name        VARCHAR(200),
            close             NUMERIC(18,2),
            change_amount     NUMERIC(18,2),
            change_rate       NUMERIC(8,2),
            volume            BIGINT,
            block_volume      BIGINT,
            block_volume_rate NUMERIC(8,2),
            date              DATE NOT NULL,
            PRIMARY KEY (symbol, date)
        );
    """)

    # kr_foreign_ownership — INSERT at kr/krx.py:1403
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_foreign_ownership (
            symbol             VARCHAR(10) NOT NULL,
            stock_name         VARCHAR(200),
            close              NUMERIC(18,2),
            change_amount      NUMERIC(18,2),
            change_rate        NUMERIC(8,2),
            listed_shares      BIGINT,
            foreign_ownership  BIGINT,
            foreign_rate       NUMERIC(8,2),
            foreign_limit      BIGINT,
            foreign_rate_limit NUMERIC(8,2),
            date               DATE NOT NULL,
            PRIMARY KEY (symbol, date)
        );
    """)

    # kr_stock_basic — INSERT at kr/krx.py:1551
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_stock_basic (
            standard_symbol     VARCHAR(20),
            symbol              VARCHAR(10) PRIMARY KEY,
            standard_stock_name VARCHAR(200),
            stock_name          VARCHAR(200),
            stock_name_eng      VARCHAR(200),
            listed_date         DATE,
            exchange            VARCHAR(50),
            securities_type     VARCHAR(50),
            department          VARCHAR(200),
            stock_type          VARCHAR(50),
            par_value           BIGINT,
            listed_shares       BIGINT,
            created_at          TIMESTAMP,
            updated_at          TIMESTAMP
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_kr_stock_basic_exchange ON kr_stock_basic(exchange);")

    # kr_stock_detail — INSERT at kr/krx.py:1698
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_stock_detail (
            symbol        VARCHAR(10) PRIMARY KEY,
            stock_name    VARCHAR(200),
            exchange      VARCHAR(20),
            department    VARCHAR(200),
            industry_code VARCHAR(20),
            industry      VARCHAR(100),
            fiscal_month  NUMERIC(4,0),
            advisor       VARCHAR(200),
            listed_shares BIGINT,
            par_value     BIGINT,
            capital       BIGINT,
            currency      VARCHAR(3),
            ceo_name      VARCHAR(100),
            phone         VARCHAR(100),
            address       VARCHAR(255),
            created_at    TIMESTAMP,
            updated_at    TIMESTAMP
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_kr_stock_detail_industry ON kr_stock_detail(industry);")

    # kr_benchmark_index — INSERT at kr/krx_index.py:392
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_benchmark_index (
            date           DATE NOT NULL,
            index_category VARCHAR(50),
            index_name     VARCHAR(100) NOT NULL,
            close          NUMERIC(18,4),
            change_amount  NUMERIC(18,4),
            change_rate    NUMERIC(8,4),
            open           NUMERIC(18,4),
            high           NUMERIC(18,4),
            low            NUMERIC(18,4),
            volume         BIGINT,
            trading_value  BIGINT,
            market_cap     BIGINT,
            PRIMARY KEY (index_name, date)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_kr_benchmark_index_date ON kr_benchmark_index(date);")

    # kr_indicators — INSERT at kr/kr_calculator.py:1156 (same columns as us_indicators)
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_indicators (
            symbol            VARCHAR(10) NOT NULL,
            stock_name        VARCHAR(200),
            date              DATE NOT NULL,
            rsi               DOUBLE PRECISION,
            macd              DOUBLE PRECISION,
            macd_signal       DOUBLE PRECISION,
            macd_hist         DOUBLE PRECISION,
            real_upper_band   DOUBLE PRECISION,
            real_middle_band  DOUBLE PRECISION,
            real_lower_band   DOUBLE PRECISION,
            vwap              DOUBLE PRECISION,
            eod_vwap          DOUBLE PRECISION,
            avg_vwap          DOUBLE PRECISION,
            price_vs_vwap     DOUBLE PRECISION,
            atr               DOUBLE PRECISION,
            slowk             DOUBLE PRECISION,
            slowd             DOUBLE PRECISION,
            mfi               DOUBLE PRECISION,
            roc               DOUBLE PRECISION,
            sma               DOUBLE PRECISION,
            ema               DOUBLE PRECISION,
            adx               DOUBLE PRECISION,
            wma               DOUBLE PRECISION,
            aroon             DOUBLE PRECISION,
            cci               DOUBLE PRECISION,
            obv               DOUBLE PRECISION,
            created_at        TIMESTAMP,
            updated_at        TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_kr_indicators_date ON kr_indicators(date);")

    # kr_research_reports — INSERT at kr/research_crawler.py:414
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_research_reports (
            report_id          VARCHAR(100) NOT NULL,
            stock_name         VARCHAR(200),
            symbol             VARCHAR(10),
            title              TEXT,
            securities_firm    VARCHAR(200) NOT NULL,
            date               DATE NOT NULL,
            target_price       NUMERIC(18,2),
            investment_opinion VARCHAR(50),
            summary            TEXT,
            pdf_url            TEXT,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at         TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (report_id, securities_firm, date)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_kr_research_reports_symbol ON kr_research_reports(symbol);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_kr_research_reports_date   ON kr_research_reports(date);")

    # bok_economic_indicators — INSERT at kr/bok.py:309
    op.execute("""
        CREATE TABLE IF NOT EXISTS bok_economic_indicators (
            stat_code     VARCHAR(20) NOT NULL,
            stat_name     TEXT,
            item_code1    VARCHAR(50) NOT NULL DEFAULT '',
            item_name1    TEXT,
            item_code2    VARCHAR(50) NOT NULL DEFAULT '',
            item_name2    TEXT,
            item_code3    VARCHAR(50) NOT NULL DEFAULT '',
            item_name3    TEXT,
            item_code4    VARCHAR(50) NOT NULL DEFAULT '',
            item_name4    TEXT,
            unit_name     VARCHAR(100),
            wgt           DOUBLE PRECISION,
            cycle         VARCHAR(20),
            time_value    DATE NOT NULL,
            time_original VARCHAR(20),
            data_value    DOUBLE PRECISION,
            created_at    TIMESTAMP,
            updated_at    TIMESTAMP,
            PRIMARY KEY (stat_code, time_value, item_code1, item_code2, item_code3, item_code4)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_bok_econ_ind_time ON bok_economic_indicators(time_value);")

    # exchange_rate — INSERT at kr/bok.py:293
    op.execute("""
        CREATE TABLE IF NOT EXISTS exchange_rate (
            stat_code     VARCHAR(20) NOT NULL,
            stat_name     TEXT,
            item_code1    VARCHAR(50) NOT NULL,
            item_name1    TEXT,
            unit_name     VARCHAR(100),
            cycle         VARCHAR(20),
            time_value    DATE NOT NULL,
            time_original VARCHAR(20),
            data_value    DOUBLE PRECISION,
            created_at    TIMESTAMP,
            updated_at    TIMESTAMP,
            PRIMARY KEY (stat_code, time_value, item_code1)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_exchange_rate_time ON exchange_rate(time_value);")

    # ==================================================================
    # DART tables — kr/dart.py
    # ==================================================================

    # dart_company_info — INSERT at kr/dart.py:152. PK from ON CONFLICT (stock_code).
    op.execute("""
        CREATE TABLE IF NOT EXISTS dart_company_info (
            corp_code     VARCHAR(20) NOT NULL,
            corp_name     VARCHAR(200),
            corp_eng_name VARCHAR(200),
            stock_code    VARCHAR(10) PRIMARY KEY,
            modify_date   VARCHAR(20)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_dart_company_info_corp_code ON dart_company_info(corp_code);")

    # kr_financial_position — INSERT at kr/dart.py:441
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_financial_position (
            stock_name         VARCHAR(200),
            symbol             VARCHAR(10),
            report_code        VARCHAR(20) NOT NULL,
            bsns_year          INTEGER NOT NULL,
            corp_code          VARCHAR(20) NOT NULL,
            fs_div             VARCHAR(10) NOT NULL,
            sj_div             VARCHAR(20),
            sj_nm              VARCHAR(200),
            account_id         VARCHAR(200),
            account_nm         VARCHAR(200),
            account_detail     VARCHAR(500),
            thstrm_nm          VARCHAR(100),
            thstrm_amount      BIGINT,
            thstrm_add_amount  BIGINT,
            frmtrm_nm          VARCHAR(100),
            frmtrm_amount      BIGINT,
            frmtrm_q_nm        VARCHAR(100),
            frmtrm_q_amount    BIGINT,
            frmtrm_add_amount  BIGINT,
            bfefrmtrm_nm       VARCHAR(100),
            bfefrmtrm_amount   BIGINT,
            ord                INTEGER,
            currency           VARCHAR(10),
            PRIMARY KEY (corp_code, bsns_year, report_code, fs_div, sj_div, account_id)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_kr_financial_position_symbol ON kr_financial_position(symbol);")

    # kr_audit — INSERT at kr/dart.py:1520
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_audit (
            stock_name              VARCHAR(200),
            symbol                  VARCHAR(10),
            rcept_no                VARCHAR(30) NOT NULL,
            corp_cls                VARCHAR(10),
            corp_code               VARCHAR(20) NOT NULL,
            corp_name               VARCHAR(200),
            bsns_year               VARCHAR(10) NOT NULL,
            adtor                   VARCHAR(200),
            adt_opinion             TEXT,
            adt_reprt_spcmnt_matter TEXT,
            emphs_matter            TEXT,
            core_adt_matter         TEXT,
            stlm_dt                 DATE,
            PRIMARY KEY (corp_code, bsns_year, rcept_no)
        );
    """)

    # kr_dividends — INSERT at kr/dart.py:1924
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_dividends (
            symbol     VARCHAR(10),
            rcept_no   VARCHAR(30) NOT NULL,
            corp_cls   VARCHAR(10),
            corp_code  VARCHAR(20) NOT NULL,
            corp_name  VARCHAR(200),
            se         VARCHAR(200) NOT NULL,
            stock_knd  VARCHAR(50) NOT NULL DEFAULT '',
            thstrm     NUMERIC(18,4),
            frmtrm     NUMERIC(18,4),
            lwfr       NUMERIC(18,4),
            stlm_dt    DATE,
            PRIMARY KEY (corp_code, rcept_no, se, stock_knd)
        );
    """)

    # kr_largest_shareholder — INSERT at kr/dart.py:2345
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_largest_shareholder (
            symbol                       VARCHAR(10),
            rcept_no                     VARCHAR(30) NOT NULL,
            corp_cls                     VARCHAR(10),
            corp_code                    VARCHAR(20) NOT NULL,
            corp_name                    VARCHAR(200),
            nm                           VARCHAR(200) NOT NULL,
            relate                       VARCHAR(200) NOT NULL DEFAULT '',
            stock_knd                    VARCHAR(50),
            bsis_posesn_stock_co         BIGINT,
            bsis_posesn_stock_qota_rt    NUMERIC(8,4),
            trmend_posesn_stock_co       BIGINT,
            trmend_posesn_stock_qota_rt  NUMERIC(8,4),
            rm                           TEXT,
            stlm_dt                      DATE,
            PRIMARY KEY (corp_code, rcept_no, nm, relate)
        );
    """)

    # kr_stockacquisitiondisposal — INSERT at kr/dart.py:1119
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_stockacquisitiondisposal (
            symbol            VARCHAR(10),
            rcept_no          VARCHAR(30) NOT NULL,
            corp_cls          VARCHAR(10),
            corp_code         VARCHAR(20) NOT NULL,
            corp_name         VARCHAR(200),
            acqs_mth1         VARCHAR(200) NOT NULL DEFAULT '',
            acqs_mth2         VARCHAR(200) NOT NULL DEFAULT '',
            acqs_mth3         VARCHAR(200) NOT NULL DEFAULT '',
            stock_knd         VARCHAR(50) NOT NULL DEFAULT '',
            bsis_qy           BIGINT,
            change_qy_acqs    BIGINT,
            change_qy_dsps    BIGINT,
            change_qy_incnr   BIGINT,
            trmend_qy         BIGINT,
            rm                TEXT,
            stlm_dt           DATE,
            PRIMARY KEY (corp_code, rcept_no, acqs_mth1, acqs_mth2, acqs_mth3, stock_knd)
        );
    """)

    # kr_executive — INSERT at kr/dart.py:2757
    op.execute("""
        CREATE TABLE IF NOT EXISTS kr_executive (
            symbol                VARCHAR(10),
            rcept_no              VARCHAR(30) NOT NULL,
            corp_cls              VARCHAR(10),
            corp_code             VARCHAR(20) NOT NULL,
            corp_name             VARCHAR(200),
            nm                    VARCHAR(200) NOT NULL,
            sexdstn               VARCHAR(10),
            birth_ym              VARCHAR(10),
            ofcps                 VARCHAR(200),
            rgist_exctv_at        VARCHAR(10),
            fte_at                VARCHAR(10),
            chrg_job              VARCHAR(500),
            main_career           TEXT,
            mxmm_shrholdr_relate  VARCHAR(200),
            hffc_pd               VARCHAR(100),
            tenure_end_on         VARCHAR(50),
            stlm_dt               DATE,
            PRIMARY KEY (corp_code, rcept_no, nm)
        );
    """)

    # ==================================================================
    # Common / cross-market tables
    # ==================================================================

    # trading_calendar — referenced by utils/schedule_helper.py:43
    op.execute("""
        CREATE TABLE IF NOT EXISTS trading_calendar (
            date              DATE PRIMARY KEY,
            day_of_week       INTEGER,
            is_kr_holiday     BOOLEAN NOT NULL DEFAULT FALSE,
            kr_holiday_name   TEXT,
            is_us_holiday     BOOLEAN NOT NULL DEFAULT FALSE,
            us_holiday_name   TEXT,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)


def downgrade() -> None:
    # Drop in roughly reverse order to satisfy any future FK refs.
    op.execute("DROP TABLE IF EXISTS trading_calendar;")
    op.execute("DROP TABLE IF EXISTS kr_executive;")
    op.execute("DROP TABLE IF EXISTS kr_stockacquisitiondisposal;")
    op.execute("DROP TABLE IF EXISTS kr_largest_shareholder;")
    op.execute("DROP TABLE IF EXISTS kr_dividends;")
    op.execute("DROP TABLE IF EXISTS kr_audit;")
    op.execute("DROP TABLE IF EXISTS kr_financial_position;")
    op.execute("DROP TABLE IF EXISTS dart_company_info;")
    op.execute("DROP TABLE IF EXISTS exchange_rate;")
    op.execute("DROP TABLE IF EXISTS bok_economic_indicators;")
    op.execute("DROP TABLE IF EXISTS kr_research_reports;")
    op.execute("DROP TABLE IF EXISTS kr_indicators;")
    op.execute("DROP TABLE IF EXISTS kr_benchmark_index;")
    op.execute("DROP TABLE IF EXISTS kr_stock_detail;")
    op.execute("DROP TABLE IF EXISTS kr_stock_basic;")
    op.execute("DROP TABLE IF EXISTS kr_foreign_ownership;")
    op.execute("DROP TABLE IF EXISTS kr_blocktrades;")
    op.execute("DROP TABLE IF EXISTS kr_program_daily_trading;")
    op.execute("DROP TABLE IF EXISTS kr_individual_investor_daily_trading;")
    op.execute("DROP TABLE IF EXISTS kr_investor_daily_trading;")
    op.execute("DROP TABLE IF EXISTS kr_intraday_total;")
    op.execute("DROP TABLE IF EXISTS kr_intraday_detail;")
    op.execute("DROP TABLE IF EXISTS kr_intraday;")
    op.execute("DROP TABLE IF EXISTS us_ipo_calendar;")
    op.execute("DROP TABLE IF EXISTS us_unemployment_rate;")
    op.execute("DROP TABLE IF EXISTS us_cpi;")
    op.execute("DROP TABLE IF EXISTS us_treasury_yield;")
    op.execute("DROP TABLE IF EXISTS us_fed_funds_rate;")
    op.execute("DROP TABLE IF EXISTS us_splits;")
    op.execute("DROP TABLE IF EXISTS us_dividends;")
    op.execute("DROP TABLE IF EXISTS us_earnings_estimates;")
    op.execute("DROP TABLE IF EXISTS us_cash_flow;")
    op.execute("DROP TABLE IF EXISTS us_balance_sheet;")
    op.execute("DROP TABLE IF EXISTS us_income_statement;")
    op.execute("DROP TABLE IF EXISTS us_monthly;")
    op.execute("DROP TABLE IF EXISTS us_weekly;")
    op.execute("DROP TABLE IF EXISTS us_insider_transactions;")
    op.execute("DROP TABLE IF EXISTS us_earnings_calendar;")
    for tname in (
        "us_pmi",
        "us_gdp",
        "us_fed_rrp",
        "us_vix",
        "us_credit_spread",
        "us_dollar_index",
    ):
        op.execute(f"DROP TABLE IF EXISTS {tname};")
    op.execute("DROP TABLE IF EXISTS us_move_index;")
    op.execute("DROP TABLE IF EXISTS us_option_daily_summary;")
    op.execute("DROP TABLE IF EXISTS us_option;")
    op.execute("DROP TABLE IF EXISTS us_vwap;")
    op.execute("DROP TABLE IF EXISTS us_atr;")
    op.execute("DROP TABLE IF EXISTS us_obv;")
    op.execute("DROP TABLE IF EXISTS us_cci;")
    op.execute("DROP TABLE IF EXISTS us_aroon;")
    op.execute("DROP TABLE IF EXISTS us_wma;")
    op.execute("DROP TABLE IF EXISTS us_adx;")
    op.execute("DROP TABLE IF EXISTS us_ema;")
    op.execute("DROP TABLE IF EXISTS us_sma;")
    op.execute("DROP TABLE IF EXISTS us_roc;")
    op.execute("DROP TABLE IF EXISTS us_mfi;")
    op.execute("DROP TABLE IF EXISTS us_stoch;")
    op.execute("DROP TABLE IF EXISTS us_bbands;")
    op.execute("DROP TABLE IF EXISTS us_macd;")
    op.execute("DROP TABLE IF EXISTS us_rsi;")
    op.execute("DROP TABLE IF EXISTS us_indicators;")
    op.execute("DROP TABLE IF EXISTS market_index;")
    op.execute("DROP TABLE IF EXISTS us_daily_etf;")
    op.execute("DROP TABLE IF EXISTS us_news;")
    op.execute("DROP TABLE IF EXISTS us_vwap_base;")
    op.execute("DROP TABLE IF EXISTS us_daily;")

    # Note: us_symbol ALTER columns are intentionally NOT reverted to avoid
    # destroying production data; downgrading 0002 keeps the extended columns.
