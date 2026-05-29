# -*- coding: utf-8 -*-
import asyncio
import asyncpg
import logging
import os
import sys
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv
import aiohttp
from asyncio import Queue
from pathlib import Path

# Add parent directory to path for importing collection_logger
sys.path.insert(0, str(Path(__file__).parent.parent))
from collection_logger import CollectionLogger

# Load environment variables
load_dotenv()

# Common User-Agent header for API requests
USER_AGENT_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# AlphaVantage throttle / retry. A symbol must not be silently dropped on a
# transient hiccup (network error, 5xx, or a rate-limit "Note"/"Information"
# response). On rate-limit we back off hard (adaptive throttle); on transient
# network/5xx errors we use exponential backoff. Invalid-symbol / no-data
# responses are NOT retried — they're terminal and correct.
AV_MAX_RETRIES = 5
AV_RETRY_BASE_DELAY = 5.0    # seconds; multiplied by attempt for backoff
AV_RATELIMIT_DELAY = 20.0    # seconds to wait when AV signals a rate limit

# Setup logging - Use Railway Volume (/app/log) if available
if os.getenv('RAILWAY_PROJECT_ID'):
    log_dir = Path('/app/log')
else:
    log_dir = Path(__file__).parent.parent / 'log'

log_dir.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'us_option_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class USOptionCollector:
    """US Options Data Collector from Alpha Vantage"""

    def __init__(self, api_key: str, database_url: str, call_interval: float,
                 target_date: date = None, summary_only: bool = False):
        self.api_key = api_key
        if database_url.startswith('postgresql+asyncpg://'):
            database_url = database_url.replace('postgresql+asyncpg://', 'postgresql://')
        self.database_url = database_url
        self.target_date = target_date if target_date else self.get_latest_business_day()
        self.call_interval = call_interval
        # summary_only: aggregate each symbol's option chain in memory and write
        # ONLY us_option_daily_summary, never persisting the raw chain to
        # us_option. Eliminates ~150k raw rows/day (31 GB/yr), the COPY+UPSERT
        # into the partitioned table, and the post-hoc delete. quant reads only
        # the summary, so nothing downstream needs the raw contracts.
        self.summary_only = summary_only

        # Setup collection logger path based on environment
        if os.getenv('RAILWAY_PROJECT_ID'):
            log_file_path = '/app/log/us_option_collected.json'
        else:
            log_file_path = str(Path(__file__).parent.parent / 'log' / 'us_option_collected.json')

        self.collection_logger = CollectionLogger(log_file_path)
        self.base_url = "https://www.alphavantage.co/query"
        self.pool = None
        self.session = None

    def get_date_string(self) -> str:
        """Get target date as string"""
        return self.target_date.strftime('%Y-%m-%d')

    def get_latest_business_day(self) -> date:
        """Get the latest business day (excluding weekends)"""
        today = date.today()
        # Check up to 3 days back to find a weekday
        for days_back in range(4):
            check_date = today - timedelta(days=days_back)
            if check_date.weekday() < 5:  # Monday to Friday
                return check_date
        # Fallback: return today
        return today

    async def init_pool(self):
        """Initialize connection pool.

        summary_only is a single sequential API worker that touches the DB only
        for the spot-price fetch and batched upserts (≤2 concurrent acquires).
        A large pool there was fatal: the options endpoint builds a fresh
        collector per date, and min=10/max=50 pools stacked across the run
        exhausted Postgres max_connections (TooManyConnectionsError → 500s,
        dropping whole dates). Keep summary pools tiny; raw path keeps the
        larger pool for its concurrent COPY workers.
        """
        if self.summary_only:
            min_size, max_size = 1, 5
        else:
            min_size, max_size = 10, 50
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=min_size,
            max_size=max_size,
            command_timeout=120,
            max_queries=50000,
            max_cached_statement_lifetime=0,
            max_cacheable_statement_size=0
        )
        self.session = aiohttp.ClientSession(headers=USER_AGENT_HEADERS)
        logger.info(
            f"[US_OPTION] Database connection pool initialized "
            f"(min={min_size}, max={max_size}, summary_only={self.summary_only})")

    async def close_pool(self):
        """Release the pool, forcibly terminating if graceful close stalls.

        The options endpoint builds one collector (one pool) per date. If
        pool.close() hangs on a connection that wasn't released, the server
        connections leak; after ~10 dates that exhausts max_connections and
        every later date 500s. wait_for + terminate guarantees the slots are
        freed before the next date's pool is created.
        """
        if self.pool:
            try:
                await asyncio.wait_for(self.pool.close(), timeout=10)
            except Exception as e:
                logger.warning(
                    f"[US_OPTION] pool.close() stalled/failed ({e}); terminating")
                self.pool.terminate()
            self.pool = None
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("[US_OPTION] Database connection pool closed")

    async def get_connection(self):
        if self.pool:
            return await self.pool.acquire()
        else:
            return await asyncpg.connect(self.database_url)

    async def get_option_data(self, symbol: str) -> Optional[Dict]:
        """Fetch historical options data from Alpha Vantage with throttle+retry.

        Retries transient failures (rate-limit Note/Information, HTTP 429/5xx,
        network timeouts) with backoff so a symbol is never dropped on a
        temporary hiccup. Terminal cases (invalid symbol 'Error Message',
        unexpected payload) return None immediately without retry.
        """
        params = {
            'function': 'HISTORICAL_OPTIONS',
            'symbol': symbol,
            'apikey': self.api_key,
        }
        if self.target_date:
            params['date'] = self.target_date.strftime('%Y-%m-%d')

        last_err = None
        for attempt in range(1, AV_MAX_RETRIES + 1):
            try:
                if self.target_date:
                    logger.info(f"[US_OPTION] Calling API for {symbol} on {params['date']} (attempt {attempt})...")
                else:
                    logger.info(f"[US_OPTION] Calling API for {symbol} (attempt {attempt})...")

                async with self.session.get(
                    self.base_url, params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    logger.info(f"[US_OPTION] Received response for {symbol}, status: {response.status}")
                    if response.status == 200:
                        data = await response.json()

                        if 'Error Message' in data:
                            # Invalid symbol / no options listed — terminal.
                            logger.error(f"[US_OPTION] API error for {symbol}: {data['Error Message']}")
                            return None
                        if 'Note' in data or 'Information' in data:
                            # Rate limit — adaptive throttle, then retry.
                            msg = data.get('Note') or data.get('Information')
                            logger.warning(
                                f"[US_OPTION] rate limit for {symbol} "
                                f"(attempt {attempt}/{AV_MAX_RETRIES}): {msg}; "
                                f"throttling {AV_RATELIMIT_DELAY}s")
                            last_err = f"rate-limit: {msg}"
                            await asyncio.sleep(AV_RATELIMIT_DELAY)
                            continue
                        if 'data' in data:
                            return data
                        logger.warning(f"[US_OPTION] Unexpected response format for {symbol}: {list(data.keys())}")
                        return None

                    if response.status == 429 or response.status >= 500:
                        # Transient server-side — exponential backoff retry.
                        delay = AV_RETRY_BASE_DELAY * attempt
                        last_err = f"HTTP {response.status}"
                        logger.warning(
                            f"[US_OPTION] {symbol} HTTP {response.status} "
                            f"(attempt {attempt}/{AV_MAX_RETRIES}); retry in {delay}s")
                        await asyncio.sleep(delay)
                        continue

                    logger.error(f"[US_OPTION] API request failed for {symbol}: Status {response.status} (no retry)")
                    return None

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                delay = AV_RETRY_BASE_DELAY * attempt
                last_err = repr(e)
                logger.warning(
                    f"[US_OPTION] {symbol} network error "
                    f"(attempt {attempt}/{AV_MAX_RETRIES}): {e}; retry in {delay}s")
                await asyncio.sleep(delay)
                continue
            except Exception as e:
                logger.error(f"[US_OPTION] Error fetching data for {symbol}: {e}")
                return None

        logger.error(
            f"[US_OPTION] {symbol}: exhausted {AV_MAX_RETRIES} retries "
            f"(last error: {last_err})")
        return None

    def safe_decimal(self, value: str, default=None) -> Optional[float]:
        """Safely convert string to decimal"""
        if not value or value.strip() in ['', 'N/A', '-', 'null', 'None']:
            return default
        try:
            return float(value.strip())
        except (ValueError, AttributeError):
            return default

    def safe_int(self, value: str, default=None) -> Optional[int]:
        """Safely convert string to integer"""
        if not value or value.strip() in ['', 'N/A', '-', 'null', 'None']:
            return default
        try:
            return int(value.strip())
        except (ValueError, AttributeError):
            return default

    def transform_option_data(self, data: Dict, symbol: str) -> List[Dict]:
        """Transform option data to match database schema"""
        transformed = []

        if 'data' not in data:
            logger.warning(f"[US_OPTION] No 'data' field in response for {symbol}")
            return transformed

        options_data = data['data']
        logger.info(f"[US_OPTION] Processing {len(options_data)} option contracts for {symbol}")

        for option in options_data:
            try:
                # Parse dates
                expiration_date = datetime.strptime(option['expiration'], '%Y-%m-%d').date()
                data_date = datetime.strptime(option['date'], '%Y-%m-%d').date()

                # Note: We don't filter by target_date because API returns the most recent trading day
                # which might not match the requested date (e.g., weekends, holidays, future dates)

                transformed.append({
                    'contract_id': option.get('contractID', ''),
                    'symbol': option.get('symbol', symbol),
                    'expiration': expiration_date,
                    'strike': self.safe_decimal(option.get('strike')),
                    'type': option.get('type', ''),
                    'last': self.safe_decimal(option.get('last')),
                    'mark': self.safe_decimal(option.get('mark')),
                    'bid': self.safe_decimal(option.get('bid')),
                    'bid_size': self.safe_int(option.get('bid_size')),
                    'ask': self.safe_decimal(option.get('ask')),
                    'ask_size': self.safe_int(option.get('ask_size')),
                    'volume': self.safe_int(option.get('volume')),
                    'open_interest': self.safe_int(option.get('open_interest')),
                    'date': data_date,
                    'implied_volatility': self.safe_decimal(option.get('implied_volatility')),
                    'delta': self.safe_decimal(option.get('delta')),
                    'gamma': self.safe_decimal(option.get('gamma')),
                    'theta': self.safe_decimal(option.get('theta')),
                    'vega': self.safe_decimal(option.get('vega')),
                    'rho': self.safe_decimal(option.get('rho')),
                })
            except Exception as e:
                logger.error(f"[US_OPTION] Error transforming option for {symbol}: {e}, Data: {option}")
                continue

        return transformed

    async def save_option_data(self, data: List[Dict]) -> int:
        """Save option data using executemany"""
        if not data:
            return 0

        conn = await self.get_connection()
        try:
            async with conn.transaction():
                query = """
                    INSERT INTO us_option (
                        contract_id, symbol, expiration, strike, type,
                        last, mark, bid, bid_size, ask, ask_size,
                        volume, open_interest, date,
                        implied_volatility, delta, gamma, theta, vega, rho
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
                    ON CONFLICT (contract_id, date) DO UPDATE SET
                        symbol = EXCLUDED.symbol,
                        expiration = EXCLUDED.expiration,
                        strike = EXCLUDED.strike,
                        type = EXCLUDED.type,
                        last = EXCLUDED.last,
                        mark = EXCLUDED.mark,
                        bid = EXCLUDED.bid,
                        bid_size = EXCLUDED.bid_size,
                        ask = EXCLUDED.ask,
                        ask_size = EXCLUDED.ask_size,
                        volume = EXCLUDED.volume,
                        open_interest = EXCLUDED.open_interest,
                        implied_volatility = EXCLUDED.implied_volatility,
                        delta = EXCLUDED.delta,
                        gamma = EXCLUDED.gamma,
                        theta = EXCLUDED.theta,
                        vega = EXCLUDED.vega,
                        rho = EXCLUDED.rho
                """

                batch_values = [
                    (
                        record['contract_id'],
                        record['symbol'],
                        record['expiration'],
                        record['strike'],
                        record['type'],
                        record['last'],
                        record['mark'],
                        record['bid'],
                        record['bid_size'],
                        record['ask'],
                        record['ask_size'],
                        record['volume'],
                        record['open_interest'],
                        record['date'],
                        record['implied_volatility'],
                        record['delta'],
                        record['gamma'],
                        record['theta'],
                        record['vega'],
                        record['rho']
                    )
                    for record in data
                ]
                await conn.executemany(query, batch_values)

            if self.pool:
                await self.pool.release(conn)
            else:
                await conn.close()
            return len(data)
        except Exception as e:
            logger.error(f"[US_OPTION] Error in save_option_data: {e}")
            if self.pool:
                await self.pool.release(conn)
            else:
                await conn.close()
            return 0

    async def save_option_data_optimized(self, data: List[Dict]) -> int:
        """Save option data using COPY command - OPTIMIZED"""
        if not data:
            return 0

        conn = await self.get_connection()
        try:
            async with conn.transaction():
                # Transaction optimization
                await conn.execute('SET LOCAL synchronous_commit = OFF')
                await conn.execute('SET LOCAL work_mem = "256MB"')

                # Create temp table
                await conn.execute('''
                    CREATE TEMP TABLE temp_us_option (
                        contract_id VARCHAR(30),
                        symbol VARCHAR(10),
                        expiration DATE,
                        strike NUMERIC(10, 2),
                        type VARCHAR(10),
                        last NUMERIC(10, 2),
                        mark NUMERIC(10, 2),
                        bid NUMERIC(10, 2),
                        bid_size INTEGER,
                        ask NUMERIC(10, 2),
                        ask_size INTEGER,
                        volume INTEGER,
                        open_interest INTEGER,
                        date DATE,
                        implied_volatility NUMERIC(10, 5),
                        delta NUMERIC(10, 5),
                        gamma NUMERIC(10, 5),
                        theta NUMERIC(10, 5),
                        vega NUMERIC(10, 5),
                        rho NUMERIC(10, 5)
                    ) ON COMMIT DROP
                ''')

                # Prepare data for COPY
                rows = [[
                    record['contract_id'],
                    record['symbol'],
                    record['expiration'],
                    record['strike'],
                    record['type'],
                    record['last'],
                    record['mark'],
                    record['bid'],
                    record['bid_size'],
                    record['ask'],
                    record['ask_size'],
                    record['volume'],
                    record['open_interest'],
                    record['date'],
                    record['implied_volatility'],
                    record['delta'],
                    record['gamma'],
                    record['theta'],
                    record['vega'],
                    record['rho']
                ] for record in data]

                # COPY to temp table
                await conn.copy_records_to_table(
                    'temp_us_option', records=rows,
                    columns=[
                        'contract_id', 'symbol', 'expiration', 'strike', 'type',
                        'last', 'mark', 'bid', 'bid_size', 'ask', 'ask_size',
                        'volume', 'open_interest', 'date',
                        'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho'
                    ]
                )

                # Upsert from temp to main
                await conn.execute('''
                    INSERT INTO us_option (
                        contract_id, symbol, expiration, strike, type,
                        last, mark, bid, bid_size, ask, ask_size,
                        volume, open_interest, date,
                        implied_volatility, delta, gamma, theta, vega, rho
                    )
                    SELECT * FROM temp_us_option
                    ON CONFLICT (contract_id, date) DO UPDATE SET
                        symbol = EXCLUDED.symbol,
                        expiration = EXCLUDED.expiration,
                        strike = EXCLUDED.strike,
                        type = EXCLUDED.type,
                        last = EXCLUDED.last,
                        mark = EXCLUDED.mark,
                        bid = EXCLUDED.bid,
                        bid_size = EXCLUDED.bid_size,
                        ask = EXCLUDED.ask,
                        ask_size = EXCLUDED.ask_size,
                        volume = EXCLUDED.volume,
                        open_interest = EXCLUDED.open_interest,
                        implied_volatility = EXCLUDED.implied_volatility,
                        delta = EXCLUDED.delta,
                        gamma = EXCLUDED.gamma,
                        theta = EXCLUDED.theta,
                        vega = EXCLUDED.vega,
                        rho = EXCLUDED.rho
                ''')

            if self.pool:
                await self.pool.release(conn)
            else:
                await conn.close()
            return len(data)
        except Exception as e:
            logger.error(f"[US_OPTION] Error in save_option_data_optimized: {e}")
            if self.pool:
                await self.pool.release(conn)
            else:
                await conn.close()
            logger.info("[US_OPTION] Falling back to regular save")
            return await self.save_option_data(data)

    async def ensure_daily_partition_exists(self, target_date: date):
        """Ensure daily partition exists for target date"""
        conn = await self.get_connection()
        try:
            partition_name = f"us_option_{target_date.strftime('%Y_%m_%d')}"
            next_date = target_date + timedelta(days=1)

            # Check if partition already exists
            check_query = """
                SELECT EXISTS (
                    SELECT 1 FROM pg_tables
                    WHERE tablename = $1 AND schemaname = 'public'
                )
            """
            exists = await conn.fetchval(check_query, partition_name)

            if exists:
                logger.info(f"[US_OPTION] Partition {partition_name} already exists")
            else:
                # Create partition
                create_query = f"""
                    CREATE TABLE {partition_name} PARTITION OF us_option
                    FOR VALUES FROM ('{target_date}') TO ('{next_date}')
                """
                await conn.execute(create_query)

                # Create unique index
                unique_idx_query = f"""
                    CREATE UNIQUE INDEX {partition_name}_unique
                    ON {partition_name} (contract_id, date)
                """
                await conn.execute(unique_idx_query)

                # Create symbol index
                symbol_idx_query = f"""
                    CREATE INDEX {partition_name}_symbol_idx
                    ON {partition_name} (symbol)
                """
                await conn.execute(symbol_idx_query)

                # Create expiration index
                expiration_idx_query = f"""
                    CREATE INDEX {partition_name}_expiration_idx
                    ON {partition_name} (expiration)
                """
                await conn.execute(expiration_idx_query)

                logger.info(f"[US_OPTION] Created partition {partition_name} for date {target_date}")

        except Exception as e:
            logger.warning(f"[US_OPTION] Error creating partition for {target_date}: {e}")
        finally:
            if self.pool:
                await self.pool.release(conn)
            else:
                await conn.close()

    async def get_active_symbols(self) -> List[str]:
        """Get US Option collection symbols from predefined list"""
        # US Option Collection Symbols - Total: 549 symbols
        # Source: US_OPTION_SYMBOLS_COLLECTION_LIST.csv
        # Last updated: 2025-10-28
        # Breakdown: ETF 49 + Individual stocks (TOP100 + SP500) 500
        US_OPTION_SYMBOLS = [
            'A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM',
            'ADP', 'ADSK', 'AEE', 'AEM', 'AEP', 'AER', 'AFRM', 'AFL', 'AGG', 'AIG',
            'AJG', 'ALAB', 'ALNY', 'AMD', 'AME', 'AMGN', 'AMP', 'AMRZ', 'AMT', 'AMZN',
            'AMAT', 'ANET', 'AON', 'APD', 'APH', 'APO', 'APP', 'ARES', 'ARKK', 'ASTS',
            'ATO', 'AVGO', 'AVB', 'AWK', 'AXP', 'AXON', 'AZO', 'BA', 'BAC', 'BBVA',
            'BC', 'BCE', 'BCS', 'BDX', 'BE', 'BIIB', 'BK', 'BKR', 'BKNG', 'BLK',
            'BMO', 'BMY', 'BN', 'BNS', 'BP', 'BR', 'BRO', 'BSX', 'BTI', 'BX',
            'C', 'CAH', 'CARR', 'CAT', 'CB', 'CBRE', 'CCJ', 'CCEP', 'CCI', 'CCL',
            'CDNS', 'CDW', 'CEG', 'CFG', 'CG', 'CHD', 'CHTR', 'CI', 'CINF', 'CL',
            'CLS', 'CM', 'CME', 'CMG', 'CMI', 'CMCSA', 'CMS', 'CNI', 'CNP', 'CNQ',
            'COF', 'COIN', 'COP', 'COR', 'COST', 'CPAY', 'CPNG', 'CPRT', 'CRCL', 'CRDO',
            'CRH', 'CRM', 'CRWD', 'CRWV', 'CSGP', 'CSX', 'CTAS', 'CTSH', 'CTVA', 'CVE',
            'CVS', 'CVX', 'CVNA', 'CW', 'D', 'DAL', 'DASH', 'DBA', 'DB', 'DD',
            'DDOG', 'DE', 'DELL', 'DEO', 'DG', 'DHI', 'DHR', 'DIA', 'DIS', 'DLR',
            'DOV', 'DRI', 'DTE', 'DUK', 'DVN', 'DXCM', 'E', 'EA', 'EBAY', 'ECL',
            'ED', 'EEM', 'EFA', 'EFX', 'EIX', 'EL', 'ELV', 'EME', 'EMR', 'ENB',
            'EOG', 'EPD', 'EQIX', 'EQT', 'ES', 'ETR', 'EW', 'EWJ', 'EXC', 'EXE',
            'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FCNCA', 'FDX', 'FE', 'FERG', 'FI',
            'FICO', 'FIG', 'FIS', 'FITB', 'FIX', 'FLUT', 'FMX', 'FNV', 'FOX', 'FOXA',
            'FSLR', 'FTS', 'FTNT', 'FWONA', 'FWONK', 'FXI', 'GD', 'GE', 'GEHC', 'GEV',
            'GIB', 'GILD', 'GIS', 'GLD', 'GLW', 'GM', 'GOOG', 'GOOGL', 'GPN', 'GRAB',
            'GRMN', 'GS', 'GWRE', 'GWW', 'HAL', 'HBAN', 'HCA', 'HD', 'HDB', 'HEI',
            'HIG', 'HLT', 'HMC', 'HON', 'HOOD', 'HPE', 'HPQ', 'HSBC', 'HSY', 'HUBB',
            'HUBS', 'HUM', 'HWM', 'HYG', 'IBB', 'IBKR', 'IBM', 'IBN', 'ICE', 'IDXX',
            'IGV', 'IHI', 'ING', 'INSM', 'INTC', 'INTU', 'IOT', 'IP', 'IQV', 'IR',
            'IRM', 'ISRG', 'ITW', 'IWD', 'IWF', 'IWM', 'JBL', 'JNJ', 'JPM', 'K',
            'KDP', 'KEYS', 'KGC', 'KHC', 'KLAC', 'KMB', 'KMI', 'KO', 'KR', 'KRE',
            'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN', 'LLY', 'LMT', 'LNG',
            'LOW', 'LPLA', 'LQD', 'LRCX', 'LULU', 'LVS', 'LYV', 'MA', 'MAR', 'MCD',
            'MCHP', 'MCK', 'MCO', 'MDB', 'MDLZ', 'MDY', 'MELI', 'MET', 'META', 'MFC',
            'MKL', 'MMC', 'MMM', 'MNST', 'MO', 'MPC', 'MPWR', 'MRK', 'MS', 'MSCI',
            'MSFT', 'MSI', 'MSTR', 'MTB', 'MTD', 'MU', 'MUFG', 'NEE', 'NET', 'NFLX',
            'NKE', 'NOC', 'NOW', 'NDAQ', 'NTRA', 'NTRS', 'NTAP', 'NTR', 'NU', 'NUE',
            'NVDA', 'NVO', 'NVR', 'NVS', 'NXPI', 'O', 'ODFL', 'OKE', 'ON', 'ORCL',
            'ORLY', 'OTIS', 'OWL', 'OXY', 'PANW', 'PAYX', 'PBA', 'PCAR', 'PCG', 'PEG',
            'PEP', 'PFE', 'PG', 'PGR', 'PH', 'PHM', 'PINS', 'PLD', 'PLTR', 'PM',
            'PNC', 'PODD', 'PPG', 'PPL', 'PRU', 'PSA', 'PSTG', 'PSX', 'PTC', 'PTR',
            'PUK', 'PWR', 'PYPL', 'QQQ', 'QCOM', 'QSR', 'RACE', 'RBLX', 'RCI', 'RCL',
            'RDDT', 'REGN', 'RF', 'RIO', 'RKLB', 'RKT', 'RMD', 'ROK', 'ROL', 'ROP',
            'ROST', 'RPRX', 'RSG', 'RTX', 'RY', 'SBAC', 'SBUX', 'SCCO', 'SCHW', 'SHW',
            'SHY', 'SLB', 'SLF', 'SLV', 'SMCI', 'SMH', 'SNOW', 'SNPS', 'SO', 'SOFI',
            'SOXX', 'SPGI', 'SPY', 'SQQQ', 'SRE', 'STE', 'STLA', 'STLD', 'STM', 'STT',
            'STX', 'STZ', 'SU', 'SW', 'SBAC', 'SYF', 'SYK', 'SYM', 'SYY', 'T',
            'TAK', 'TDG', 'TD', 'TDY', 'TEAM', 'TECK', 'TEF', 'TEL', 'TER', 'TFC',
            'TGT', 'TJX', 'TLT', 'TM', 'TMO', 'TMUS', 'TOST', 'TPG', 'TPL', 'TPR',
            'TQQQ', 'TRGP', 'TRI', 'TRP', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TT', 'TTD',
            'TTWO', 'TU', 'TW', 'TXN', 'TYL', 'U', 'UAL', 'UBER', 'UBS', 'UI',
            'UL', 'ULTA', 'UNG', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'USO', 'UVXY',
            'V', 'VEEV', 'VG', 'VICI', 'VIK', 'VLO', 'VLTO', 'VMC', 'VRT', 'VRSN',
            'VRSK', 'VRTX', 'VST', 'VTI', 'VTR', 'VTV', 'VUG', 'VWO', 'VXX', 'VZ',
            'W', 'WAB', 'WAT', 'WBD', 'WCN', 'WDC', 'WDAY', 'WEC', 'WELL', 'WFC',
            'WIT', 'WM', 'WMB', 'WMT', 'WPM', 'WRB', 'WSM', 'WST', 'WTW', 'XBI',
            'XEL', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU',
            'XLV', 'XLY', 'XOM', 'XYL', 'XYZ', 'YUM', 'ZBH', 'ZM', 'ZS', 'ZTS',
        ]

        logger.info(f"[US_OPTION] Loaded {len(US_OPTION_SYMBOLS)} predefined symbols for option collection")

        # === Dynamic top-N mode (used by 2-pass orchestrator) ===========
        # Two modes are supported, both opt-in via env vars:
        #
        # 1. **Per-date** (legacy): set ``US_OPTION_DYNAMIC_TOP_N`` and
        #    ``US_OPTION_TARGET_GRADE_DATE``. Returns the top-N grades on the
        #    target date only — different symbol set per date. Causes 252-day
        #    history gaps for boundary symbols that bounce in/out of top-N.
        #
        # 2. **Range-union** (recommended for backtest backfill): also set
        #    ``US_OPTION_DYNAMIC_START_DATE`` and ``US_OPTION_DYNAMIC_END_DATE``.
        #    Returns the union of per-date top-N symbols across the whole
        #    range — same symbol set for every date in the backfill. Ensures
        #    agent_metrics 252-day IV percentile and volatility_adjustment
        #    have continuous history for every union symbol.
        #
        # Falls through to whitelist on any failure.
        dyn_top_n  = os.getenv("US_OPTION_DYNAMIC_TOP_N")
        dyn_date   = os.getenv("US_OPTION_TARGET_GRADE_DATE")
        dyn_start  = os.getenv("US_OPTION_DYNAMIC_START_DATE")
        dyn_end    = os.getenv("US_OPTION_DYNAMIC_END_DATE")
        if dyn_top_n and dyn_date:
            try:
                top_n_int = int(dyn_top_n)
                # asyncpg's date codec needs date objects, not env-var strings,
                # for the $n::date params below.
                dyn_date_d = datetime.strptime(dyn_date, "%Y-%m-%d").date()
                dyn_start_d = (
                    datetime.strptime(dyn_start, "%Y-%m-%d").date()
                    if dyn_start else None
                )
                dyn_end_d = (
                    datetime.strptime(dyn_end, "%Y-%m-%d").date()
                    if dyn_end else None
                )
                conn = await self.get_connection()
                try:
                    if dyn_start_d and dyn_end_d:
                        rows = await conn.fetch(
                            """
                            SELECT DISTINCT symbol FROM (
                              SELECT symbol, final_score,
                                ROW_NUMBER() OVER (
                                  PARTITION BY date ORDER BY final_score DESC NULLS LAST
                                ) AS rn
                              FROM us_stock_grade
                              WHERE date BETWEEN $1::date AND $2::date
                                AND final_grade IN
                                    ('STRONG_BUY','BUY','NEUTRAL','강력 매수','매수','매수 고려','중립')
                                AND final_score IS NOT NULL
                            ) t WHERE rn <= $3
                            """,
                            dyn_start_d, dyn_end_d, top_n_int,
                        )
                        mode_label = f"UNION top-{top_n_int} over {dyn_start}~{dyn_end}"
                    else:
                        rows = await conn.fetch(
                            """
                            SELECT symbol FROM us_stock_grade
                            WHERE date = $1::date
                              AND final_grade IN
                                  ('STRONG_BUY','BUY','NEUTRAL','강력 매수','매수','매수 고려','중립')
                              AND final_score IS NOT NULL
                            ORDER BY final_score DESC NULLS LAST
                            LIMIT $2
                            """,
                            dyn_date_d, top_n_int,
                        )
                        mode_label = f"per-date top-{top_n_int} for {dyn_date}"
                finally:
                    await conn.close()
                if rows:
                    dyn_symbols = [r["symbol"] for r in rows]
                    logger.info(
                        f"[US_OPTION] DYNAMIC {mode_label}: "
                        f"{len(dyn_symbols)} symbols (overrides whitelist)"
                    )
                    return dyn_symbols
                else:
                    logger.warning(
                        f"[US_OPTION] DYNAMIC mode: no grades found ({mode_label}), "
                        "falling back to whitelist"
                    )
            except Exception as e:
                logger.error(
                    f"[US_OPTION] DYNAMIC mode failed ({e}), falling back to whitelist"
                )

        return US_OPTION_SYMBOLS

    # ----- summary-only path (raw-bypass optimization) ------------------
    @staticmethod
    def _gamma_flip_distance(strike_net_gex: Dict[float, float], spot: float) -> Optional[float]:
        """Nearest gamma-flip strike distance (% from spot).

        Mirrors populate_option_summary SQL: walk strikes ascending, find where
        per-strike net GEX changes sign, linear-interpolate the flip strike, and
        pick the flip whose (current strike − spot) is smallest. Returns
        (flip_strike − spot)/spot × 100.
        """
        pts = sorted((s, g) for s, g in strike_net_gex.items() if g != 0)
        best_dist = None
        best_flip = None
        for i in range(1, len(pts)):
            s0, g0 = pts[i - 1]
            s1, g1 = pts[i]
            sign0 = (g0 > 0) - (g0 < 0)
            sign1 = (g1 > 0) - (g1 < 0)
            if sign0 != 0 and sign1 != 0 and sign0 != sign1:
                flip = s0 + (s1 - s0) * abs(g0) / (abs(g0) + abs(g1))
                dist = abs(s1 - spot)  # SQL: distance_from_spot = ABS(strike - spot)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_flip = flip
        if best_flip is None or spot <= 0:
            return None
        return (best_flip - spot) / spot * 100

    @staticmethod
    def _summarize_contracts(symbol: str, contracts: List[Dict],
                              spot: Optional[float] = None) -> Optional[Dict]:
        """Aggregate one symbol's option chain into a single daily-summary row.

        Pure-Python mirror of populate_option_summary's full aggregation
        (volume / IV / GEX / gamma_flip) — lets us write us_option_daily_summary
        directly without persisting raw contracts. Qualified IV = IV>0 AND
        volume>0 AND open_interest>0. GEX = gamma × OI × 100 × spot (calls minus
        puts); requires spot (us_daily close). gamma_flip_distance from per-strike
        sign reversal.
        """
        if not contracts:
            return None
        actual_date = contracts[0]['date']  # API returns the nearest trading day

        def _f(v):
            return float(v) if v is not None else None

        call_vol = sum((c['volume'] or 0) for c in contracts if c['type'] == 'call')
        put_vol = sum((c['volume'] or 0) for c in contracts if c['type'] == 'put')

        all_iv = [_f(c['implied_volatility']) for c in contracts
                  if c['implied_volatility'] is not None]
        all_iv = [v for v in all_iv if v is not None]

        def _qual_iv(typ):
            return [_f(c['implied_volatility']) for c in contracts
                    if c['type'] == typ
                    and c['implied_volatility'] is not None
                    and _f(c['implied_volatility']) > 0
                    and (c['volume'] or 0) > 0
                    and (c['open_interest'] or 0) > 0]
        call_ivs = _qual_iv('call')
        put_ivs = _qual_iv('put')

        # ---- GEX (gamma exposure) — needs spot price ----
        call_gex = put_gex = None
        net_gex = gex_ratio = gamma_flip_distance = None
        if spot is not None and spot > 0:
            cg = pg = 0.0
            total_oi = 0
            strike_net_gex: Dict[float, float] = {}
            for c in contracts:
                g = c.get('gamma')
                oi = c.get('open_interest')
                if g is None or oi is None or oi <= 0:
                    continue
                gex = _f(g) * oi * 100 * spot
                total_oi += oi
                strike = _f(c.get('strike'))
                if c['type'] == 'call':
                    cg += gex
                    if strike is not None:
                        strike_net_gex[strike] = strike_net_gex.get(strike, 0.0) + gex
                elif c['type'] == 'put':
                    pg += gex
                    if strike is not None:
                        strike_net_gex[strike] = strike_net_gex.get(strike, 0.0) - gex
            call_gex, put_gex = cg, pg
            net_gex = cg - pg
            gex_ratio = (net_gex / (spot * total_oi * 100)) if total_oi > 0 else None
            gamma_flip_distance = USOptionCollector._gamma_flip_distance(strike_net_gex, spot)

        return {
            'symbol': symbol,
            'date': actual_date,
            'total_call_volume': call_vol,
            'total_put_volume': put_vol,
            'avg_implied_volatility': (sum(all_iv) / len(all_iv)) if all_iv else None,
            'min_implied_volatility': min(all_iv) if all_iv else None,
            'max_implied_volatility': max(all_iv) if all_iv else None,
            'avg_call_iv': (sum(call_ivs) / len(call_ivs)) if call_ivs else None,
            'avg_put_iv': (sum(put_ivs) / len(put_ivs)) if put_ivs else None,
            'call_option_count': len(call_ivs),
            'put_option_count': len(put_ivs),
            'call_gex': call_gex,
            'put_gex': put_gex,
            'net_gex': net_gex,
            'gex_ratio': gex_ratio,
            'gamma_flip_distance': gamma_flip_distance,
        }

    async def save_summaries_batch(self, summaries: List[Dict]) -> int:
        """UPSERT a batch of pre-aggregated summary rows into
        us_option_daily_summary (summary_only path, no raw table touched)."""
        if not summaries:
            return 0
        cols = ['symbol', 'date', 'total_call_volume', 'total_put_volume',
                'avg_implied_volatility', 'min_implied_volatility',
                'max_implied_volatility', 'avg_call_iv', 'avg_put_iv',
                'call_option_count', 'put_option_count',
                'call_gex', 'put_gex', 'net_gex', 'gex_ratio', 'gamma_flip_distance']
        records = [tuple(s.get(c) for c in cols) for s in summaries]
        conn = await self.get_connection()
        try:
            async with conn.transaction():
                await conn.execute(
                    "CREATE TEMP TABLE _opt_sum (LIKE us_option_daily_summary "
                    "INCLUDING DEFAULTS) ON COMMIT DROP")
                await conn.copy_records_to_table('_opt_sum', records=records, columns=cols)
                await conn.execute(f"""
                    INSERT INTO us_option_daily_summary ({', '.join(cols)})
                    SELECT {', '.join(cols)} FROM _opt_sum
                    ON CONFLICT (symbol, date) DO UPDATE SET
                        total_call_volume = EXCLUDED.total_call_volume,
                        total_put_volume = EXCLUDED.total_put_volume,
                        avg_implied_volatility = EXCLUDED.avg_implied_volatility,
                        min_implied_volatility = EXCLUDED.min_implied_volatility,
                        max_implied_volatility = EXCLUDED.max_implied_volatility,
                        avg_call_iv = EXCLUDED.avg_call_iv,
                        avg_put_iv = EXCLUDED.avg_put_iv,
                        call_option_count = EXCLUDED.call_option_count,
                        put_option_count = EXCLUDED.put_option_count,
                        call_gex = EXCLUDED.call_gex,
                        put_gex = EXCLUDED.put_gex,
                        net_gex = EXCLUDED.net_gex,
                        gex_ratio = EXCLUDED.gex_ratio,
                        gamma_flip_distance = EXCLUDED.gamma_flip_distance
                """)
        finally:
            if self.pool:
                await self.pool.release(conn)
            else:
                await conn.close()
        return len(records)

    async def _summary_api_worker(self, symbols: List[str], queue: Queue,
                                   spot_map: Dict[str, float]):
        """Fetch + transform + in-memory summarize; queue summary dicts.

        spot_map: {symbol: us_daily close on the actual option date} — used for
        GEX (gamma × OI × 100 × spot). Keyed by the summary's actual date.
        """
        for i, symbol in enumerate(symbols, 1):
            try:
                api_data = await self.get_option_data(symbol)
                if api_data:
                    contracts = self.transform_option_data(api_data, symbol)
                    if contracts:
                        spot = spot_map.get((symbol, contracts[0]['date']))
                        summary = self._summarize_contracts(symbol, contracts, spot=spot)
                        if summary:
                            await queue.put(summary)
                if i < len(symbols):
                    await asyncio.sleep(self.call_interval)
            except Exception as e:
                logger.error(f"[US_OPTION SUMMARY] {symbol}: {e}")
        await queue.put(None)

    async def run_collection_summary_only(self):
        """Summary-only collection: never persists raw us_option."""
        logger.info(f"[US_OPTION SUMMARY-ONLY] Starting for {self.target_date}")
        await self.init_pool()
        try:
            symbols = await self.get_active_symbols()
            if not symbols:
                logger.error("[US_OPTION] No symbols found")
                return

            # Spot prices for GEX (gamma × OI × 100 × spot). The option API may
            # return the nearest trading day rather than exactly target_date, so
            # fetch a small window of us_daily closes and key by (symbol, date).
            spot_map: Dict[tuple, float] = {}
            conn = await self.get_connection()
            try:
                rows = await conn.fetch(
                    """SELECT symbol, date, close FROM us_daily
                       WHERE symbol = ANY($1::text[])
                         AND date BETWEEN $2::date - INTERVAL '7 days' AND $2::date""",
                    symbols, self.target_date)
                for r in rows:
                    if r['close'] is not None:
                        spot_map[(r['symbol'], r['date'])] = float(r['close'])
            finally:
                if self.pool:
                    await self.pool.release(conn)
                else:
                    await conn.close()

            queue: Queue = Queue(maxsize=100)
            api_task = asyncio.create_task(
                self._summary_api_worker(symbols, queue, spot_map))

            batch, total = [], 0
            while True:
                item = await queue.get()
                if item is None:
                    if batch:
                        total += await self.save_summaries_batch(batch)
                    break
                batch.append(item)
                if len(batch) >= 100:
                    total += await self.save_summaries_batch(batch)
                    batch = []
            await api_task
            logger.info(f"[US_OPTION SUMMARY-ONLY] {self.target_date}: {total} summaries upserted")
        finally:
            await self.close_pool()

    async def aggregate_daily_summary(self, target_date: date) -> int:
        """Aggregate us_option to us_option_daily_summary using single query (including M19 strategy columns)"""
        conn = await self.get_connection()

        try:
            logger.info(f"[SUMMARY] Starting aggregation for {target_date}")

            # Single query aggregation using date index
            # Includes both basic stats and M19 strategy columns
            query = """
            INSERT INTO us_option_daily_summary (
                symbol, date,
                total_call_volume, total_put_volume,
                avg_implied_volatility,
                min_implied_volatility,
                max_implied_volatility,
                avg_call_iv,
                avg_put_iv,
                call_option_count,
                put_option_count
            )
            SELECT
                symbol,
                date,
                SUM(CASE WHEN type = 'call' THEN volume ELSE 0 END),
                SUM(CASE WHEN type = 'put' THEN volume ELSE 0 END),
                AVG(implied_volatility),
                MIN(implied_volatility),
                MAX(implied_volatility),
                AVG(CASE WHEN type = 'call'
                         AND implied_volatility IS NOT NULL
                         AND implied_volatility > 0
                         AND volume > 0
                         AND open_interest > 0
                         THEN implied_volatility END),
                AVG(CASE WHEN type = 'put'
                         AND implied_volatility IS NOT NULL
                         AND implied_volatility > 0
                         AND volume > 0
                         AND open_interest > 0
                         THEN implied_volatility END),
                COUNT(CASE WHEN type = 'call'
                           AND implied_volatility IS NOT NULL
                           AND implied_volatility > 0
                           AND volume > 0
                           AND open_interest > 0
                           THEN 1 END),
                COUNT(CASE WHEN type = 'put'
                           AND implied_volatility IS NOT NULL
                           AND implied_volatility > 0
                           AND volume > 0
                           AND open_interest > 0
                           THEN 1 END)
            FROM us_option
            WHERE date = $1
            GROUP BY symbol, date
            ON CONFLICT (symbol, date) DO UPDATE SET
                total_call_volume = EXCLUDED.total_call_volume,
                total_put_volume = EXCLUDED.total_put_volume,
                avg_implied_volatility = EXCLUDED.avg_implied_volatility,
                min_implied_volatility = EXCLUDED.min_implied_volatility,
                max_implied_volatility = EXCLUDED.max_implied_volatility,
                avg_call_iv = EXCLUDED.avg_call_iv,
                avg_put_iv = EXCLUDED.avg_put_iv,
                call_option_count = EXCLUDED.call_option_count,
                put_option_count = EXCLUDED.put_option_count
            """

            await conn.execute(query, target_date)

            # Get summary count
            count_query = """
            SELECT COUNT(*)
            FROM us_option_daily_summary
            WHERE date = $1
            """
            symbol_count = await conn.fetchval(count_query, target_date)

            logger.info(f"[SUMMARY] Completed: {symbol_count} symbols aggregated (with M19 columns)")

            return symbol_count

        except Exception as e:
            logger.error(f"[SUMMARY ERROR] {e}")
            raise
        finally:
            if self.pool:
                await self.pool.release(conn)
            else:
                await conn.close()

    async def api_worker(self, symbols: List[str], data_queue: Queue, partition_created_flag: asyncio.Event):
        """API worker for pipeline pattern - OPTIMIZED"""
        logger.info(f"[US_OPTION API] Starting API worker for {len(symbols)} symbols")
        for i, symbol in enumerate(symbols, 1):
            try:
                date_list = [self.get_date_string()]

                if self.collection_logger.is_collected('us_option', symbol, date_list):
                    logger.debug(f"[US_OPTION] Skip {symbol} - already collected")
                    continue

                logger.info(f"[US_OPTION API] [{i}/{len(symbols)}] Fetching {symbol}...")
                api_data = await self.get_option_data(symbol)

                #  First symbol: Check actual date and create partition
                if i == 1 and not partition_created_flag.is_set():
                    if api_data and 'data' in api_data and len(api_data['data']) > 0:
                        actual_date_str = api_data['data'][0]['date']
                        actual_date = datetime.strptime(actual_date_str, '%Y-%m-%d').date()
                        logger.info(f"[US_OPTION PARTITION] API returns data for actual date: {actual_date}")
                        await self.ensure_daily_partition_exists(actual_date)
                        partition_created_flag.set()
                    else:
                        logger.warning(f"[US_OPTION PARTITION] First symbol {symbol} has no data, will retry with next symbol")

                if api_data:
                    logger.info(f"[US_OPTION API] [{i}/{len(symbols)}] Received API data for {symbol}")
                    transformed = self.transform_option_data(api_data, symbol)
                    logger.info(f"[US_OPTION API] [{i}/{len(symbols)}] Transformed {len(transformed)} options for {symbol}")
                    if transformed:
                        await data_queue.put((symbol, transformed, date_list))
                        logger.info(f"[US_OPTION API] [{i}/{len(symbols)}] Queued {symbol} ({len(transformed)} options)")

                        #  Retry partition creation if first symbol failed
                        if not partition_created_flag.is_set():
                            actual_date = transformed[0]['date']
                            logger.info(f"[US_OPTION PARTITION] Creating partition from symbol {symbol}, date: {actual_date}")
                            await self.ensure_daily_partition_exists(actual_date)
                            partition_created_flag.set()
                    else:
                        logger.warning(f"[US_OPTION API] [{i}/{len(symbols)}] No option data for {symbol} after transformation")
                else:
                    logger.warning(f"[US_OPTION API] [{i}/{len(symbols)}] No API data received for {symbol}")

                if i < len(symbols):
                    await asyncio.sleep(self.call_interval)
            except Exception as e:
                logger.error(f"[US_OPTION API] Error {symbol}: {e}")

        await data_queue.put(None)

    async def db_worker(self, data_queue: Queue, batch_size: int = 100):
        """DB worker for pipeline pattern - OPTIMIZED"""
        logger.info(f"[US_OPTION DB] Starting DB worker, batch_size={batch_size}")
        batch = []
        total_saved = 0
        symbols_batch = []
        date_lists_batch = []
        items_received = 0

        while True:
            logger.info(f"[US_OPTION DB] Waiting for data from queue... (received {items_received} items so far, batch size: {len(batch)})")
            try:
                item = await data_queue.get()
                logger.info(f"[US_OPTION DB] Received item from queue: {type(item)}")
            except Exception as e:
                logger.error(f"[US_OPTION DB] Error getting item from queue: {e}")
                break

            if item is None:
                logger.info(f"[US_OPTION DB] Received None (end signal), processing final batch of {len(batch)} records")
                if batch:
                    logger.info(f"[US_OPTION DB] Saving final batch of {len(batch)} records")
                    saved = await self.save_option_data_optimized(batch)
                    logger.info(f"[US_OPTION DB] Final batch saved: {saved} records")
                    if saved > 0:
                        total_saved += saved
                        for symbol, date_list in zip(symbols_batch, date_lists_batch):
                            self.collection_logger.mark_collected('us_option', symbol, date_list, saved)
                        self.collection_logger.save_log()
                break

            items_received += 1
            symbol, data, date_list = item
            logger.info(f"[US_OPTION DB] Received {len(data)} records for {symbol}, adding to batch (current batch: {len(batch)})")
            batch.extend(data)
            symbols_batch.append(symbol)
            date_lists_batch.append(date_list)

            if len(batch) >= batch_size:
                logger.info(f"[US_OPTION DB] Batch size reached ({len(batch)} >= {batch_size}), saving to database...")
                try:
                    saved = await self.save_option_data_optimized(batch)
                    logger.info(f"[US_OPTION DB] Successfully saved {saved} records")
                    if saved > 0:
                        total_saved += saved
                        for sym, dl in zip(symbols_batch, date_lists_batch):
                            self.collection_logger.mark_collected('us_option', sym, dl, saved // len(symbols_batch))
                        self.collection_logger.save_log()
                    batch = []
                    symbols_batch = []
                    date_lists_batch = []
                except Exception as e:
                    logger.error(f"[US_OPTION DB] Error saving batch: {e}", exc_info=True)
                    batch = []
                    symbols_batch = []
                    date_lists_batch = []

        logger.info(f"[US_OPTION DB] DB worker finished. Total items received: {items_received}, Total saved: {total_saved}")
        return total_saved

    async def run_collection_optimized(self):
        """OPTIMIZED collection with pipeline pattern"""
        logger.info(f"[US_OPTION OPTIMIZED] Starting for {self.target_date}")
        logger.info(f"[US_OPTION OPTIMIZED] API rate: {60/self.call_interval:.0f} calls/min")

        await self.init_pool()

        all_symbols = await self.get_active_symbols()
        if not all_symbols:
            logger.error("[US_OPTION] No symbols found")
            await self.close_pool()
            return

        #  Put SPY first to ensure we get reliable data for partition creation
        if 'SPY' in all_symbols:
            all_symbols.remove('SPY')
            all_symbols.insert(0, 'SPY')
            logger.info("[US_OPTION] SPY moved to first position for partition detection")

        date_list = [self.get_date_string()]
        symbols = [s for s in all_symbols if not self.collection_logger.is_collected('us_option', s, date_list)]

        logger.info(f"[US_OPTION] Total: {len(all_symbols)}, To process: {len(symbols)}")

        if not symbols:
            logger.info("[US_OPTION] All symbols already collected")
            await self.close_pool()
            return

        start_time = datetime.now()

        #  Create Event flag for partition creation tracking
        partition_created = asyncio.Event()

        data_queue = Queue(maxsize=50)
        api_task = asyncio.create_task(self.api_worker(symbols, data_queue, partition_created))
        db_task = asyncio.create_task(self.db_worker(data_queue, batch_size=100))

        await api_task
        total_saved = await db_task

        await self.close_pool()

        duration = datetime.now() - start_time
        logger.info(f"[US_OPTION OPTIMIZED] Completed in {duration}, saved {total_saved} records")

        # Aggregate to summary table
        logger.info(f"[US_OPTION] Starting summary aggregation for {self.target_date}")
        await self.init_pool()
        try:
            summary_count = await self.aggregate_daily_summary(self.target_date)
            logger.info(f"[US_OPTION] Summary aggregation completed: {summary_count} symbols")

            # Free the per-date partition rows once they have been summarized.
            # ``us_option`` raw rows (one per (contract, date), 100k+/day) are
            # intermediate — only ``us_option_daily_summary`` is read by quant
            # downstream. Without this delete, 1 date ≈ 150 MB and 252 dates ≈
            # 31 GB of raw chain data accumulates and exhausts the Postgres
            # tablespace (we hit DiskFullError after ~191 dates).
            if summary_count > 0:
                conn = await self.get_connection()
                try:
                    deleted = await conn.execute(
                        "DELETE FROM us_option WHERE date = $1",
                        self.target_date)
                    logger.info(f"[US_OPTION] Freed raw rows for {self.target_date}: {deleted}")
                finally:
                    if self.pool:
                        await self.pool.release(conn)
                    else:
                        await conn.close()
        except Exception as e:
            logger.error(f"[US_OPTION] Summary aggregation failed: {e}")
        finally:
            await self.close_pool()


async def main():
    API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')
    DATABASE_URL = os.getenv('DATABASE_URL')

    if not API_KEY:
        logger.error("ALPHAVANTAGE_API_KEY environment variable is required")
        return

    if not DATABASE_URL:
        logger.error("DATABASE_URL environment variable is required")
        return

    # Collect data from 2025-10-01 to yesterday
    start_date = date(2025, 10, 1)
    end_date = date.today() - timedelta(days=1)

    print("\n" + "="*70)
    print("US Options Data Collection - Historical Range")
    print("="*70)
    print(f"Period: {start_date} to {end_date}")
    print(f"Total days: {(end_date - start_date).days + 1}")
    print(f"API rate: 300 calls/min (0.2s interval)")
    print(f"Auto partition: Daily RANGE partitioning enabled")
    print("="*70 + "\n")

    # Call interval: 0.2 seconds = 300 calls/min
    call_interval = 0.2

    # Loop through each date
    current_date = start_date
    dates_processed = 0
    dates_skipped = 0

    while current_date <= end_date:
        print(f"\n{'='*70}")
        print(f"Processing date: {current_date} ({dates_processed + 1}/{(end_date - start_date).days + 1})")
        print(f"{'='*70}")

        collector = USOptionCollector(API_KEY, DATABASE_URL, call_interval, current_date)

        try:
            await collector.run_collection_optimized()
            dates_processed += 1
        except Exception as e:
            logger.error(f"Error processing date {current_date}: {e}")
            dates_skipped += 1

        current_date += timedelta(days=1)

    print("\n" + "="*70)
    print("US Options Collection Completed")
    print("="*70)
    print(f"Dates processed: {dates_processed}")
    print(f"Dates skipped: {dates_skipped}")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
