"""MongoDB-backed collection progress tracker.

Replaces the prior JSON-file-per-collection scheme. All collectors share one
Mongo collection (`collection_log` in DB `alphafolio`), keyed by
(collection_name, symbol). This way re-runs across container restarts and
across collector types preserve which (symbol, dates) have already been
fetched, sparing redundant Alpha Vantage / Finnhub / KRX API calls.

Document schema (collection: `collection_log`):
{
  _id: "<collection_name>:<symbol>",
  collection_name: "us_stock_basic" | "us_daily" | "us_vwap" | "us_option" | ...
  symbol: "AAPL",
  dates: ["YYYY-MM-DD", ...],
  records_count: <int>,
  collected_at: ISODate
}

Public API kept identical to the prior JSON-based class:
- is_collected(collection_name, symbol, date_list) -> bool
- mark_collected(collection_name, symbol, date_list, records_count)
- save_log()           # flushes any buffered writes to Mongo
- get_collected_symbols(collection_name)
- get_symbol_entry(collection_name, symbol)
- self.collection_log  # dict shaped as {"collection_log": {<type>: {<sym>: entry}}}

Behavior on first start:
- Connects to MongoDB. Creates indexes.
- Loads all existing docs into memory (one find()).
- If the legacy JSON file at `log_file_path` exists and contains entries
  not yet in Mongo, those are migrated and the JSON file renamed to
  "<path>.migrated".
- If Mongo is unreachable, falls back to JSON for current session only.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, date
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from pymongo import MongoClient, ASCENDING, UpdateOne

logger = logging.getLogger(__name__)

MONGO_URL = os.getenv("MONGO_URL", "mongodb://mongo:27017")
MONGO_DB = os.getenv("MONGO_DB", "alphafolio")
COLLECTION_NAME_IN_MONGO = os.getenv("MONGO_COLLECTION_LOG", "collection_log")

# Buffer writes up to this many entries before auto-flushing. save_log()
# always flushes regardless of buffer size.
WRITE_BUFFER_THRESHOLD = 100


def _to_str_dates(date_list: Union[None, str, date, datetime, Iterable]) -> List[str]:
    if date_list is None:
        return []
    if isinstance(date_list, (str, date, datetime)):
        items: Iterable = [date_list]
    else:
        items = date_list
    out: List[str] = []
    for d in items:
        if d is None:
            continue
        if isinstance(d, (date, datetime)):
            out.append(d.strftime("%Y-%m-%d"))
        else:
            out.append(str(d))
    return out


def _parse_dt(v) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        except Exception:
            return None
    return None


class CollectionLogger:
    """MongoDB-backed multi-collection symbol-date progress tracker."""

    def __init__(self, log_file_path: str):
        # Kept only for backward compat + JSON migration source.
        self.log_file_path = log_file_path
        self._mongo: Optional[MongoClient] = None
        self._coll = None
        self._dirty: Set[Tuple[str, str]] = set()

        # In-memory mirror, structured to match the prior JSON-based API.
        self.collection_log: Dict[str, Any] = {"collection_log": {}, "meta": {}}

        try:
            self._mongo = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
            self._coll = self._mongo[MONGO_DB][COLLECTION_NAME_IN_MONGO]
            self._ensure_indexes()
            self._load_cache_from_db()
            self._maybe_migrate_json()
            logger.info(
                f"[CollectionLogger] MongoDB {MONGO_URL}/{MONGO_DB}/"
                f"{COLLECTION_NAME_IN_MONGO}: "
                f"{sum(len(v) for v in self.collection_log['collection_log'].values())} entries "
                f"across {len(self.collection_log['collection_log'])} collections"
            )
        except Exception as e:
            logger.warning(
                f"[CollectionLogger] MongoDB unavailable ({e}), falling back to JSON "
                f"at {log_file_path}"
            )
            self._coll = None
            self._load_from_json()

    # ------------------------------------------------------------------ Init
    def _ensure_indexes(self):
        self._coll.create_index(
            [("collection_name", ASCENDING), ("symbol", ASCENDING)], unique=True
        )
        self._coll.create_index("collection_name")
        self._coll.create_index("collected_at")

    def _load_cache_from_db(self):
        cache: Dict[str, Dict[str, dict]] = {}
        for doc in self._coll.find({}):
            cache.setdefault(doc["collection_name"], {})[doc["symbol"]] = {
                "dates": doc.get("dates", []),
                "records_count": int(doc.get("records_count", 0)),
                "collected_at": doc.get("collected_at"),
            }
        self.collection_log = {"collection_log": cache, "meta": {}}

    def _maybe_migrate_json(self):
        """One-time migration: import entries from legacy JSON file into Mongo."""
        if not self.log_file_path or not os.path.exists(self.log_file_path):
            return
        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(
                f"[CollectionLogger] can't read legacy {self.log_file_path}: {e}"
            )
            return

        # Tolerate both wrapped ({"collection_log": {...}}) and bare ({...}) shapes.
        wrapped = data.get("collection_log") if isinstance(data, dict) else None
        json_log = wrapped if isinstance(wrapped, dict) else data
        if not isinstance(json_log, dict):
            return

        bulk: List[UpdateOne] = []
        for collection_name, symbols in json_log.items():
            if not isinstance(symbols, dict):
                continue
            for symbol, entry in symbols.items():
                if not isinstance(entry, dict):
                    continue
                # Skip if already in Mongo
                if self.collection_log["collection_log"].get(collection_name, {}).get(symbol):
                    continue
                dates = entry.get("dates") or entry.get("date_list") or []
                if not isinstance(dates, list):
                    dates = [dates] if dates else []
                entry_doc = {
                    "collection_name": collection_name,
                    "symbol": symbol,
                    "dates": [str(d) for d in dates],
                    "records_count": int(entry.get("records_count", 0)),
                    "collected_at": _parse_dt(entry.get("collected_at")) or datetime.utcnow(),
                }
                bulk.append(
                    UpdateOne(
                        {"_id": f"{collection_name}:{symbol}"},
                        {"$setOnInsert": entry_doc},
                        upsert=True,
                    )
                )
                # Update cache
                self.collection_log["collection_log"].setdefault(collection_name, {})[symbol] = {
                    "dates": entry_doc["dates"],
                    "records_count": entry_doc["records_count"],
                    "collected_at": entry_doc["collected_at"],
                }

        if bulk:
            try:
                res = self._coll.bulk_write(bulk, ordered=False)
                logger.info(
                    f"[CollectionLogger] migrated {res.upserted_count} JSON entries "
                    f"from {self.log_file_path} → MongoDB"
                )
                # Rename to mark as migrated
                migrated_path = self.log_file_path + ".migrated"
                try:
                    os.replace(self.log_file_path, migrated_path)
                    logger.info(
                        f"[CollectionLogger] {self.log_file_path} → {migrated_path}"
                    )
                except OSError as e:
                    logger.warning(f"[CollectionLogger] couldn't rename JSON: {e}")
            except Exception as e:
                logger.error(f"[CollectionLogger] migration bulk_write failed: {e}")

    def _load_from_json(self):
        """JSON fallback used only when Mongo is unreachable. No persistence
        across restart, but at least intra-session dedupe works."""
        if not os.path.exists(self.log_file_path):
            return
        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "collection_log" in data:
                self.collection_log = data
            else:
                self.collection_log = {"collection_log": data, "meta": {}}
        except Exception as e:
            logger.warning(
                f"[CollectionLogger] couldn't load JSON {self.log_file_path}: {e}"
            )

    # ----------------------------------------------------------------- Query
    def is_collected(
        self,
        collection_name: str,
        symbol: str,
        date_list: Union[None, str, date, datetime, Iterable] = None,
    ) -> bool:
        coll = self.collection_log.get("collection_log", {})
        entry = coll.get(collection_name, {}).get(symbol)
        if not entry:
            return False
        targets = set(_to_str_dates(date_list))
        if not targets:
            return True
        recorded = set(str(d) for d in (entry.get("dates") or []))
        return targets.issubset(recorded)

    def get_collected_symbols(self, collection_name: str) -> List[str]:
        return list(self.collection_log.get("collection_log", {}).get(collection_name, {}).keys())

    def get_symbol_entry(self, collection_name: str, symbol: str) -> Optional[dict]:
        return self.collection_log.get("collection_log", {}).get(collection_name, {}).get(symbol)

    # ----------------------------------------------------------------- Mutate
    def mark_collected(
        self,
        collection_name: str,
        symbol: str,
        date_list: Union[None, str, date, datetime, Iterable] = None,
        records_count: int = 0,
    ) -> None:
        coll = self.collection_log.setdefault("collection_log", {})
        type_log = coll.setdefault(collection_name, {})
        prev = type_log.get(symbol, {})

        existing_dates = set(prev.get("dates", []))
        new_dates = set(_to_str_dates(date_list))
        merged_dates = sorted(existing_dates | new_dates)

        entry = {
            "dates": merged_dates,
            "records_count": int(prev.get("records_count", 0)) + int(records_count or 0),
            "collected_at": datetime.utcnow(),
        }
        type_log[symbol] = entry
        self._dirty.add((collection_name, symbol))

        if len(self._dirty) >= WRITE_BUFFER_THRESHOLD:
            self._flush()

    def save_log(self) -> None:
        """Flush any buffered upserts to Mongo. Safe to call repeatedly."""
        self._flush()

    def _flush(self) -> None:
        if not self._dirty:
            return
        if self._coll is None:
            # Mongo fallback unavailable — write to JSON
            self._save_to_json()
            self._dirty.clear()
            return

        bulk: List[UpdateOne] = []
        coll = self.collection_log.get("collection_log", {})
        for (cn, sym) in self._dirty:
            entry = coll.get(cn, {}).get(sym)
            if not entry:
                continue
            bulk.append(
                UpdateOne(
                    {"_id": f"{cn}:{sym}"},
                    {"$set": {
                        "collection_name": cn,
                        "symbol": sym,
                        "dates": entry["dates"],
                        "records_count": entry["records_count"],
                        "collected_at": entry["collected_at"],
                    }},
                    upsert=True,
                )
            )

        if not bulk:
            self._dirty.clear()
            return
        try:
            res = self._coll.bulk_write(bulk, ordered=False)
            # Per-collection totals — gives a feel for backfill progress in real time
            totals = {}
            for cn, _sym in self._dirty:
                totals[cn] = totals.get(cn, 0) + 1
            total_after = self._coll.estimated_document_count()
            logger.info(
                f"[CollectionLogger] flushed {len(bulk)} ops to MongoDB "
                f"(upserted={res.upserted_count}, modified={res.modified_count}, "
                f"by_collection={totals}, mongo_total={total_after})"
            )
        except Exception as e:
            logger.error(f"[CollectionLogger] bulk_write failed ({len(bulk)} ops): {e}")
        finally:
            self._dirty.clear()

    def _save_to_json(self):
        """JSON fallback write — used only when Mongo unavailable."""
        try:
            dirpath = os.path.dirname(self.log_file_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            self.collection_log.setdefault("meta", {})["last_saved"] = (
                datetime.utcnow().isoformat()
            )
            tmp = self.log_file_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.collection_log, f, ensure_ascii=False, indent=2, default=str)
            os.replace(tmp, self.log_file_path)
        except Exception as e:
            logger.error(f"[CollectionLogger] JSON fallback save failed: {e}")

    # ----------------------------------------------------------------- Admin
    def clear(self, collection_name: Optional[str] = None) -> None:
        if collection_name is None:
            self.collection_log = {"collection_log": {}, "meta": {}}
            self._dirty.clear()
            if self._coll is not None:
                self._coll.delete_many({})
        else:
            self.collection_log.setdefault("collection_log", {}).pop(collection_name, None)
            self._dirty = {(cn, sym) for (cn, sym) in self._dirty if cn != collection_name}
            if self._coll is not None:
                self._coll.delete_many({"collection_name": collection_name})
