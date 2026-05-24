# alpha/data/kr/gcs_handler.py
#
# CSV 파일 저장소 — 이전에는 Google Cloud Storage였으나 MongoDB로 교체됨.
# 호출부(krx.py, krx_배포용.py, krx_과거수집용.py) 변경 최소화를 위해
# 클래스/함수 이름(GCSHandler / get_gcs_handler)과 메서드 시그니처를 그대로 유지.
#
# 도큐먼트 스키마 (collection: csv_files):
#   { "_id": ObjectId,
#     "file_name": "<folder>/<file>"  ← 인덱스, GCS blob path와 동일 형식
#     "folder": "<folder>",
#     "data": "<csv text>",
#     "uploaded_at": ISODate }

import os
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
import logging

from pymongo import MongoClient, ASCENDING, DESCENDING

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONGO_URL = os.getenv("MONGO_URL", "mongodb://mongo:27017")
MONGO_DB = os.getenv("MONGO_DB", "alphafolio")
COLLECTION_NAME = os.getenv("MONGO_CSV_COLLECTION", "csv_files")


class GCSHandler:
    """MongoDB 기반 CSV 저장소. 이름은 호환성을 위해 GCSHandler 유지."""

    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.collection = None
        self._initialize_client()

    def _initialize_client(self):
        try:
            self.client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
            self.collection = self.client[MONGO_DB][COLLECTION_NAME]
            self.collection.create_index([("file_name", ASCENDING)], unique=True)
            self.collection.create_index([("uploaded_at", DESCENDING)])
            self.collection.create_index([("folder", ASCENDING)])
            logger.info(f"MongoDB CSV store initialized: {MONGO_URL}/{MONGO_DB}/{COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB client: {e}")
            self.client = None
            self.collection = None

    @staticmethod
    def _build_path(file_name: str, folder: str = "") -> str:
        return f"{folder}/{file_name}" if folder else file_name

    def upload_file(self, file_name: str, data: str, folder: str = "") -> bool:
        """CSV 텍스트를 MongoDB에 upsert."""
        if self.collection is None or not data:
            logger.error("No Mongo collection or data to upload")
            return False

        path = self._build_path(file_name, folder)
        try:
            self.collection.replace_one(
                {"file_name": path},
                {
                    "file_name": path,
                    "folder": folder,
                    "data": data,
                    "uploaded_at": datetime.utcnow(),
                },
                upsert=True,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upload to Mongo: {e}")
            return False

    def get_latest_file(self, prefix: str) -> Tuple[Optional[str], Optional[str]]:
        """지정 prefix로 시작하는 가장 최신 도큐먼트 (file_name, data) 반환."""
        if self.collection is None:
            return None, None

        try:
            # file_name이 timestamp suffix를 포함하므로 file_name 역순 정렬 = 최신
            # (기존 GCS 구현도 file_name max로 정렬)
            doc = self.collection.find_one(
                {"file_name": {"$regex": f"^{prefix}"}},
                sort=[("file_name", DESCENDING)],
            )
            if not doc:
                logger.warning(f"No files found with prefix {prefix}")
                return None, None
            logger.info(f"Found latest file: {doc['file_name']}")
            return doc["file_name"], doc["data"]
        except Exception as e:
            logger.error(f"Failed to get latest file from Mongo: {e}")
            return None, None

    def list_files(self, prefix: str = "") -> List[str]:
        if self.collection is None:
            return []
        try:
            query = {"file_name": {"$regex": f"^{prefix}"}} if prefix else {}
            files = [d["file_name"] for d in self.collection.find(query, {"file_name": 1})]
            logger.info(f"Found {len(files)} files with prefix '{prefix}'")
            return files
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []

    def delete_file(self, file_path: str) -> bool:
        if self.collection is None:
            return False
        try:
            res = self.collection.delete_one({"file_name": file_path})
            if res.deleted_count:
                logger.info(f"Deleted file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False

    def delete_old_files(self, prefix: str, days_old: int = 30) -> int:
        if self.collection is None:
            return 0
        try:
            cutoff = datetime.utcnow() - timedelta(days=days_old)
            res = self.collection.delete_many({
                "file_name": {"$regex": f"^{prefix}"},
                "uploaded_at": {"$lt": cutoff},
            })
            logger.info(f"Deleted {res.deleted_count} old files (older than {days_old} days)")
            return res.deleted_count
        except Exception as e:
            logger.error(f"Failed to delete old files: {e}")
            return 0

    def delete_all_files_in_folder(self, folder: str) -> int:
        if self.collection is None:
            return 0
        try:
            res = self.collection.delete_many({"file_name": {"$regex": f"^{folder}/"}})
            logger.info(f"Deleted {res.deleted_count} files from folder '{folder}'")
            return res.deleted_count
        except Exception as e:
            logger.error(f"Failed to delete files in folder {folder}: {e}")
            return 0

    def cleanup_files_by_pattern(self, prefix: str, keep_latest: int = 5) -> int:
        """패턴별 최신 N개만 유지, 나머지 삭제."""
        if self.collection is None:
            return 0
        try:
            docs = list(
                self.collection.find(
                    {"file_name": {"$regex": f"^{prefix}"}},
                    {"_id": 1, "uploaded_at": 1, "file_name": 1},
                ).sort("uploaded_at", DESCENDING)
            )
            if len(docs) <= keep_latest:
                logger.info(f"Only {len(docs)} files found, no cleanup needed")
                return 0

            ids_to_delete = [d["_id"] for d in docs[keep_latest:]]
            res = self.collection.delete_many({"_id": {"$in": ids_to_delete}})
            logger.info(
                f"Cleanup completed: kept {keep_latest} latest files, "
                f"deleted {res.deleted_count} files"
            )
            return res.deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup files: {e}")
            return 0


_gcs_handler: Optional[GCSHandler] = None


def get_gcs_handler() -> GCSHandler:
    global _gcs_handler
    if _gcs_handler is None:
        _gcs_handler = GCSHandler()
    return _gcs_handler
