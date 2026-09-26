from __future__ import annotations

from datetime import UTC, datetime
from threading import Lock
from typing import Any

from app.EvoRAG.config import EvoRAGSettings, settings as evorag_settings


_lock = Lock()
_initialized = False
_last_error = ""
_last_checked_at = ""


def initialize_evorag_database() -> None:
    global _initialized, _last_error, _last_checked_at

    with _lock:
        _last_checked_at = now_iso()
        try:
            from app.EvoRAG.entity_store.repository import MySQLEntityRepository

            MySQLEntityRepository().init_schema()
            _initialized = True
            _last_error = ""
        except Exception as exc:
            _initialized = False
            _last_error = str(exc)


def connect_evorag_mysql(config: EvoRAGSettings | None = None):
    try:
        import pymysql  # type: ignore
    except Exception as exc:
        raise RuntimeError("pymysql is required for EvoRAG MySQL storage. Install pymysql>=1.1.0.") from exc

    active_config = config or evorag_settings
    return pymysql.connect(
        host=active_config.mysql_host,
        port=active_config.mysql_port,
        user=active_config.mysql_user,
        password=active_config.mysql_password,
        database=active_config.mysql_database,
        charset=active_config.mysql_charset,
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
    )


def evorag_database_status() -> dict[str, Any]:
    status = {
        "initialized": _initialized,
        "available": False,
        "host": evorag_settings.mysql_host,
        "port": evorag_settings.mysql_port,
        "database": evorag_settings.mysql_database,
        "last_checked_at": _last_checked_at,
        "error": _last_error,
    }

    try:
        with connect_evorag_mysql() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        status["available"] = True
        status["error"] = ""
    except Exception as exc:
        status["error"] = str(exc)

    return status


def now_iso() -> str:
    return datetime.now(UTC).isoformat()
