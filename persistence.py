"""
SQLite-backed persistence for DeepAgent Harness.
Stores agent configurations, harness definitions, and project sessions.
All data survives app restarts.
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from config import DB_PATH


def _get_connection() -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode for concurrent reads."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = _get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agents (
            name TEXT PRIMARY KEY,
            config_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS harnesses (
            name TEXT PRIMARY KEY,
            config_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            files_json TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


class AgentStore:
    """CRUD operations for custom agent configurations."""

    @staticmethod
    def save(config: dict) -> None:
        conn = _get_connection()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO agents (name, config_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET config_json=excluded.config_json, updated_at=excluded.updated_at",
            (config["name"], json.dumps(config), now, now),
        )
        conn.commit()
        conn.close()

    @staticmethod
    def get(name: str) -> Optional[dict]:
        conn = _get_connection()
        row = conn.execute("SELECT config_json FROM agents WHERE name = ?", (name,)).fetchone()
        conn.close()
        return json.loads(row["config_json"]) if row else None

    @staticmethod
    def list_all() -> list[dict]:
        conn = _get_connection()
        rows = conn.execute("SELECT config_json FROM agents ORDER BY created_at").fetchall()
        conn.close()
        return [json.loads(r["config_json"]) for r in rows]

    @staticmethod
    def delete(name: str) -> bool:
        conn = _get_connection()
        cursor = conn.execute("DELETE FROM agents WHERE name = ?", (name,))
        conn.commit()
        conn.close()
        return cursor.rowcount > 0

    @staticmethod
    def exists(name: str) -> bool:
        conn = _get_connection()
        row = conn.execute("SELECT 1 FROM agents WHERE name = ?", (name,)).fetchone()
        conn.close()
        return row is not None


class HarnessStore:
    """CRUD operations for harness workflow configurations."""

    @staticmethod
    def save(config: dict) -> None:
        conn = _get_connection()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO harnesses (name, config_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET config_json=excluded.config_json, updated_at=excluded.updated_at",
            (config["name"], json.dumps(config), now, now),
        )
        conn.commit()
        conn.close()

    @staticmethod
    def get(name: str) -> Optional[dict]:
        conn = _get_connection()
        row = conn.execute("SELECT config_json FROM harnesses WHERE name = ?", (name,)).fetchone()
        conn.close()
        return json.loads(row["config_json"]) if row else None

    @staticmethod
    def list_all() -> list[dict]:
        conn = _get_connection()
        rows = conn.execute("SELECT config_json FROM harnesses ORDER BY created_at").fetchall()
        conn.close()
        return [json.loads(r["config_json"]) for r in rows]

    @staticmethod
    def delete(name: str) -> bool:
        conn = _get_connection()
        cursor = conn.execute("DELETE FROM harnesses WHERE name = ?", (name,))
        conn.commit()
        conn.close()
        return cursor.rowcount > 0


class ProjectStore:
    """CRUD operations for project sessions."""

    @staticmethod
    def save(project_id: str, name: str, thread_id: str, files: Optional[dict] = None) -> None:
        conn = _get_connection()
        now = datetime.now(timezone.utc).isoformat()
        files_json = json.dumps(files or {})
        conn.execute(
            "INSERT INTO projects (id, name, thread_id, files_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET name=excluded.name, files_json=excluded.files_json, updated_at=excluded.updated_at",
            (project_id, name, thread_id, files_json, now, now),
        )
        conn.commit()
        conn.close()

    @staticmethod
    def get(project_id: str) -> Optional[dict]:
        conn = _get_connection()
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        conn.close()
        if not row:
            return None
        return {
            "id": row["id"], "name": row["name"], "thread_id": row["thread_id"],
            "files": json.loads(row["files_json"]),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    @staticmethod
    def list_all() -> list[dict]:
        conn = _get_connection()
        rows = conn.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
        conn.close()
        return [
            {"id": r["id"], "name": r["name"], "thread_id": r["thread_id"],
             "files": json.loads(r["files_json"]),
             "created_at": r["created_at"], "updated_at": r["updated_at"]}
            for r in rows
        ]

    @staticmethod
    def delete(project_id: str) -> bool:
        conn = _get_connection()
        cursor = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        conn.commit()
        conn.close()
        return cursor.rowcount > 0

    @staticmethod
    def create_new(name: str = "Untitled Project") -> dict:
        project_id = str(uuid.uuid4())
        thread_id = str(uuid.uuid4())
        ProjectStore.save(project_id, name, thread_id)
        return {"id": project_id, "name": name, "thread_id": thread_id}


init_db()
