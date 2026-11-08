"""
DeepAgent Harness 的持久化模块。
存储智能体配置、harness 定义、项目会话，以及文件的轻量元数据。
文件实体一律落在本地的 sandbox/<项目> 目录（同时是 Docker 容器的挂载源），
数据库只保存元数据（标识 / 大小 / 类型 / 时间），不再存储文件内容。
"""
import json
import mimetypes
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import DB_PATH, SANDBOX_DIR


def _get_connection() -> sqlite3.Connection:
    """获取一个开启 WAL 模式的 SQLite 连接（元数据读写均为短连接，用完即关）。"""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    # 并行子进程可能同时读配置，加锁等待避免偶发的 database is locked
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """创建所有数据表（如果尚不存在）。"""
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
        CREATE TABLE IF NOT EXISTS file_metadata (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            mime_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (task_id, file_path)
        );
        CREATE INDEX IF NOT EXISTS idx_file_metadata_task ON file_metadata (task_id);
        CREATE TABLE IF NOT EXISTS project_messages (
            message_id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            channel TEXT DEFAULT 'main',
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_calls TEXT DEFAULT '',
            tool_call_id TEXT DEFAULT '',
            name TEXT DEFAULT '',
            reasoning TEXT DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_project_messages_ws ON project_messages (workspace_id);
    """)
    conn.commit()
    # 老库补列：思考过程与执行渠道都是后加的字段，已有库需要就地迁移
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(project_messages)")}
    if "reasoning" not in columns:
        conn.execute("ALTER TABLE project_messages ADD COLUMN reasoning TEXT DEFAULT ''")
        conn.commit()
    if "channel" not in columns:
        conn.execute("ALTER TABLE project_messages ADD COLUMN channel TEXT DEFAULT 'main'")
        conn.commit()
    conn.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_name(name: str) -> str:
    """把项目名清洗成安全的目录名。"""
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "-", str(name or "")).strip("-.")
    return cleaned[:40] or "project"


# ── 文件元数据表 CRUD ────────────────────────────────────────────────

def upsert_file_meta(task_id: str, rel_path: str, size: int) -> dict:
    """登记/刷新一条文件元数据，返回该记录的完整字段。"""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT id, created_at FROM file_metadata WHERE task_id = ? AND file_path = ?",
            (task_id, rel_path),
        ).fetchone()
        mime = mimetypes.guess_type(rel_path)[0] or "text/plain"
        if row:
            conn.execute(
                "UPDATE file_metadata SET file_size = ?, mime_type = ? WHERE id = ?",
                (size, mime, row["id"]),
            )
            record = {
                "id": row["id"], "task_id": task_id,
                "file_name": Path(rel_path).name, "file_path": rel_path,
                "file_size": size, "mime_type": mime, "created_at": row["created_at"],
            }
        else:
            record = {
                "id": f"file_{uuid.uuid4().hex[:10]}", "task_id": task_id,
                "file_name": Path(rel_path).name, "file_path": rel_path,
                "file_size": size, "mime_type": mime, "created_at": _utc_now(),
            }
            conn.execute(
                "INSERT INTO file_metadata (id, task_id, file_name, file_path, file_size, mime_type, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (record["id"], record["task_id"], record["file_name"], record["file_path"],
                 record["file_size"], record["mime_type"], record["created_at"]),
            )
        conn.commit()
        return record
    finally:
        conn.close()


def delete_file_meta(task_id: str, rel_path: str) -> bool:
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "DELETE FROM file_metadata WHERE task_id = ? AND file_path = ?",
            (task_id, rel_path),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def list_file_meta(task_id: str) -> list[dict]:
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM file_metadata WHERE task_id = ? ORDER BY file_path",
            (task_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_file_meta_by_task(task_id: str) -> int:
    conn = _get_connection()
    try:
        cursor = conn.execute("DELETE FROM file_metadata WHERE task_id = ?", (task_id,))
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


class AgentStore:
    """自定义智能体配置的增删改查（CRUD）操作。"""

    @staticmethod
    def save(config: dict) -> None:
        conn = _get_connection()
        now = _utc_now()
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
    """harness 工作流配置的增删改查（CRUD）操作。"""

    @staticmethod
    def save(config: dict) -> None:
        conn = _get_connection()
        now = _utc_now()
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
    """项目会话的增删改查（CRUD）操作。"""

    @staticmethod
    def save(project_id: str, name: str, thread_id: str, files: Optional[dict] = None) -> None:
        conn = _get_connection()
        now = _utc_now()
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
    def get_by_thread(thread_id: str) -> Optional[dict]:
        """按线程 id 反查项目（用于把工作区定位到 sandbox 目录名）。"""
        conn = _get_connection()
        row = conn.execute("SELECT * FROM projects WHERE thread_id = ?", (thread_id,)).fetchone()
        conn.close()
        if not row:
            return None
        return {"id": row["id"], "name": row["name"], "thread_id": row["thread_id"]}

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


class WorkspaceStore:
    """每个 project 一个工作区：文件实体落本地 sandbox 目录，数据库只留元数据。

    sandbox/<项目名>_<thread_id前8位>/ 就是虚拟文件系统的物理位置，
    同时是该项目 Docker 容器的挂载源（容器内 /app/workspace）。
    旧版「SQLite 存内容 + 模块级长连接 + 写穿缓存 + 全局锁」的逻辑已全部移除：
    文件读写直接走操作系统，元数据为低频短连接访问。
    """

    # thread_id -> 目录名 的进程内缓存，避免热路径反复查项目表
    _dir_cache: dict[str, str] = {}
    # thread_id -> 外部临时工作区目录。并行子智能体在独立进程里把工作区
    # 重定向到自己的沙盒目录，从而隔离多个子任务对同一文件的并发修改。
    _sandbox_roots: dict[str, Path] = {}

    @classmethod
    def bind_sandbox_root(cls, workspace_id: str, root: str | Path) -> Path:
        """把某个工作区临时重定向到外部目录（仅当前进程生效）。"""
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        cls._sandbox_roots[workspace_id] = path
        return path

    @classmethod
    def unbind_sandbox_root(cls, workspace_id: str) -> None:
        """解除外部目录绑定，恢复按 sandbox/<项目> 读写。"""
        cls._sandbox_roots.pop(workspace_id, None)

    @classmethod
    def is_sandboxed(cls, workspace_id: str) -> bool:
        """当前工作区是否被重定向到外部沙盒目录。"""
        return workspace_id in cls._sandbox_roots

    @classmethod
    def workspace_dir(cls, workspace_id: str) -> Path:
        """定位工作区目录（不存在则创建）。目录名 = 项目名清洗 + thread_id 前 8 位。"""
        override = cls._sandbox_roots.get(workspace_id)
        if override is not None:
            override.mkdir(parents=True, exist_ok=True)
            return override
        dir_name = cls._dir_cache.get(workspace_id)
        if not dir_name:
            project = ProjectStore.get_by_thread(workspace_id)
            safe = _sanitize_name(project["name"]) if project else "project"
            dir_name = f"{safe}_{workspace_id[:8]}"
            cls._dir_cache[workspace_id] = dir_name
        path = SANDBOX_DIR / dir_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def dir_name_of(cls, workspace_id: str) -> str:
        override = cls._sandbox_roots.get(workspace_id)
        if override is not None:
            return override.name
        cls.workspace_dir(workspace_id)
        return cls._dir_cache[workspace_id]

    @classmethod
    def rename_workspace(cls, workspace_id: str, new_project_name: str) -> None:
        """项目重命名时同步重命名 sandbox 目录，保持目录名可读。"""
        old_dir = cls.workspace_dir(workspace_id)
        new_dir_name = f"{_sanitize_name(new_project_name)}_{workspace_id[:8]}"
        new_dir = SANDBOX_DIR / new_dir_name
        if new_dir != old_dir:
            if new_dir.exists():
                # 目标目录已被占用（同名项目）：放弃重命名，保留旧目录
                cls._dir_cache[workspace_id] = old_dir.name
                return
            try:
                old_dir.rename(new_dir)
            except OSError:
                return
        cls._dir_cache[workspace_id] = new_dir_name

    @staticmethod
    def normalize_path(path: str) -> Optional[str]:
        """把虚拟路径归一化成工作区内的相对路径；越出工作区返回 None。

        子智能体在容器里工作时经常直接给出容器内的工作区绝对路径
        （/app/workspace/xxx），这里统一剥掉该前缀，保证读写与索引都用同一个键。
        子智能体还常把工作区当成容器根目录（/reports/a.md、/arch/notes.md），
        这类路径同样去掉前导 / 按工作区相对路径处理——它不是真实系统路径，
        落盘时仍拼在工作区目录下，因此不会读写到工作区之外。
        """
        raw = str(path or "").strip().replace("\\", "/")
        container = "/app/workspace"
        if raw == container:
            return ""
        if raw.startswith(container + "/"):
            raw = raw[len(container) + 1:]
        elif raw.startswith("/"):
            raw = raw.lstrip("/")
        parts = [p for p in raw.split("/") if p not in ("", ".")]
        # 「..」一律拒绝：joinpath 允许它跳出工作区目录
        if not parts or ".." in parts:
            return None
        return "/".join(parts)

    @classmethod
    def _abs_path(cls, workspace_id: str, path: str) -> Optional[Path]:
        """把虚拟相对路径解析为工作区内绝对路径；路径逃逸返回 None。"""
        rel = cls.normalize_path(path)
        if rel is None:
            return None
        if not rel:
            return cls.workspace_dir(workspace_id)
        return cls.workspace_dir(workspace_id).joinpath(*rel.split("/"))

    @classmethod
    def read(cls, workspace_id: str, path: str) -> Optional[str]:
        """读取文件内容（直接读 sandbox 目录）。文件不存在返回 None。"""
        dest = cls._abs_path(workspace_id, path)
        if dest is None or not dest.is_file():
            return None
        try:
            return dest.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    @classmethod
    def write(cls, workspace_id: str, path: str, content: str) -> dict:
        """写入文件实体到 sandbox 目录，并登记元数据。返回该文件的元数据记录。"""
        rel_path = cls.normalize_path(path)
        if not rel_path:
            raise ValueError(f"非法路径：{path}")
        dest = cls.workspace_dir(workspace_id).joinpath(*rel_path.split("/"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        if cls.is_sandboxed(workspace_id):
            # 沙盒产物由汇总节点决定是否写回主工作区，这里不登记元数据，避免污染
            return cls._shadow_entry(rel_path, content)
        rel = cls._rel_of(workspace_id, rel_path)
        record = upsert_file_meta(workspace_id, rel, len(content.encode("utf-8")))
        return cls._entry_of(record)

    @staticmethod
    def _shadow_entry(path: str, content: str) -> dict:
        """沙盒模式下构造与 state.files 结构一致的轻量索引条目（不写元数据表）。"""
        return {
            "size": len(content.encode("utf-8")),
            "mime": mimetypes.guess_type(path)[0] or "text/plain",
            "created_at": _utc_now(),
        }

    @classmethod
    def delete(cls, workspace_id: str, path: str) -> bool:
        dest = cls._abs_path(workspace_id, path)
        if dest is None or not dest.exists():
            return False
        try:
            dest.unlink()
        except OSError:
            return False
        # 顺带清理空父目录
        parent = dest.parent
        while parent != cls.workspace_dir(workspace_id):
            try:
                parent.rmdir()
            except OSError:
                break
        if not cls.is_sandboxed(workspace_id):
            delete_file_meta(workspace_id, cls._rel_of(workspace_id, path))
        return True

    @classmethod
    def _rel_of(cls, workspace_id: str, vpath: str) -> str:
        """虚拟路径 → 元数据表里的 file_path（sandbox/<目录>/<相对路径>）。"""
        return f"sandbox/{cls.dir_name_of(workspace_id)}/{vpath}"

    @staticmethod
    def _entry_of(record: dict) -> dict:
        """元数据记录 → state.files 索引条目（轻量，不含内容）。"""
        return {
            "size": record["file_size"],
            "mime": record["mime_type"],
            "created_at": record["created_at"],
        }

    @classmethod
    def manifest(cls, workspace_id: str) -> dict:
        """返回 {vpath: {size, mime, created_at}}，用于填进 state。"""
        dir_name = cls.dir_name_of(workspace_id)
        prefix = f"sandbox/{dir_name}/"
        result = {}
        for record in list_file_meta(workspace_id):
            if record["file_path"].startswith(prefix):
                result[record["file_path"][len(prefix):]] = cls._entry_of(record)
        return result

    @classmethod
    def read_all(cls, workspace_id: str) -> dict:
        """一次性取出全部文件内容，供导出与人工查看使用。"""
        root = cls.workspace_dir(workspace_id)
        files = {}
        for full in sorted(root.rglob("*")):
            if not full.is_file():
                continue
            files[full.relative_to(root).as_posix()] = full.read_text(
                encoding="utf-8", errors="replace"
            )
        return files

    @classmethod
    def scan_dir(cls, workspace_id: str) -> dict:
        """扫描工作区目录，返回 {vpath: (mtime_ns, size)}；跳过噪音目录与文件。"""
        root = cls.workspace_dir(workspace_id)
        snapshot = {}
        for full in root.rglob("*"):
            if not full.is_file():
                continue
            rel = full.relative_to(root).as_posix()
            if any(part in ("__pycache__", ".git", ".pytest_cache", "node_modules", ".venv", ".idea")
                   for part in full.relative_to(root).parts[:-1]):
                continue
            if rel.endswith((".pyc", ".pyo")):
                continue
            try:
                stat = full.stat()
            except OSError:
                continue
            snapshot[rel] = (stat.st_mtime_ns, stat.st_size)
        return snapshot

    @classmethod
    def sync_dir(cls, workspace_id: str, before: Optional[dict] = None) -> tuple[dict, list[str]]:
        """把目录里的新增/变更文件登记进元数据表。

        before 为执行前快照（scan_dir 的返回值）；传 None 时登记全部文件。
        返回 (entries: {vpath: entry}, changed: [vpath])。
        """
        after = cls.scan_dir(workspace_id)
        entries: dict = {}
        changed: list[str] = []
        sandboxed = cls.is_sandboxed(workspace_id)
        for rel, (mtime, size) in after.items():
            if before is not None and before.get(rel) == (mtime, size):
                continue  # 未发生变化
            if sandboxed:
                # 沙盒模式只回传索引，不写主工作区的元数据表
                entries[rel] = {
                    "size": size,
                    "mime": mimetypes.guess_type(rel)[0] or "text/plain",
                    "created_at": _utc_now(),
                }
            else:
                record = upsert_file_meta(workspace_id, cls._rel_of(workspace_id, rel), size)
                entries[rel] = cls._entry_of(record)
            changed.append(rel)
        return entries, changed

    @classmethod
    def import_legacy(cls, workspace_id: str, files: dict) -> dict:
        """把旧版「state.files 直接存内容」的数据搬进 sandbox 目录，返回新的索引。"""
        for path, content in (files or {}).items():
            if isinstance(content, str):
                cls.write(workspace_id, path, content)
        return cls.manifest(workspace_id)

    @classmethod
    def delete_workspace(cls, workspace_id: str) -> None:
        """删除整个工作区：sandbox 目录 + 全部元数据记录。"""
        import shutil

        dir_name = cls._dir_cache.pop(workspace_id, None)
        if dir_name is None:
            dir_name = cls.dir_name_of(workspace_id)
            cls._dir_cache.pop(workspace_id, None)
        target = SANDBOX_DIR / dir_name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        delete_file_meta_by_task(workspace_id)


def _migrate_legacy_workspace_files() -> None:
    """一次性迁移：把旧版 workspace_files 表里的文件内容搬到 sandbox 目录，然后删表。"""
    conn = _get_connection()
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "workspace_files" not in tables:
            return
        rows = conn.execute(
            "SELECT workspace_id, path, content FROM workspace_files"
        ).fetchall()
        conn.close()
        for row in rows:
            if isinstance(row["content"], str):
                try:
                    WorkspaceStore.write(row["workspace_id"], row["path"], row["content"])
                except (OSError, ValueError):
                    continue
        conn = _get_connection()
        conn.execute("DROP TABLE workspace_files")
        conn.commit()
    finally:
        conn.close()


init_db()
_migrate_legacy_workspace_files()
