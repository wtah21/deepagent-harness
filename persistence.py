"""
DeepAgent Harness 的持久化模块。
存储智能体配置、harness 定义、项目会话，以及文件的轻量元数据。
文件实体一律落在本地的 sandbox/<项目> 目录（同时是 Docker 容器的挂载源），
数据库只保存元数据（标识 / 大小 / 类型 / 时间），不再存储文件内容。
"""
import json
import mimetypes
import os
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import DB_PATH, FILE_LOCK_TIMEOUT, SANDBOX_DIR


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
            workspace_path TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
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
    # 项目可绑定用户指定的本地工作目录，空字符串表示使用默认 sandbox 目录
    project_columns = {row["name"] for row in conn.execute("PRAGMA table_info(projects)")}
    if "workspace_path" not in project_columns:
        conn.execute("ALTER TABLE projects ADD COLUMN workspace_path TEXT DEFAULT ''")
        conn.commit()
    # 文件实体统一落本地工作区目录：旧版的文件元数据表与项目内文件索引不再使用。
    # 老库里的 files_json 列保留（SQLite 删列代价高），内容清空即可。
    conn.execute("DROP TABLE IF EXISTS file_metadata")
    if "files_json" in project_columns:
        conn.execute("UPDATE projects SET files_json = '{}'")
    conn.commit()
    conn.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_name(name: str) -> str:
    """把项目名清洗成安全的目录名。"""
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "-", str(name or "")).strip("-.")
    return cleaned[:40] or "project"


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
    def save(
        project_id: str,
        name: str,
        thread_id: str,
        workspace_path: Optional[str] = None,
    ) -> None:
        """保存项目档案。workspace_path 为 None 时保留原值，传空串表示恢复默认沙盒目录。

        文件不再入库：项目只记录名称、线程与绑定的工作目录，文件实体都在目录里。
        """
        conn = _get_connection()
        now = _utc_now()
        conn.execute(
            "INSERT INTO projects (id, name, thread_id, workspace_path, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET name=excluded.name, "
            "workspace_path=COALESCE(?, workspace_path), updated_at=excluded.updated_at",
            (project_id, name, thread_id, workspace_path or "", now, now, workspace_path),
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
            "workspace_path": row["workspace_path"] or "",
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    @staticmethod
    def get_by_thread(thread_id: str) -> Optional[dict]:
        """按线程 id 反查项目（用于把工作区定位到 sandbox 目录名或外部目录）。"""
        conn = _get_connection()
        row = conn.execute("SELECT * FROM projects WHERE thread_id = ?", (thread_id,)).fetchone()
        conn.close()
        if not row:
            return None
        return {
            "id": row["id"], "name": row["name"], "thread_id": row["thread_id"],
            "workspace_path": row["workspace_path"] or "",
        }

    @staticmethod
    def list_all() -> list[dict]:
        conn = _get_connection()
        rows = conn.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
        conn.close()
        return [
            {"id": r["id"], "name": r["name"], "thread_id": r["thread_id"],
             "workspace_path": r["workspace_path"] or "",
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
    """每个 project 一个工作区：文件实体直接落本地 sandbox 目录，数据库不存文件元数据。

    sandbox/<项目名>_<thread_id前8位>/ 就是工作区的物理位置，
    同时是该项目 Docker 容器的挂载源（容器内 /app/workspace）。
    文件读写直接走操作系统，文件一览按需扫描目录。
    """

    # thread_id -> 目录名 的进程内缓存，避免热路径反复查项目表
    _dir_cache: dict[str, str] = {}
    # thread_id -> 项目绑定的外部工作目录（None 表示使用默认 sandbox 目录）。
    # 外部目录是用户自己的文件，agent 直接读写原目录，不做拷贝。
    _external_dirs: dict[str, Optional[Path]] = {}

    @classmethod
    def external_dir_of(cls, workspace_id: str) -> Optional[Path]:
        """项目绑定的外部工作目录；未绑定、或绑定后目录已不可用时返回 None。"""
        if workspace_id in cls._external_dirs:
            return cls._external_dirs[workspace_id]
        project = ProjectStore.get_by_thread(workspace_id)
        raw = (project or {}).get("workspace_path") or ""
        path = Path(raw).expanduser() if raw else None
        resolved = path.resolve() if (path and path.is_dir()) else None
        cls._external_dirs[workspace_id] = resolved
        return resolved

    @classmethod
    def bind_external_dir(cls, workspace_id: str, root: str | Path) -> Path:
        """把项目工作区指向外部目录（仅更新进程内缓存，持久化由调用方写项目表）。"""
        path = Path(root).expanduser().resolve()
        cls._external_dirs[workspace_id] = path
        return path

    @classmethod
    def unbind_external_dir(cls, workspace_id: str) -> None:
        """解除外部目录绑定，恢复默认 sandbox 工作区（持久化由调用方清空项目表字段）。"""
        cls._external_dirs[workspace_id] = None

    @staticmethod
    def validate_external_dir(raw: str) -> tuple[Optional[Path], str]:
        """校验用户填写的工作目录，返回 (绝对路径, 错误信息)。"""
        text = str(raw or "").strip()
        if not text:
            return None, "请填写目录路径。"
        path = Path(text).expanduser()
        if not path.is_absolute():
            return None, "需要填写绝对路径（例如 D:\\\\projects\\\\demo）。"
        if not path.exists():
            return None, "目录不存在。"
        if not path.is_dir():
            return None, "该路径不是目录。"
        if not os.access(path, os.R_OK):
            return None, "目录不可读，请检查权限。"
        return path.resolve(), ""

    # 文件写锁：并行分支共享同一工作区，按 (workspace, 相对路径) 粒度串行化写入。
    # 不同文件的写入互不阻塞，同一文件的 read-modify-write 不会互相覆盖。
    _file_locks: dict[str, threading.RLock] = {}
    _file_locks_guard = threading.Lock()

    @classmethod
    def _file_lock(cls, workspace_id: str, rel_path: str) -> threading.RLock:
        """取某个文件的进程内可重入锁（首次访问时创建）。"""
        key = f"{workspace_id}::{rel_path}"
        with cls._file_locks_guard:
            return cls._file_locks.setdefault(key, threading.RLock())

    @classmethod
    @contextmanager
    def write_lock(cls, workspace_id: str, path: str, timeout: float = FILE_LOCK_TIMEOUT):
        """按文件粒度加写锁；等锁超时抛 TimeoutError，由调用方转成可读错误。"""
        rel = cls.normalize_path(path) or str(path or "")
        lock = cls._file_lock(workspace_id, rel)
        if not lock.acquire(timeout=timeout):
            raise TimeoutError(f"等待文件锁超时（{timeout:.0f}s）：{rel}")
        try:
            yield
        finally:
            lock.release()

    @classmethod
    def workspace_dir(cls, workspace_id: str) -> Path:
        """定位工作区目录：项目绑定的外部目录 > sandbox/<项目名_线程前8位>。"""
        external = cls.external_dir_of(workspace_id)
        if external is not None:
            return external
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
        if cls.external_dir_of(workspace_id) is not None:
            # 外部目录不参与 sandbox 目录命名，用固定前缀保证容器名稳定
            return f"ext_{workspace_id[:8]}"
        cls.workspace_dir(workspace_id)
        return cls._dir_cache[workspace_id]

    @classmethod
    def rename_workspace(cls, workspace_id: str, new_project_name: str) -> None:
        """项目重命名时同步重命名 sandbox 目录，保持目录名可读。"""
        if cls.external_dir_of(workspace_id) is not None:
            # 外部工作目录是用户自己的文件夹，名字不随项目改名变化
            return
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
        """写入文件实体到工作区目录，返回该文件的内存索引条目。

        文件内容与索引都以磁盘为准，不写数据库；写锁按文件粒度生效，
        并行分支写同一路径时串行执行，写不同文件互不阻塞。
        """
        rel_path = cls.normalize_path(path)
        if not rel_path:
            raise ValueError(f"非法路径：{path}")
        dest = cls.workspace_dir(workspace_id).joinpath(*rel_path.split("/"))
        with cls.write_lock(workspace_id, rel_path):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        try:
            stat = dest.stat()
            mtime_ns, size = stat.st_mtime_ns, stat.st_size
        except OSError:
            mtime_ns, size = 0, len(content.encode("utf-8"))
        return cls._entry_for(rel_path, mtime_ns, size)

    @classmethod
    def delete(cls, workspace_id: str, path: str) -> bool:
        rel = cls.normalize_path(path)
        if not rel:
            return False
        with cls.write_lock(workspace_id, rel):
            dest = cls._abs_path(workspace_id, rel)
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
            return True

    @staticmethod
    def _entry_for(path: str, mtime_ns: int, size: int) -> dict:
        """构造轻量索引条目（不含内容，不落库）。"""
        return {
            "size": size,
            "mime": mimetypes.guess_type(path)[0] or "text/plain",
            "created_at": datetime.fromtimestamp(
                mtime_ns / 1_000_000_000, tz=timezone.utc
            ).isoformat(),
        }
    @classmethod
    def scan_index(cls, workspace_id: str) -> dict[str, dict]:
        """扫描工作区目录，返回索引 {path: {size, mime, created_at}}。

        工作区目录与索引都直接来自这里，数据库不再参与。
        """
        return {
            rel: cls._entry_for(rel, mtime_ns, size)
            for rel, (mtime_ns, size) in cls.scan_dir(workspace_id).items()
        }

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
        """对比目录前后快照，返回新增/变更的文件索引（纯内存，不写数据库）。

        before 为执行前快照（scan_dir 的返回值）；传 None 时返回全部文件。
        返回 (entries: {vpath: entry}, changed: [vpath])。
        """
        after = cls.scan_dir(workspace_id)
        entries: dict = {}
        changed: list[str] = []
        for rel, (mtime_ns, size) in after.items():
            if before is not None and before.get(rel) == (mtime_ns, size):
                continue  # 未发生变化
            entries[rel] = cls._entry_for(rel, mtime_ns, size)
            changed.append(rel)
        return entries, changed

    @classmethod
    def delete_workspace(cls, workspace_id: str) -> None:
        """删除整个工作区：绑定了外部目录时只解除绑定，绝不删除用户自己的文件夹。"""
        import shutil

        external = cls.external_dir_of(workspace_id) is not None
        cls._external_dirs.pop(workspace_id, None)
        if external:
            return
        dir_name = cls._dir_cache.pop(workspace_id, None)
        if dir_name is None:
            dir_name = cls.dir_name_of(workspace_id)
            cls._dir_cache.pop(workspace_id, None)
        target = SANDBOX_DIR / dir_name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)


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
