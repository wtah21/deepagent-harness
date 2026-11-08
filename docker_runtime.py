"""
DeepAgent Harness 的 Docker 隔离执行运行时。
每个项目（thread_id）一个常驻容器：--cpus=1.0 -m 1g，
把本地 sandbox/<项目>_<thread前8位> 目录挂载为容器内 /app/workspace，
执行时通过 docker exec 在容器内跑 Python 代码 / shell 命令。

设计要点：
- 容器常驻（tail -f /dev/null），执行走 docker exec，免去反复起停容器；
- 工作区目录即挂载卷：容器内写文件 = 宿主机 sandbox 目录写文件，
  产物不需要额外拷贝，由 persistence.WorkspaceStore.sync_dir 回收元数据；
- 所有 Docker 调用失败都抛 DockerError（不静默），由调用方决定是否回退本地沙盒。
"""
import subprocess
from pathlib import Path

from config import CONTAINER_PREFIX, DOCKER_CPUS, DOCKER_IMAGE, DOCKER_MEMORY
from persistence import WorkspaceStore

# 容器内固定的工作目录 / 挂载点
CONTAINER_WORKDIR = "/app/workspace"

# 容器归属标签：本项目创建的所有容器统一打标，服务退出时按标签批量停止，
# 避免常驻容器在应用关闭后继续驻留内存
LABEL_MANAGED = "harness.managed=true"  # 标识「由本项目管理」的容器
LABEL_PROJECT = "harness.project"       # 值为项目目录名，标识容器归属的项目


class DockerError(RuntimeError):
    """Docker 不可用或命令执行失败。"""


def _docker(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    """执行一条 docker CLI 命令；CLI 缺失/超时统一抛 DockerError。"""
    try:
        return subprocess.run(
            ["docker", *args],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as e:
        raise DockerError("docker CLI 未安装或不在 PATH 中") from e
    except subprocess.TimeoutExpired as e:
        raise DockerError(f"docker {' '.join(args[:2])} 执行超时（{timeout}s）") from e


def is_docker_available() -> bool:
    """探测 Docker daemon 是否可连接（连接失败视为不可用，调用方回退本地沙盒）。"""
    try:
        return _docker(["version", "--format", "{{.Server.Version}}"], timeout=10).returncode == 0
    except DockerError:
        return False


def container_name(workspace_id: str) -> str:
    """workspace（thread_id）→ 容器名：harness_ws_<项目目录名>。"""
    return f"{CONTAINER_PREFIX}{WorkspaceStore.dir_name_of(workspace_id)}"


def _mount_spec(workspace_id: str) -> str:
    """构造卷挂载参数：宿主机 sandbox 项目目录（正斜杠形式）:/app/workspace。"""
    src = WorkspaceStore.workspace_dir(workspace_id)
    return f"{Path(src).as_posix()}:{CONTAINER_WORKDIR}"


def _ensure_image() -> None:
    """本地缺镜像时先拉取。"""
    if _docker(["image", "inspect", DOCKER_IMAGE], timeout=15).returncode == 0:
        return
    pull = _docker(["pull", DOCKER_IMAGE], timeout=600)
    if pull.returncode != 0:
        raise DockerError(f"拉取镜像 {DOCKER_IMAGE} 失败：{pull.stderr.strip()[:300]}")


def get_or_create_container(workspace_id: str) -> str:
    """获取或创建该项目的常驻容器，返回容器名。

    已存在且运行中 → 直接返回；已存在但停止 → docker start；
    不存在 → 按规格创建（1 CPU / 1GB 内存 / 挂载工作区 / 工作目录 /app/workspace）。
    """
    name = container_name(workspace_id)
    inspect = _docker(["inspect", "-f", "{{.State.Running}}", name], timeout=15)
    if inspect.returncode == 0:
        if inspect.stdout.strip() == "true":
            return name
        start = _docker(["start", name], timeout=60)
        if start.returncode != 0:
            raise DockerError(f"启动容器 {name} 失败：{start.stderr.strip()[:300]}")
        return name

    _ensure_image()
    run = _docker(
        [
            "run", "-d",
            "--name", name,
            "--cpus", DOCKER_CPUS,
            "-m", DOCKER_MEMORY,
            "--label", LABEL_MANAGED,
            "--label", f"{LABEL_PROJECT}={WorkspaceStore.dir_name_of(workspace_id)}",
            "-v", _mount_spec(workspace_id),
            "-w", CONTAINER_WORKDIR,
            "-e", "PYTHONUNBUFFERED=1",
            DOCKER_IMAGE,
            "tail", "-f", "/dev/null",
        ],
        timeout=120,
    )
    if run.returncode != 0:
        raise DockerError(f"创建容器 {name} 失败：{run.stderr.strip()[:300]}")
    return name


def exec_in_workspace(
    workspace_id: str,
    payload: str,
    is_shell: bool = False,
    timeout: int = 30,
) -> str:
    """在该项目的容器内执行 Python 代码或 shell 命令，返回输出文本。

    用户代码的非零退出码属于正常业务输出（附在结果里），
    只有 Docker 自身故障（连不上、容器没了等）才抛 DockerError。
    """
    name = get_or_create_container(workspace_id)
    if is_shell:
        args = ["exec", "-w", CONTAINER_WORKDIR, name, "sh", "-c", payload]
    else:
        args = ["exec", "-w", CONTAINER_WORKDIR, name, "python", "-c", payload]

    try:
        result = _docker(args, timeout=timeout)
    except DockerError as e:
        if "超时" in str(e):
            # 超时后重启容器以终止仍在跑的 exec 进程（尽力而为）
            _docker(["restart", name], timeout=60)
        raise

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    code = result.returncode

    # docker 自身故障的典型退出码：125（daemon/运行时错误）、126（不可执行）
    if code in (125, 126):
        raise DockerError(f"docker exec 失败（exit={code}）：{stderr[:300]}")

    lines = []
    if stdout:
        lines.append(f"STDOUT:\n{stdout}")
    if stderr:
        lines.append(f"STDERR:\n{stderr}")
    if code not in (0, 127) and not stderr:
        # 非零退出且无 stderr 时显式标注退出码
        lines.append(f"[exit code: {code}]")
    return "\n\n".join(lines) if lines else "(no output)"


def remove_container(workspace_id: str) -> bool:
    """删除该项目的容器（强制），用于项目删除/重命名时的清理。忽略不存在的情况。"""
    try:
        result = _docker(["rm", "-f", container_name(workspace_id)], timeout=30)
        return result.returncode == 0
    except DockerError:
        return False


def stop_container(workspace_id: str) -> bool:
    """停止该项目的容器（不删除）。"""
    try:
        result = _docker(["stop", container_name(workspace_id)], timeout=60)
        return result.returncode == 0
    except DockerError:
        return False


def stop_all_managed_containers() -> int:
    """停止所有由本项目创建的容器（按 harness.managed=true 标签过滤）。

    返回实际停止的容器数量；Docker 不可用或无匹配容器时静默返回 0。
    tail -f 不会响应 SIGTERM，用 -t 1 让每个容器最多优雅等待 1 秒后强停。
    """
    try:
        ps = _docker(["ps", "-q", "--filter", f"label={LABEL_MANAGED}"], timeout=30)
    except DockerError:
        return 0
    if ps.returncode != 0 or not ps.stdout.strip():
        return 0
    ids = ps.stdout.split()
    try:
        stop = _docker(["stop", "-t", "1", *ids], timeout=180)
    except DockerError:
        return 0
    return len(ids) if stop.returncode == 0 else 0


_cleanup_registered = False


def register_exit_cleanup() -> None:
    """注册进程退出钩子：应用退出时停止本项目所有常驻容器，避免内存泄漏。

    只在 Streamlit 主进程入口调用；parallel_worker 子进程虽 import 本模块，
    但不注册钩子，避免子进程结束时误停主进程正在使用的容器。重复调用幂等。
    """
    global _cleanup_registered
    if _cleanup_registered:
        return
    import atexit

    atexit.register(stop_all_managed_containers)
    _cleanup_registered = True
