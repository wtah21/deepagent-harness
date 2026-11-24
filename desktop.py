"""
DeepAgent Harness 桌面入口。

用 pywebview 起一个轻量桌面窗口承载 Streamlit 界面：
- 先选一个空闲端口启动 Streamlit 子进程；
- 轮询 /_stcore/health 等端口就绪后再打开窗口；
- 关闭窗口或进程退出时终止 Streamlit 子进程，避免残留服务与内存占用。

用法：python desktop.py
"""
import atexit
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_SCRIPT = ROOT / "streamlit_app.py"


def _free_port() -> int:
    """让系统分配一个空闲端口，避免和已占用的 8501 冲突。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_ready(url: str, timeout: float = 60.0) -> bool:
    """轮询 Streamlit 健康检查接口，直到服务可以接受页面请求。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def main() -> int:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", str(APP_SCRIPT),
            "--server.port", str(port),
            "--server.address", "127.0.0.1",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=str(ROOT),
    )

    def cleanup() -> None:
        """关闭桌面窗口时一并结束 Streamlit 子进程（含其进程树）。"""
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    atexit.register(cleanup)
    try:
        if not _wait_until_ready(f"{url}/_stcore/health"):
            print("Streamlit 服务启动超时，请检查依赖是否安装完整。", file=sys.stderr)
            return 1
        import webview

        webview.create_window("DeepAgent Harness", url, width=1440, height=900)
        webview.start()
    finally:
        cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
