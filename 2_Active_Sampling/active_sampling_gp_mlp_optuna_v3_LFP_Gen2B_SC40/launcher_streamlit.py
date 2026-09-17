from __future__ import annotations

import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path
from urllib.request import urlopen


def runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def data_root() -> Path:
    # PyInstaller extracts/collects data under _MEIPASS in frozen mode.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return runtime_root()


def log_path() -> Path:
    return runtime_root() / "launcher.log"


def log(msg: str) -> None:
    try:
        with log_path().open("a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")
    except Exception:
        pass


def find_available_port(start: int = 8501, end: int = 8599) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No available port in range {start}-{end}")


def open_browser_when_ready(url: str, health_url: str, timeout_sec: int = 40) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urlopen(health_url, timeout=2) as resp:
                if int(getattr(resp, "status", 0)) == 200:
                    webbrowser.open(url, new=1)
                    log(f"[INFO] Browser opened: {url}")
                    return
        except Exception:
            time.sleep(0.5)
    log(f"[WARN] Browser open skipped; health check timeout: {health_url}")


def create_runtime_entry_script(root: Path) -> Path:
    entry = root / "_st_app_entry.py"
    code = (
        "from streamlit_surrogate_app import main\n"
        "main()\n"
    )
    entry.write_text(code, encoding="utf-8")
    return entry


def main() -> int:
    root = runtime_root()
    runtime_app_script = root / "_st_app_entry.py"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(data_root()) not in sys.path:
        sys.path.insert(0, str(data_root()))
    log("[INFO] Launcher started")
    log(f"[INFO] root={root}")
    log(f"[INFO] data_root={data_root()}")
    log(f"[INFO] runtime_app_script={runtime_app_script}")

    try:
        create_runtime_entry_script(root)
        log("[INFO] Generated runtime Streamlit entry script")
    except Exception as exc:
        msg = f"[ERROR] Failed to create runtime entry script: {exc}"
        print(msg)
        log(msg)
        log(traceback.format_exc())
        return 1

    # Keep runtime files in the same folder in portable mode.
    try:
        os.chdir(root)
        log(f"[INFO] cwd={Path.cwd()}")
    except Exception:
        log("[WARN] Failed to change cwd")
        pass

    port = find_available_port(8501, 8599)
    app_url = f"http://127.0.0.1:{port}"
    health_url = f"{app_url}/_stcore/health"
    log(f"[INFO] selected_port={port}")

    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ["STREAMLIT_BROWSER_SERVER_ADDRESS"] = "127.0.0.1"
    os.environ["STREAMLIT_BROWSER_SERVER_PORT"] = str(port)

    try:
        (root / "opened_url.txt").write_text(app_url + "\n", encoding="utf-8")
    except Exception:
        pass

    # Import app module so PyInstaller can collect transitive imports reliably.
    try:
        import streamlit_surrogate_app  # noqa: F401
    except Exception as exc:
        msg = f"[ERROR] Failed to import app module: {exc}"
        print(msg)
        log(msg)
        log(traceback.format_exc())
        return 1

    try:
        from streamlit.web import cli as stcli
    except Exception as exc:
        msg = f"[ERROR] Failed to import streamlit cli: {exc}"
        print(msg)
        log(msg)
        log(traceback.format_exc())
        return 1

    sys.argv = [
        "streamlit",
        "run",
        str(runtime_app_script),
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.serverAddress=127.0.0.1",
        f"--browser.serverPort={port}",
        "--browser.gatherUsageStats=false",
    ]
    log(f"[INFO] argv={' '.join(sys.argv)}")

    threading.Thread(
        target=open_browser_when_ready,
        args=(app_url, health_url),
        daemon=True,
    ).start()

    try:
        rc = stcli.main()
        # Streamlit may return None on normal control flow.
        if rc is None:
            log("[INFO] streamlit returned None; treating as rc=0")
            return 0
        log(f"[INFO] streamlit returned rc={rc}")
        return int(rc)
    except SystemExit as se:
        code = se.code if isinstance(se.code, int) else 0
        log(f"[INFO] streamlit SystemExit code={code}")
        return code
    except Exception as exc:
        msg = f"[ERROR] Runtime failure: {exc}"
        print(msg)
        log(msg)
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
