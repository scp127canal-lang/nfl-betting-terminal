"""Arranca la API FastAPI (Uvicorn) en segundo plano y la interfaz Streamlit.

Uso:  python run.py
Puertos: API_PORT (8000) y UI_PORT (8501). Si están ocupados, usa el siguiente libre.
Ctrl+C detiene ambos procesos.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((HOST, port)) != 0


def pick_port(preferred: int, tries: int = 20) -> int:
    for port in range(preferred, preferred + tries):
        if port_free(port):
            return port
    raise RuntimeError(f"Sin puertos libres entre {preferred} y {preferred + tries - 1}")


def wait_for(url: str, proc: subprocess.Popen, timeout: float = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(1)
    return False


def stop(proc: subprocess.Popen | None) -> None:
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    api_port = pick_port(int(os.getenv("API_PORT", "8000")))
    ui_port = pick_port(int(os.getenv("UI_PORT", "8501")))
    api_url = f"http://{HOST}:{api_port}"
    env = {**os.environ, "NFL_API_URL": api_url, "PYTHONUTF8": "1"}

    print(f"[run] API FastAPI  -> {api_url}  (docs: {api_url}/docs)", flush=True)
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", HOST, "--port", str(api_port)],
        cwd=ROOT, env=env)
    ui = None
    try:
        if wait_for(f"{api_url}/health", api):
            print("[run] API lista.", flush=True)
        else:
            print("[run] AVISO: la API no respondió a /health; la interfaz arranca igualmente.", flush=True)

        print(f"[run] Terminal Streamlit -> http://{HOST}:{ui_port}", flush=True)
        ui = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py",
             "--server.address", HOST, "--server.port", str(ui_port)],
            cwd=ROOT, env=env)
        return ui.wait()
    except KeyboardInterrupt:
        print("\n[run] Deteniendo...", flush=True)
        return 0
    finally:
        stop(ui)
        stop(api)


if __name__ == "__main__":
    sys.exit(main())
