import subprocess
import sys

def run_once(entry_script: str) -> None:
    cmd = [sys.executable, "-u", entry_script]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_once("Runners/runner_1.py")
