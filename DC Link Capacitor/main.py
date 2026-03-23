import subprocess
import sys

def run_once(entry_script: str) -> None:
    cmd = [sys.executable, "-u", entry_script]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":

    pf_values = [1, 0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6,0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]

    for i, pf in enumerate(pf_values, start=1):
        print(f"runner_{pf}: Simulation_{i}")
        run_once(f"Runners/runner_{pf}.py")
        print("##################################")




