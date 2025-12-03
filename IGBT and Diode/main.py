import subprocess
import sys

def run_once(entry_script: str) -> None:
    cmd = [sys.executable, "-u", entry_script]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":

    print("pf", 1)
    run_once("Runners/runner_1.py")
    print('##################################')

    print("pf", 0)
    run_once("Runners/runner_0.py")
    print('##################################')

    print("pf", 0.9)
    run_once("Runners/runner_0.9.py")
    print('##################################')

    print("pf", -0.9)
    run_once("Runners/runner_-0.9.py")
    print('##################################')

    print("pf", 0.8)
    run_once("Runners/runner_0.8.py")
    print('##################################')

    print("pf", -0.8)
    run_once("Runners/runner_-0.8.py")
    print('##################################')

    print("pf", 0.7)
    run_once("Runners/runner_0.7.py")
    print('##################################')

    print("pf", -0.7)
    run_once("Runners/runner_-0.7.py")
    print('##################################')

    print("pf", 0.6)
    run_once("Runners/runner_0.6.py")
    print('##################################')

    print("pf", -0.6)
    run_once("Runners/runner_-0.6.py")
    print('##################################')

    print("pf", 0.5)
    run_once("Runners/runner_0.5.py")
    print('##################################')

    print("pf", -0.5)
    run_once("Runners/runner_-0.5.py")
    print('##################################')

    print("pf", 0.4)
    run_once("Runners/runner_0.4.py")
    print('##################################')

    print("pf", -0.4)
    run_once("Runners/runner_-0.4.py")
    print('##################################')

    print("pf", 0.3)
    run_once("Runners/runner_0.3.py")
    print('##################################')

    print("pf", -0.3)
    run_once("Runners/runner_-0.3.py")
    print('##################################')

    print("pf", 0.2)
    run_once("Runners/runner_0.2.py")
    print('##################################')

    print("pf", -0.2)
    run_once("Runners/runner_-0.2.py")
    print('##################################')

    print("pf", 0.1)
    run_once("Runners/runner_0.1.py")
    print('##################################')

    print("pf", -0.1)
    run_once("Runners/runner_-0.1.py")
    print('##################################')
    

