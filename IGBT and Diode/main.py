import subprocess
import sys

def run_once(entry_script: str) -> None:
    cmd = [sys.executable, "-u", entry_script]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":


    print("runner_1: Simulation_1")
    run_once("Runners/runner_1.py")
    print('##################################')

    print("runner_0: Simulation_2")
    run_once("Runners/runner_0.py")
    print('##################################')

    '''

    print("runner_02U_1: Simulation_1")
    run_once("Runners/runner_02U_1.py")
    print('##################################')

    print("runner_02U_2: Simulation_2")
    run_once("Runners/runner_02U_2.py")
    print('##################################')

    print("runner_02U_3: Simulation_3")
    run_once("Runners/runner_02U_3.py")
    print('##################################')

    print("runner_02U_4: Simulation_4")
    run_once("Runners/runner_02U_4.py")
    print('##################################')

    print("runner_04U_1: Simulation_5")
    run_once("Runners/runner_04U_1.py")
    print('##################################')

    print("runner_04U_2: Simulation_6")
    run_once("Runners/runner_04U_2.py")
    print('##################################')

    print("runner_04U_3: Simulation_7")
    run_once("Runners/runner_04U_3.py")
    print('##################################')

    print("runner_04U_4: Simulation_8")
    run_once("Runners/runner_04U_4.py")
    print('##################################')

    print("runner_06U_1: Simulation_9")
    run_once("Runners/runner_06U_1.py")
    print('##################################')

    print("runner_06U_2: Simulation_10")
    run_once("Runners/runner_06U_2.py")
    print('##################################')

    print("runner_06U_3: Simulation_11")
    run_once("Runners/runner_06U_3.py")
    print('##################################')

    print("runner_06U_4: Simulation_12")
    run_once("Runners/runner_06U_4.py")
    print('##################################')

    print("runner_08U_1: Simulation_13")
    run_once("Runners/runner_08U_1.py")
    print('##################################')

    print("runner_08U_2: Simulation_14")
    run_once("Runners/runner_08U_2.py")
    print('##################################')

    print("runner_08U_3: Simulation_15")
    run_once("Runners/runner_08U_3.py")
    print('##################################')

    print("runner_08U_4: Simulation_16")
    run_once("Runners/runner_08U_4.py")
    print('##################################')

    print("runner_10U_1: Simulation_17")
    run_once("Runners/runner_10U_1.py")
    print('##################################')

    print("runner_10U_2: Simulation_18")
    run_once("Runners/runner_10U_2.py")
    print('##################################')

    print("runner_10U_3: Simulation_19")
    run_once("Runners/runner_10U_3.py")
    print('##################################')

    print("runner_10U_4: Simulation_20")
    run_once("Runners/runner_10U_4.py")
    print('##################################')
    
    '''





