"""
src/multi_process_1.py

Note:
Jupyter Notebook has many limitations when running multiprocessing applications.
Therefore, in most cases, we will only include the code here and execute it from the `src` folder.
Please run the actual code by executing the file located in the `src` folder.
"""

import torch.multiprocessing as mp
# Usually the alias `mp` is used.


# Area executed concurrently in subprocesses
def fn(rank, param1, param2):
    print(f"{param1} {param2} - rank: {rank}")


# Main process
if __name__ == "__main__":
    processes = []
    # Set start method
    mp.set_start_method("spawn")

    for rank in range(4):
        process = mp.Process(target=fn, args=(rank, "A0", "B1"))
        # Create subprocess
        # process.daemon = False
        process.daemon = True
        # False means that child processes will run independently of the main process
        # and will not be terminated when the main process exits.
        # Daemon flag (terminate together when the main process exits)
        process.start()
        # Start subprocess
        processes.append(process)

    for process in processes:
        process.join()  # main process is waiting for the subprocesses to exit
        # Subprocess join (= exit after completion)

    print("Main Process is done")
