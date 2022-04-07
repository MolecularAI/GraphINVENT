"""
Example submission script for a GraphINVENT training job (distribution-
based training, not fine-tuning/optimization job). This can be used to
pre-train a model before a fine-tuning (via reinforcement learning) job.

To run, type:
(graphinvent) ~/GraphINVENT$ python submit-pre-training.py

This script was modified to run on the MIT Supercloud.
"""
# load general packages and functions
import csv
import sys
import os
from pathlib import Path
import subprocess
import time
import torch


# define what you want to do for the specified job(s)
DATASET          = "gdb13_1K-debug"    # dataset name in "./data/pre-training/"
JOB_TYPE         = "train"             # "preprocess", "train", "generate", or "test"
JOBDIR_START_IDX = 0                   # where to start indexing job dirs
N_JOBS           = 1                   # number of jobs to run per model
RESTART          = False               # whether or not this is a restart job
FORCE_OVERWRITE  = True                # overwrite job directories which already exist
JOBNAME          = "example-job-name"  # used to create a sub directory

# if running using LLsub, specify params below
USE_LLSUB = True                       # use LLsub or not
MEM_GB    = 20                         # required RAM in GB

# for LLsub jobs, set number of CPUs per task
if JOB_TYPE == "preprocess":
    CPUS_PER_TASK = 1
    DEVICE = "cpu"
else:
    CPUS_PER_TASK = 10
    DEVICE = "cuda"

# set paths here
HOME             = str(Path.home())
PYTHON_PATH      = f"{HOME}/path/to/graphinvent/bin/python"
GRAPHINVENT_PATH = "./graphinvent/"
DATA_PATH        = "./data/pre-training/"

# define dataset-specific parameters
params = {
    "atom_types"   : ["C", "N", "O", "S", "Cl"],
    "formal_charge": [-1, 0, +1],
    "max_n_nodes"  : 13,
    "job_type"     : JOB_TYPE,
    "dataset_dir"  : f"{DATA_PATH}{DATASET}/",
    "restart"      : RESTART,
    "model"        : "GGNN",
    "sample_every" : 2,
    "init_lr"      : 1e-4,
    "epochs"       : 100,
    "batch_size"   : 50,
    "block_size"   : 1000,
    "device"       : DEVICE,
    "n_samples"    : 100,
    # additional paramaters can be defined here, if different from the "defaults"
    # for instance, for "generate" jobs, don't forget to specify "generation_epoch"
    # and "n_samples"
}


def submit() -> None:
    """
    Creates and submits submission script. Uses global variables defined at top
    of this file.
    """
    check_paths()

    # create an output directory
    dataset_output_path = f"{HOME}/GraphINVENT/output_{DATASET}"
    tensorboard_path    = os.path.join(dataset_output_path, "tensorboard")
    if JOBNAME != "":
        dataset_output_path = os.path.join(dataset_output_path, JOBNAME)
        tensorboard_path    = os.path.join(tensorboard_path, JOBNAME)

    os.makedirs(dataset_output_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)
    print(f"* Creating dataset directory {dataset_output_path}/", flush=True)

    # submit `N_JOBS` separate jobs
    jobdir_end_idx = JOBDIR_START_IDX + N_JOBS
    for job_idx in range(JOBDIR_START_IDX, jobdir_end_idx):

        # specify and create the job subdirectory if it does not exist
        params["job_dir"]         = f"{dataset_output_path}/job_{job_idx}/"
        params["tensorboard_dir"] = f"{tensorboard_path}/job_{job_idx}/"

        # create the directory if it does not exist already, otherwise raises an
        # error, which is good because *might* not want to override data our
        # existing directories!
        os.makedirs(params["tensorboard_dir"], exist_ok=True)
        try:
            job_dir_exists_already = bool(
                JOB_TYPE in ["generate", "test"] or FORCE_OVERWRITE
            )
            os.makedirs(params["job_dir"], exist_ok=job_dir_exists_already)
            print(
                f"* Creating model subdirectory {dataset_output_path}/job_{job_idx}/",
                flush=True,
            )
        except FileExistsError:
            print(
                f"-- Model subdirectory {dataset_output_path}/job_{job_idx}/ already exists.",
                flush=True,
            )
            if not RESTART:
                continue

        # write the `input.csv` file
        write_input_csv(params_dict=params, filename="input.csv")

        # write `submit.sh` and submit
        if USE_LLSUB:
            print("* Writing submission script.", flush=True)
            write_submission_script(job_dir=params["job_dir"],
                                    job_idx=job_idx,
                                    job_type=params["job_type"],
                                    max_n_nodes=params["max_n_nodes"],
                                    cpu_per_task=CPUS_PER_TASK,
                                    python_bin_path=PYTHON_PATH)

            print("* Submitting batch job using LLsub.", flush=True)
            subprocess.run(["LLsub", params["job_dir"] + "submit.sh"],
                           check=True)
        else:
            print("* Running job as a normal process.", flush=True)
            subprocess.run(["ls", f"{PYTHON_PATH}"], check=True)
            subprocess.run([f"{PYTHON_PATH}",
                            f"{GRAPHINVENT_PATH}main.py",
                            "--job-dir",
                            params["job_dir"]],
                           check=True)

        # sleep a few secs before submitting next job
        print("-- Sleeping 2 seconds.")
        time.sleep(2)


def write_input_csv(params_dict : dict, filename : str="params.csv") -> None:
    """
    Writes job parameters/hyperparameters in `params_dict` to CSV using the specified
    `filename`.
    """
    dict_path = params_dict["job_dir"] + filename

    with open(dict_path, "w") as csv_file:

        writer = csv.writer(csv_file, delimiter=";")
        for key, value in params_dict.items():
            writer.writerow([key, value])


def write_submission_script(job_dir : str, job_idx : int, job_type : str, max_n_nodes : int,
                            cpu_per_task : int, python_bin_path : str) -> None:
    """
    Writes a submission script (`submit.sh`).

    Args:
    ----
        job_dir (str)         : Job running directory.
        job_idx (int)         : Job idx.
        job_type (str)        : Type of job to run.
        max_n_nodes (int)     : Maximum number of nodes in dataset.
        cpu_per_task (int)    : How many CPUs to use per task.
        python_bin_path (str) : Path to Python binary to use.
    """
    submit_filename = job_dir + "submit.sh"
    with open(submit_filename, "w") as submit_file:
        submit_file.write("#!/bin/bash\n")
        submit_file.write(f"#SBATCH --job-name={job_type}{max_n_nodes}_{job_idx}\n")
        submit_file.write(f"#SBATCH --output={job_type}{max_n_nodes}_{job_idx}o\n")
        submit_file.write(f"#SBATCH --cpus-per-task={cpu_per_task}\n")
        if DEVICE == "cuda":
            submit_file.write("#SBATCH --gres=gpu:volta:1\n")
        submit_file.write("hostname\n")
        submit_file.write("export QT_QPA_PLATFORM='offscreen'\n")
        submit_file.write(f"{python_bin_path} {GRAPHINVENT_PATH}main.py --job-dir {job_dir}")
        submit_file.write(f" > {job_dir}output.o${{LLSUB_RANK}}\n")


def check_paths() -> None:
    """
    Checks that paths to Python binary, data, and GraphINVENT are properly
    defined before running a job, and tells the user to define them if not.
    """
    for path in [PYTHON_PATH, GRAPHINVENT_PATH, DATA_PATH]:
        if "path/to/" in path:
            print("!!!")
            print("* Update the following paths in `submit.py` before running:")
            print("-- `PYTHON_PATH`\n-- `GRAPHINVENT_PATH`\n-- `DATA_PATH`")
            sys.exit(0)

if __name__ == "__main__":
    submit()
