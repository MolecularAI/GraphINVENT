"""
Example submission script for a GraphINVENT fine-tuning job. This can be used to
fine-tune a pre-trained model via reinforcement learning.

To run, type:
(graphinvent) ~/GraphINVENT$ python submit-fine-tuning.py
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
DATASET          = "gdb13_1K-debug"
JOB_TYPE         = "fine-tune"         # "fine-tune", or "generate"
JOBDIR_START_IDX = 0                   # where to start indexing job dirs
N_JOBS           = 1                   # number of jobs to run per model
RESTART          = False
FORCE_OVERWRITE  = True                # overwrite job directories which already exist
JOBNAME          = "example_job_name"  # used to create a sub directory

# if running using SLURM sbatch, specify params below
USE_SLURM = False                      # use SLURM or not
RUN_TIME  = "1-00:00:00"               # hh:mm:ss
MEM_GB    = 20                         # required RAM in GB

# for SLURM jobs, set partition to run job on (preprocessing jobs run entirely on
# CPU, so no need to request GPU partition; all other job types benefit from running
# on a GPU)
if JOB_TYPE == "preprocess":
    PARTITION     = "core"
    CPUS_PER_TASK = 1
else:
    PARTITION     = "gpu"
    CPUS_PER_TASK = 4

# set paths here
HOME             = str(Path.home())
PYTHON_PATH      = f"{HOME}/miniconda3/envs/graphinvent/bin/python"
GRAPHINVENT_PATH = "./graphinvent/"
DATA_PATH        = "./data/fine-tuning/"

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# define dataset-specific parameters
params = {
    "atom_types"          : ["C", "N", "O", "S", "Cl"],  # <-- should match pre-trained model param
    "formal_charge"       : [-1, 0, +1],                 # <-- should match pre-trained model param
    "max_n_nodes"         : 13,                          # <-- should match pre-trained model param
    "job_type"            : JOB_TYPE,
    "dataset_dir"         : f"{DATA_PATH}{DATASET}/",
    "restart"             : RESTART,
    "device"              : DEVICE,
    "model"               : "GGNN",                      # <-- should match pre-trained model param
    "sample_every"        : 2,
    "init_lr"             : 1e-4,
    "epochs"              : 100,                         # <-- number of fine-tuning steps
    "batch_size"          : 64,
    "block_size"          : 1000,
    "n_workers"           : 0,
    "sigma"               : 20,                          # <-- see loss function
    "alpha"               : 0.5,                         # <-- see loss function
    "pretrained_model_dir": f"output_{DATASET}/example/job_0/",
    "generation_epoch"    : 80,                          # <-- which pre-trained model epoch to use
    "n_samples"           : 100,                         # <-- how many graphs to sample every step
    # additional paramaters can be defined here, if different from the "defaults"
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
        if USE_SLURM:
            print("* Writing submission script.", flush=True)
            write_submission_script(job_dir=params["job_dir"],
                                    job_idx=job_idx,
                                    job_type=params["job_type"],
                                    max_n_nodes=params["max_n_nodes"],
                                    runtime=RUN_TIME,
                                    mem=MEM_GB,
                                    ptn=PARTITION,
                                    cpu_per_task=CPUS_PER_TASK,
                                    python_bin_path=PYTHON_PATH)

            print("* Submitting job to SLURM.", flush=True)
            subprocess.run(["sbatch", params["job_dir"] + "submit.sh"],
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
                            runtime : str, mem : int, ptn : str, cpu_per_task : int,
                            python_bin_path : str) -> None:
    """
    Writes a submission script (`submit.sh`).

    Args:
    ----
        job_dir (str)         : Job running directory.
        job_idx (int)         : Job idx.
        job_type (str)        : Type of job to run.
        max_n_nodes (int)     : Maximum number of nodes in dataset.
        runtime (str)         : Job run-time limit in hh:mm:ss format.
        mem (int)             : Gigabytes to reserve.
        ptn (str)             : Partition to use, either "core" (CPU) or "gpu" (GPU).
        cpu_per_task (int)    : How many CPUs to use per task.
        python_bin_path (str) : Path to Python binary to use.
    """
    submit_filename = job_dir + "submit.sh"
    with open(submit_filename, "w") as submit_file:
        submit_file.write("#!/bin/bash\n")
        submit_file.write(f"#SBATCH --job-name={job_type}{max_n_nodes}_{job_idx}\n")
        submit_file.write(f"#SBATCH --output={job_type}{max_n_nodes}_{job_idx}o\n")
        submit_file.write(f"#SBATCH --time={runtime}\n")
        submit_file.write(f"#SBATCH --mem={mem}g\n")
        submit_file.write(f"#SBATCH --partition={ptn}\n")
        submit_file.write("#SBATCH --nodes=1\n")
        submit_file.write(f"#SBATCH --cpus-per-task={cpu_per_task}\n")
        if ptn == "gpu":
            submit_file.write("#SBATCH --gres=gpu:1\n")
        submit_file.write("hostname\n")
        submit_file.write("export QT_QPA_PLATFORM='offscreen'\n")
        submit_file.write(f"{python_bin_path} {GRAPHINVENT_PATH}main.py --job-dir {job_dir}")
        submit_file.write(f" > {job_dir}output.o${{SLURM_JOB_ID}}\n")


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
