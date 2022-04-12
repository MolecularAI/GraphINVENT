"""
Example submission script for a GraphINVENT preprocessing job (before distribution-
based training, not fine-tuning/optimization job), when the dataset is large and
we want to split one large preprocessing job into multiple smaller preprocessing
jobs, aggregating the final HDF files at the end. The HDF dataset created can be
used to pre-train a model before a fine-tuning (via reinforcement learning) job.

To run, you can first split the dataset as follows (do this within an interactive session):
(graphinvent)$ python submit-split-preprocessing-supercloud.py --type split

Then, submit the separate preprocessing jobs for the split dataset as follows:
(graphinvent)$ python submit-split-preprocessing-supercloud.py --type submit

When the above jobs have completed, aggregate the generated HDFs for each dataset split into the main dataset dir:
(graphinvent)$ python submit-split-preprocessing-supercloud.py --type aggregate

The above script also cleans up extra files.

This script was modified to run on the MIT Supercloud.
"""
# load general packages and functions
import csv
import argparse
import sys
import os
import shutil
from pathlib import Path
import subprocess
import time
from math import ceil
from typing import Union
import numpy as np
import h5py
import torch

# define the argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)

# define potential arguments for using this script 
parser.add_argument("--type",
                    type=str,
                    default="split",
                    help="Acceptable values include 'split', 'submit', 'aggregate', and 'cleanup'.")
args = parser.parse_args()

# define what you want to do for the specified job(s)
DATASET          = "gdb13_1K-debug"  # dataset name in "./data/pre-training/"
JOB_TYPE         = "preprocess"      # "preprocess", "train", "generate", or "test"
JOBDIR_START_IDX = 0                 # where to start indexing job dirs
N_JOBS           = 1                 # number of jobs to run per model
RESTART          = False             # whether or not this is a restart job
FORCE_OVERWRITE  = True              # overwrite job directories which already exist
JOBNAME          = "preprocessing"   # used to create a sub directory

# if running using LLsub, specify params below
USE_LLSUB = True                     # use LLsub or not
MEM_GB    = 20                       # required RAM in GB

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
        # error, which is good because *might* not want to override data in our
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

def split_file(filename : str, n_lines_per_split : int=100000) -> None:
    """
    _summary_

    Args:
    ----
        filename (str)          : The filename.
        n_lines_per_split (int) : Number of lines per file.

    Returns:
    -------
        n_splits (int)          : Number of splits.
    """
    output_base = filename[:-4]
    input       = open(filename, "r")
    extension   = filename[-3:]

    count = 0
    at    = 0
    dest  = None
    for line in input:
        if count % n_lines_per_split == 0:
            if dest: dest.close()
            dest = open(f"{output_base}.{at}.{extension}", "w")
            at += 1
        dest.write(line)
        count += 1

    n_splits = at
    return n_splits

def get_n_splits(filename : str, n_lines_per_split : int=100000) -> None:
    """
    _summary_

    Args:
    ----
        filename (str)          : The filename.
        n_lines_per_split (int) : Number of lines per file.

    Returns:
    -------
        n_splits (int)          : Number of splits.
    """
    output_base = filename[:-4]
    input       = open(filename, "r")
    n_splits    = ceil(len(input) / n_lines_per_split)
    return n_splits

def load_ts_properties_from_csv(csv_path : str) -> Union[dict, None]:
    """
    Loads CSV file containing training set properties and returns contents as a dictionary.
    """
    print("* Loading training set properties.", flush=True)

    # read dictionaries from csv
    try:
        with open(csv_path, "r") as csv_file:
            reader   = csv.reader(csv_file, delimiter=";")
            csv_dict = dict(reader)
    except:
        return None

    # fix file types within dict in going from `csv_dict` --> `properties_dict`
    properties_dict = {}
    for key, value in csv_dict.items():

        # first determine if key is a tuple
        key = eval(key)
        if len(key) > 1:
            tuple_key = (str(key[0]), str(key[1]))
        else:
            tuple_key = key

        # then convert the values to the correct data type
        try:
            properties_dict[tuple_key] = eval(value)
        except (SyntaxError, NameError):
            properties_dict[tuple_key] = value

        # convert any `list`s to `torch.Tensor`s (for consistency)
        if type(properties_dict[tuple_key]) == list:
            properties_dict[tuple_key] = torch.Tensor(properties_dict[tuple_key])

    return properties_dict

def write_ts_properties_to_csv(ts_properties_dict : dict, split : str) -> None:
    """
    Writes the training set properties in `ts_properties_dict` to a CSV file.
    """
    dict_path = f"data/{dataset}/{split}.csv"

    with open(dict_path, "w") as csv_file:

        csv_writer = csv.writer(csv_file, delimiter=";")
        for key, value in ts_properties_dict.items():
            if "validity_tensor" in key:
                continue  # skip writing the validity tensor because it is really long
            elif type(value) == np.ndarray:
                csv_writer.writerow([key, list(value)])
            elif type(value) == torch.Tensor:
                try:
                    csv_writer.writerow([key, float(value)])
                except ValueError:
                    csv_writer.writerow([key, [float(i) for i in value]])
            else:
                csv_writer.writerow([key, value])

def get_dims() -> dict:
    """
    Gets the dims corresponding to the three datasets in each preprocessed HDF
    file: "nodes", "edges", and "APDs".
    """
    dims = {}
    dims["nodes"] = [max_n_nodes, n_atom_types + n_formal_charges]
    dims["edges"] = [max_n_nodes, max_n_nodes, n_bond_types]
    dim_f_add     = [max_n_nodes, n_atom_types, n_formal_charges, n_bond_types]
    dim_f_conn    = [max_n_nodes, n_bond_types]
    dims["APDs"]  = [np.prod(dim_f_add) + np.prod(dim_f_conn) + 1]

    return dims

def get_total_n_subgraphs(paths : list) -> int:
    """
    Gets the total number of subgraphs saved in all the HDF files in the `paths`,
    where `paths` is a list of strings containing the path to each HDF file we want
    to combine.
    """
    total_n_subgraphs = 0
    for path in paths:
        print("path:", path)
        hdf_file           = h5py.File(path, "r")
        nodes              = hdf_file.get("nodes")
        n_subgraphs        = nodes.shape[0]
        total_n_subgraphs += n_subgraphs
        hdf_file.close()

    return total_n_subgraphs

def combine_HDFs(paths : list, training_set : bool, split : str) -> None:
    """
    Combine many small HDF files (their paths defined in `paths`) into one large 
    HDF file. Works assuming HDFs were created for the preprocessed dataset 
    following the following directory structure:
    data/
     |-- {dataset}_1/
     |-- {dataset}_2/
     |-- {dataset}_3/
     |...
     |-- {dataset}_{n_dirs}/
    """
    total_n_subgraphs = get_total_n_subgraphs(paths)
    dims              = get_dims()

    print(f"* Creating HDF file to contain {total_n_subgraphs} subgraphs")
    new_hdf_file = h5py.File(f"data/{dataset}/{split}.h5", "a")
    new_dataset_nodes = new_hdf_file.create_dataset("nodes",
                                                    (total_n_subgraphs, *dims["nodes"]),
                                                    dtype=np.dtype("int8"))
    new_dataset_edges = new_hdf_file.create_dataset("edges",
                                                    (total_n_subgraphs, *dims["edges"]),
                                                    dtype=np.dtype("int8"))
    new_dataset_APDs  = new_hdf_file.create_dataset("APDs",
                                                    (total_n_subgraphs, *dims["APDs"]),
                                                    dtype=np.dtype("int8"))

    print("* Combining data from smaller HDFs into a new larger HDF.")
    init_index = 0
    for path in paths:
        print("path:", path)
        hdf_file = h5py.File(path, "r")

        nodes = hdf_file.get("nodes")
        edges = hdf_file.get("edges")
        APDs  = hdf_file.get("APDs")

        n_subgraphs = nodes.shape[0]

        new_dataset_nodes[init_index:(init_index + n_subgraphs)] = nodes
        new_dataset_edges[init_index:(init_index + n_subgraphs)] = edges
        new_dataset_APDs[init_index:(init_index + n_subgraphs)]  = APDs

        init_index += n_subgraphs
        hdf_file.close()

    new_hdf_file.close()

    if training_set:
        print(f"* Combining data from respective `{split}.csv` files into one.")
        csv_list = [f"{path[:-2]}csv" for path in paths]

        ts_properties_old = None
        csv_files_processed = 0
        for path in csv_list:
            ts_properties     = load_ts_properties_from_csv(csv_path=path)
            ts_properties_new = {}
            if ts_properties_old and ts_properties:
                for key, value in ts_properties_old.items():
                    if type(value) == float:
                        ts_properties_new[key] = (
                            value * csv_files_processed + ts_properties[key]
                        )/(csv_files_processed + 1)
                    else:
                        new_list = []
                        for i, value_i in enumerate(value):
                            new_list.append(
                                float(
                                    value_i * csv_files_processed + ts_properties[key][i]
                                )/(csv_files_processed + 1)
                            )
                        ts_properties_new[key] = new_list
            else:
                ts_properties_new = ts_properties
            ts_properties_old = ts_properties_new
            csv_files_processed += 1

        write_ts_properties_to_csv(ts_properties_dict=ts_properties_new, split=split)

if __name__ == "__main__":
    dataset = DATASET

    if args.type == "split":
        # --------- SPLIT THE DATASET ----------
        # 1) first, split the training set
        n_training_splits = split_file(filename=f"{DATA_PATH}{DATASET}/train.smi",
                                       n_lines_per_split=100000)

        # 2) then, split the test set (if necessary)
        n_test_splits     = split_file(filename=f"{DATA_PATH}{DATASET}/test.smi",
                                       n_lines_per_split=100000)

        # 3) finally, split the validation set (if necessary)
        n_valid_splits    = split_file(filename=f"{DATA_PATH}{DATASET}/valid.smi",
                                       n_lines_per_split=100000)

    elif args.type == "submit":
        # ---------- MOVE EACH SPLIT INTO ITS OWN DIRECTORY AND SUBMIT EACH AS SEPARATE JOB ----------
        # first get the number of splits for each train/test/valid split if each 
        # file is split into files of max 100000 lines
        n_training_splits = get_n_splits(filename=f"{DATA_PATH}{DATASET}/train.smi",
                                         n_lines_per_split=100000)
        n_test_splits     = get_n_splits(filename=f"{DATA_PATH}{DATASET}/test.smi",
                                         n_lines_per_split=100000)
        n_valid_splits    = get_n_splits(filename=f"{DATA_PATH}{DATASET}/valid.smi",
                                         n_lines_per_split=100000)

        # 1) train set
        for split_idx in range(n_training_splits):
            if not os.path.exists(f"{DATA_PATH}{dataset}_{split_idx}/"):
                os.mkdir(f"{DATA_PATH}{dataset}_{split_idx}/")  # make the dir
            os.rename(f"{DATA_PATH}{dataset}/train.{split_idx}.smi", f"{DATA_PATH}{dataset}_{split_idx}/train.smi")  # move the file to the dir and rename

            DATASET               = f"{dataset}_{split_idx}/"
            params["dataset_dir"] = f"{DATA_PATH}{DATASET}"
            submit()

        # 2) test set
        for split_idx in range(n_test_splits):
            if not os.path.exists(f"{DATA_PATH}{dataset}_{split_idx}/"):
                os.mkdir(f"{DATA_PATH}{dataset}_{split_idx}/")  # make the dir
            os.rename(f"{DATA_PATH}{dataset}/test.{split_idx}.smi", f"{DATA_PATH}{dataset}_{split_idx}/test.smi")  # move the file to the dir and rename

            DATASET               = f"{dataset}_{split_idx}/"
            params["dataset_dir"] = f"{DATA_PATH}{DATASET}"
            submit()

        # 3) valid set
        for split_idx in range(n_valid_splits):
            if not os.path.exists(f"{DATA_PATH}{dataset}_{split_idx}/"):
                os.mkdir(f"{DATA_PATH}{dataset}_{split_idx}/")  # make the dir
            os.rename(f"{DATA_PATH}{dataset}/valid.{split_idx}.smi", f"{DATA_PATH}{dataset}_{split_idx}/valid.smi")  # move the file to the dir and rename

            DATASET               = f"{dataset}_{split_idx}/"
            params["dataset_dir"] = f"{DATA_PATH}{DATASET}"
            submit()

    elif args.type == "aggregate":
        # first get the number of splits for each train/test/valid split if each 
        # file is split into files of max 100000 lines
        n_training_splits = get_n_splits(filename=f"{DATA_PATH}{DATASET}/train.smi",
                                         n_lines_per_split=100000)
        n_test_splits     = get_n_splits(filename=f"{DATA_PATH}{DATASET}/test.smi",
                                         n_lines_per_split=100000)
        n_valid_splits    = get_n_splits(filename=f"{DATA_PATH}{DATASET}/valid.smi",
                                         n_lines_per_split=100000)
        # ---------- AGGREGATE THE RESULTS ----------
        # set variables
        n_atom_types     = len(params["atom_types"])     # number of atom types used in preprocessing the data
        n_formal_charges = len(params["formal_charge"])  # number of formal charges used in preprocessing the data
        n_bond_types     = 3                             # number of bond types used in preprocessing the data
        max_n_nodes      = len(params["max_n_nodes"])    # maximum number of nodes in the data

        # 1) combine the training files
        path_list        = [f"data/{dataset}_{i}/train.h5" for i in range(0, n_training_splits)]
        combine_HDFs(path_list, training_set=True, split="train")

        # 2) combine the test files
        path_list        = [f"data/{dataset}_{i}/test.h5" for i in range(0, n_test_splits)]
        combine_HDFs(path_list, training_set=False, split="test")

        # 3) combine the validation files
        path_list        = [f"data/{dataset}_{i}/valid.h5" for i in range(0, n_valid_splits)]
        combine_HDFs(path_list, training_set=False, split="valid")

        # ---------- DELETE TEMPORARY FILES ----------
        for split_idx in range(max(n_training_splits, n_test_splits, n_valid_splits)):
            shutil.rmtree(f"{DATA_PATH}{dataset}_{split_idx}/")  # remove the dir and all files in it
    else:
        raise ValueError("Not a valid job type.")
