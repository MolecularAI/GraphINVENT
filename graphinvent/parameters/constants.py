"""
Loads input parameters from `defaults.py`, and defines other global constants
that depend on the input features, creating a `namedtuple` from them;
additionally, if there exists an `input.csv` in the job directory, loads those
arguments and overrides default values in `defaults.py`.
"""
# load general packages and functions
from collections import namedtuple
import pickle
import csv
import os
import sys
from typing import Tuple
import numpy as np
from rdkit.Chem.rdchem import BondType

# load GraphINVENT-specific functions
sys.path.insert(1, "./parameters/")  # search "parameters/" directory
import parameters.args as args
import parameters.defaults as defaults


def get_feature_dimensions(parameters : dict) -> Tuple[int, int, int, int]:
    """
    Returns dimensions of all node features.
    """
    n_atom_types    = len(parameters["atom_types"])
    n_formal_charge = len(parameters["formal_charge"])
    n_numh          = (
        int(not parameters["use_explicit_H"] and not parameters["ignore_H"])
        * len(parameters["imp_H"])
    )
    n_chirality     = int(parameters["use_chirality"]) * len(parameters["chirality"])

    return n_atom_types, n_formal_charge, n_numh, n_chirality


def get_tensor_dimensions(n_atom_types : int, n_formal_charge : int, n_num_h : int,
                          n_chirality : int, n_node_features : int, n_edge_features : int,
                          parameters : dict) -> Tuple[list, list, list, list, int]:
    """
    Returns dimensions for all tensors that describe molecular graphs. Tensor dimensions
    are `list`s, except for `dim_f_term` which is  simply an `int`. Each element
    of the lists indicate the corresponding dimension of a particular subgraph matrix
    (i.e. `nodes`, `f_add`, etc).
    """
    max_nodes = parameters["max_n_nodes"]

    # define the matrix dimensions as `list`s
    # first for the graph reps...
    dim_nodes = [max_nodes, n_node_features]

    dim_edges = [max_nodes, max_nodes, n_edge_features]

    # ... then for the APDs
    if parameters["use_chirality"]:
        if parameters["use_explicit_H"] or parameters["ignore_H"]:
            dim_f_add = [
                parameters["max_n_nodes"],
                n_atom_types,
                n_formal_charge,
                n_chirality,
                n_edge_features,
            ]
        else:
            dim_f_add = [
                parameters["max_n_nodes"],
                n_atom_types,
                n_formal_charge,
                n_num_h,
                n_chirality,
                n_edge_features,
            ]
    else:
        if parameters["use_explicit_H"] or parameters["ignore_H"]:
            dim_f_add = [
                parameters["max_n_nodes"],
                n_atom_types,
                n_formal_charge,
                n_edge_features,
            ]
        else:
            dim_f_add = [
                parameters["max_n_nodes"],
                n_atom_types,
                n_formal_charge,
                n_num_h,
                n_edge_features,
            ]

    dim_f_conn = [parameters["max_n_nodes"], n_edge_features]

    dim_f_term = 1

    return dim_nodes, dim_edges, dim_f_add, dim_f_conn, dim_f_term


def load_params(input_csv_path : str) -> dict:
    """
    Loads job parameters/hyperparameters from CSV (in `input_csv_path`).
    """
    params_to_override = {}
    with open(input_csv_path, "r") as csv_file:

        params_reader = csv.reader(csv_file, delimiter=";")

        for key, value in params_reader:
            try:
                params_to_override[key] = eval(value)
            except NameError:  # `value` is a `str`
                params_to_override[key] = value
            except SyntaxError:  # to avoid "unexpected `EOF`"
                params_to_override[key] = value

    return params_to_override


def override_params(all_params : dict) -> dict:
    """
    If there exists an `input.csv` in the job directory, loads those arguments
    and overrides their default values from `features.py`.
    """
    input_csv_path = all_params["job_dir"] + "input.csv"

    # check if there exists and `input.csv` in working directory
    if os.path.exists(input_csv_path):
        # override default values for parameters in `input.csv`
        params_to_override_dict = load_params(input_csv_path)
        for key, value in params_to_override_dict.items():
            all_params[key] = value

    return all_params


def collect_global_constants(parameters : dict, job_dir : str) -> namedtuple:
    """
    Collects constants defined in `features.py` with those defined by the
    ArgParser (`args.py`), and returns the bundle as a `namedtuple`.

    Args:
    ----
        parameters (dict) : Dictionary of parameters defined in `features.py`.
        job_dir (str)     : Current job directory, defined on the command line.

    Returns:
    -------
        constants (namedtuple) : Collected constants.
    """
    # first override any arguments from `input.csv`:
    parameters["job_dir"] = job_dir
    parameters            = override_params(all_params=parameters)

    # then calculate any global constants below:
    if parameters["use_explicit_H"] and parameters["ignore_H"]:
        raise ValueError("Cannot use explicit Hs and ignore Hs at "
                         "the same time. Please fix flags.")

    # define edge feature (rdkit `GetBondType()` result -> `int`) constants
    bondtype_to_int = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2}

    if parameters["use_aromatic_bonds"]:
        bondtype_to_int[BondType.AROMATIC] = 3

    int_to_bondtype = dict(map(reversed, bondtype_to_int.items()))

    n_edge_features = len(bondtype_to_int)

    # define node feature constants
    n_atom_types, n_formal_charge, n_imp_H, n_chirality = get_feature_dimensions(parameters)

    n_node_features = n_atom_types + n_formal_charge + n_imp_H + n_chirality

    # define matrix dimensions
    (dim_nodes, dim_edges, dim_f_add,
     dim_f_conn, dim_f_term) = get_tensor_dimensions(n_atom_types,
                                                     n_formal_charge,
                                                     n_imp_H,
                                                     n_chirality,
                                                     n_node_features,
                                                     n_edge_features,
                                                     parameters)

    len_f_add           = np.prod(dim_f_add[:])
    len_f_add_per_node  = np.prod(dim_f_add[1:])
    len_f_conn          = np.prod(dim_f_conn[:])
    len_f_conn_per_node = np.prod(dim_f_conn[1:])

    # create a dictionary of global constants, and add `job_dir` to it; this
    # will ultimately be converted to a `namedtuple`
    constants_dict = {
        "big_negative"       : -1e6,
        "big_positive"       : 1e6,
        "bondtype_to_int"    : bondtype_to_int,
        "int_to_bondtype"    : int_to_bondtype,
        "n_edge_features"    : n_edge_features,
        "n_atom_types"       : n_atom_types,
        "n_formal_charge"    : n_formal_charge,
        "n_imp_H"            : n_imp_H,
        "n_chirality"        : n_chirality,
        "n_node_features"    : n_node_features,
        "dim_nodes"          : dim_nodes,
        "dim_edges"          : dim_edges,
        "dim_f_add"          : dim_f_add,
        "dim_f_conn"         : dim_f_conn,
        "dim_f_term"         : dim_f_term,
        "dim_apd"            : [np.prod(dim_f_add) + np.prod(dim_f_conn) + 1],
        "len_f_add"          : len_f_add,
        "len_f_add_per_node" : len_f_add_per_node,
        "len_f_conn"         : len_f_conn,
        "len_f_conn_per_node": len_f_conn_per_node,
    }

    # join with `features.args_dict`
    constants_dict.update(parameters)

    # define path to dataset splits
    constants_dict["test_set"]       = parameters["dataset_dir"] + "test.smi"
    constants_dict["training_set"]   = parameters["dataset_dir"] + "train.smi"
    constants_dict["validation_set"] = parameters["dataset_dir"] + "valid.smi"

    # check (if a job is not a preprocessing job) that parameters  match those for
    # the original preprocessing job
    if constants_dict["job_type"] != "preprocess":
        print(
            "* Running job using HDF datasets located at "
            + parameters["dataset_dir"],
            flush=True,
        )
        print(
            "* Checking that the relevant parameters match "
            "those used in preprocessing the dataset.",
            flush=True,
        )

        # load preprocessing parameters for comparison (if they exist already)
        csv_file        = parameters["dataset_dir"] + "preprocessing_params.csv"
        params_to_check = load_params(input_csv_path=csv_file)

        for key, value in params_to_check.items():
            if key in constants_dict.keys() and value != constants_dict[key]:
                raise ValueError(
                    f"Check that training job parameters match those used in "
                    f"preprocessing. {key} does not match."
                )

        # if above error never raised, then all relevant parameters match! :)
        print("-- Job parameters match preprocessing parameters.", flush=True)

    # load QSAR models (sklearn activity model)
    if constants_dict["job_type"] == "fine-tune":
        print("-- Loading pre-trained scikit-learn activity model.", flush=True)
        for qsar_model_name, qsar_model_path in constants_dict["qsar_models"].items():
            with open(qsar_model_path, 'rb') as file:
                model_dict                                     = pickle.load(file)
                activity_model                                 = model_dict["classifier_sv"]
                constants_dict["qsar_models"][qsar_model_name] = activity_model

    # convert `CONSTANTS` dictionary into a namedtuple (immutable + cleaner)
    Constants = namedtuple("CONSTANTS", sorted(constants_dict))
    constants = Constants(**constants_dict)

    return constants

# collect the constants using the functions defined above
constants = collect_global_constants(parameters=defaults.parameters,
                                     job_dir=args.job_dir)
