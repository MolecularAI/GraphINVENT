"""
Combines preprocessed HDF files. Useful when preprocessing large datasets, as
one can split the `train.smi` into multiple files (and directories), preprocess
them separately, and then combine using this script.

To use script, modify the variables below to automatically create a list of
paths **assuming** HDFs were created with the following directory structure:
 data/
  |-- {dataset}_1/
  |-- {dataset}_2/
  |-- {dataset}_3/
  |...
  |-- {dataset}_{n_dirs}/

The variables are also used in setting the dimensions of the HDF datasets later on.

If directories were not named as above, then simply replace `path_list` below
with a list of the paths to all the HDFs to combine.

Then, run:
python combine_HDFs.py
"""
import csv
import numpy as np
import h5py
import torch


# set variables
dataset          = "moses_canonical"
n_dirs           = 8    # how many times was `train.smi` split?
n_atom_types     = 7    # number of atom types used in preprocessing the data
n_formal_charges = 1    # number of formal charges used in preprocessing the data
n_bond_types     = 3    # number of bond types used in preprocessing the data
max_n_nodes      = 27   # maximum number of nodes in the data

# automatically generate a list of the HDF paths
path_list = [f"data/{dataset}_{i}/train.h5" for i in range(1, n_dirs + 1)]



def load_ts_properties_from_csv(csv_path : str) -> dict:
    """
    Loads CSV file containing training set properties and returns contents as a dictionary.
    """
    print("* Loading training set properties.", flush=True)

    # read dictionaries from csv
    with open(csv_path, "r") as csv_file:
        reader   = csv.reader(csv_file, delimiter=";")
        csv_dict = dict(reader)

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

def write_ts_properties_to_csv(ts_properties_dict : dict) -> None:
    """
    Writes the training set properties in `ts_properties_dict` to a CSV file.
    """
    dict_path = f"data/{dataset}/train.csv"

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

def main(paths : list) -> None:
    """
    Combine many small HDF files (their paths defined in `paths`) into one large HDF file.
    """
    total_n_subgraphs = get_total_n_subgraphs(paths)
    dims              = get_dims()

    print(f"* Creating HDF file to contain {total_n_subgraphs} subgraphs")
    new_hdf_file = h5py.File(f"data/{dataset}/train.h5", "a")
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

    print("* Combining data from respective `train.csv` files into one.")
    csv_list = [f"data/{dataset}_{i}/train.csv" for i in range(1, n_dirs + 1)]

    ts_properties_old = None
    csv_files_processed = 0
    for path in csv_list:
        ts_properties     = load_ts_properties_from_csv(csv_path=path)
        ts_properties_new = {}
        if ts_properties_old:
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

    write_ts_properties_to_csv(ts_properties_dict=ts_properties_new)


if __name__ == "__main__":
    # combine the HDFs defined in `path_list`
    main(path_list)
    print("Done.", flush=True)
