# load general packages and functions
import numpy as np
import rdkit
import h5py
import os
from tqdm import tqdm

# load program-specific functions
import analyze as anal
import apd
from parameters.constants import constants as C
import parameters.load as load
from MolecularGraph import PreprocessingGraph
import random
import util

# functions for preprocessing training data



def group_subgraphs(init_idx, molecule_set, dataset_dict, is_training_set, ts_properties_old=None):
    """ Collects graphs along all graphs in the decoding route for molecules
    in the training dataset by checking if they are equivalent (*not*
    isomorphic). Then, sums APDs for equivalent graphs and saves to HDF.

    Args:
      init_idx (int) : Index at which to start analyzing molecules in
        `molecule_set`; this is needed as analysis is done in blocks/slices, so
        `init_idx` is the start index for the next block/slice.
      molecule_set (list) : Contains `rdkit.Chem.Mol` objects.
      dataset_dict (dict) : Contains `h5py.Dataset`s.
      is_training_set (bool) : Indicates if data belongs to training set (as
        opposed to validation or testing set), in which case the
        `PreprocessingGraph` objects will be saved.
      ts_properties_old (dict or None) : If provided, it contains the properties
        of the previously processed group of structures. This is only done for
        the training set.

    Return:
      molecules_processed (int) : Total number of molecules processed, used for
        slicing `molecule_set` in the next block (i.e. even though each group
        contains `C.group_size` subgraphs, it is a variable number of molecules
        that may have generated `C.group_size` subgraphs, so this needs to be
        kept track of).
      dataset_dict (dict) : Contains `h5py.Dataset`s.
      group_size (int) : Either `C.group_size` or size of last processed group.
      ts_properties (dict or None) : If provided, it contains the properties
        of the current group of structures. This is only done for the training
        set.
    """
    data_subgraphs = []        # initialize
    data_APDs = []             # initialize
    molecular_graph_list = []  # initialize

    # convert all molecules in `molecule_set` to `MolecularGraphs` to loop over
    molecular_graph_generator = map(get_graph, molecule_set)

    molecules_processed = 0  # start counting # of molecules processed
    for graph in molecular_graph_generator:

        molecules_processed += 1

        # store `PreprocessingGraph` object
        molecular_graph_list.append(graph)

        # get the number of decoding graphs
        n_SGs = apd.get_decoding_route_length(molecular_graph=graph)

        for new_SG_idx in range(n_SGs):  # **note: "idx" == "idx"

            # `get_decoding_route_state() returns a list of [`SG`, `APD`],
            # where `SG := "subgraph"; APD := "action probability distribution"
            SG, APD = apd.get_decoding_route_state(molecular_graph=graph,
                                                   subgraph_idx=new_SG_idx)

            # "collect" all APDs corresponding to pre-existing subgraphs,
            # otherwise append both new subgraph and new APD
            count = 0
            for idx, existing_subgraph in enumerate(data_subgraphs):

                count += 1
                # check if subgraph `SG` is "already" in `data_subgraphs` as
                # `existing_subgraph`, and if so, add the "new" APD to the "old"
                try:  # first compare the node feature matrices
                    nodes_equal = (SG[0] == existing_subgraph[0]).all()
                except AttributeError:
                    nodes_equal = False
                try:  # then compare the edge feature tensors
                    edges_equal = (SG[1] == existing_subgraph[1]).all()
                except AttributeError:
                    edges_equal = False

                # if both matrices have a match, then subgraphs are the same
                if nodes_equal and edges_equal:
                    existing_APD = data_APDs[idx]

                    # add APDs
                    existing_APD += APD
                    break

            # if subgraph is not already in `data_subgraphs`, append it
            if count == len(data_subgraphs) or count == 0:
                data_subgraphs.append(SG)
                data_APDs.append(APD)

            # if `C.group_size` unique subgraphs have been processed, save
            # group to the HDF dataset
            len_data_subgraphs = len(data_subgraphs)
            if len_data_subgraphs == C.group_size:
                dataset_dict = save_group(dataset_dict=dataset_dict,
                                          group_size=C.group_size,
                                          data_subgraphs=data_subgraphs,
                                          data_APDs=data_APDs,
                                          init_idx=init_idx)

                # get molecular properties for group iff it's the training set
                ts_properties = get_ts_properties(
                    is_training_set=is_training_set,
                    molecular_graphs=molecular_graph_list,
                    group_size=C.group_size,
                    ts_properties_old=ts_properties_old)

                # return the datasets, now updated with an additional group
                return molecules_processed, dataset_dict, C.group_size, ts_properties

    # save group with < `C.group_size` subgraphs (e.g. the last block)
    dataset_dict = save_group(dataset_dict=dataset_dict,
                              group_size=len_data_subgraphs,
                              data_subgraphs=data_subgraphs,
                              data_APDs=data_APDs,
                              init_idx=init_idx)

    # get molecular properties for this group iff it's the training set
    ts_properties = get_ts_properties(is_training_set=is_training_set,
                                      molecular_graphs=molecular_graph_list,
                                      group_size=len_data_subgraphs,
                                      ts_properties_old=ts_properties_old)

    # return the datasets, now updated with an additional group
    return molecules_processed, dataset_dict, len_data_subgraphs, ts_properties


def create_datasets(hdf_file, max_length, dataset_name_list, dims):
    """ Creates a dictionary of HDF5 datasets.

    Args:
      hdf_file (h5py._hl.files.File) : HDF5 file to contain datasets.
      max_length (int) : Max dataset length (to be resized later; can only
        be resized if `chunks`==`True`).
      dataset_name_list (list) : Contains names of datasets.
      dims (dict) : Contains the dimensions to use for each dataset.
    """
    ds = {}  # initialize

    # use the name of the dataset as keys in the dictionary of datasets
    for ds_name in dataset_name_list:
        ds[ds_name] = hdf_file.create_dataset(ds_name,
                                              (max_length, *dims[ds_name]),
                                              chunks=True,
                                              dtype=np.dtype("int8"))

    return ds


def create_HDF_file(path, is_training_set=False):
    """ Preprocesses training data specified in `path` and writes it to HDF.

    Args:
      path (str) : Full path/filename to SMILES file containing molecules.
      is_training_set (bool) : Indicates if this is the training set.
    """
    # load the molecules
    molecule_set = load.molecules(path)

    # calculate the total number of molecules and the total number of subgraphs
    n_molecules = len(molecule_set)
    total_n_subgraphs = get_n_subgraphs(molecule_set=molecule_set)
    print(f"-- {n_molecules} molecules in set.", flush=True)
    print(f"-- {total_n_subgraphs} total subgraphs in set.", flush=True)

    # create special datatype for each set of arrays
    dataset_names = ["nodes", "edges", "APDs"]
    dims = get_dataset_dims()

    # prepare HDF5 file to save 6 different datasets to it
    with h5py.File(f"{path[:-3]}h5.chunked", "a") as hdf_file:

        # if a restart file exists and job is set to restart, then restart the
        # preprocessing where it left off, otherwise process as normal
        restart_index_file = C.dataset_dir + "index.restart"
        if C.restart and os.path.exists(restart_index_file):
            last_molecule_idx = util.read_last_molecule_idx(restart_file_path=C.dataset_dir)
            skip_collection = bool(last_molecule_idx == n_molecules and is_training_set)

            # load dictionary of previously created datasets (`ds` below)
            ds = load_datasets(hdf_file=hdf_file,
                               dataset_name_list=dataset_names)

        else:
            last_molecule_idx = 0
            skip_collection = False

            # create a dictionary of HDF datasets (`ds` below)
            ds = create_datasets(hdf_file=hdf_file,
                                 max_length=total_n_subgraphs,
                                 dataset_name_list=dataset_names,
                                 dims=dims)

        dataset_size = 0  # keep track of size to resize dataset later
        ts_properties = None

        # loop over subgraphs in blocks of size `C.group_size`
        for init_idx in range(0, total_n_subgraphs, C.group_size):
            # if `skip_collection` == True, skip directly to resizing/shuffling
            # of HDF datasets (e.g. skip the bit below)
            if not skip_collection:
                # get a slice of molecules based on the molecules that have
                # already been processed, indicated by `last_molecule_idx`
                molecule_subset = get_molecule_subset(molecule_set=molecule_set,
                                                      init_idx=last_molecule_idx,
                                                      n_molecules=n_molecules,
                                                      subset_size=C.group_size)

                # collect equivalent subgraphs
                (final_molecule_idx, ds, group_size,
                 ts_properties) = group_subgraphs(init_idx=init_idx,
                                                  molecule_set=molecule_subset,
                                                  dataset_dict=ds,
                                                  is_training_set=is_training_set,
                                                  ts_properties_old=ts_properties)

                # keep track of the last molecule to be processed
                last_molecule_idx += final_molecule_idx
                util.write_last_molecule_idx(last_molecule_idx=last_molecule_idx,
                                             restart_file_path=C.dataset_dir)
                dataset_size += group_size

            # grouping of graphs' APDs means that the number of groups will be
            # less than (`total_n_subgraphs` % `C.group_size`), so line below
            # breaks the loop once the last molecule in group is the last in
            # the dataset
            if last_molecule_idx == n_molecules:

                # resize HDF datasets by removing extra padding from initialization
                resize_datasets(dataset_dict=ds,
                                dataset_names=dataset_names,
                                dataset_size=dataset_size,
                                dataset_dims=dims)

                print("Datasets resized.", flush=True)
                if is_training_set:

                    print("Writing training set properties.", flush=True)
                    util.write_ts_properties(ts_properties_dict=ts_properties)

                    print("Shuffling training dataset.", flush=True)
                    for _ in range(int(np.sqrt(dataset_size))):
                        random1 = random.randrange(0, dataset_size, 5)
                        random2 = random.randrange(0, dataset_size, 5)
                        ds = shuffle_datasets(dataset_dict=ds,
                                              dataset_names=dataset_names,
                                              idx1=random1,
                                              idx2=random2)

                break

    print(f"* Resaving datasets in unchunked format.")
    with h5py.File(f"{path[:-3]}h5.chunked", "r", swmr=True) as chunked_file:
        keys = list(chunked_file.keys())
        data = [chunked_file.get(key)[:] for key in keys]
        data_zipped = tuple(zip(data, keys))

        with h5py.File(f"{path[:-3]}h5", "w") as unchunked_file:
            for d, k in tqdm(data_zipped):
                unchunked_file.create_dataset(k, chunks=None, data=d, dtype=np.dtype("int8"))

    # remove the restart file and chunked file if all steps are done
    os.remove(restart_index_file)
    os.remove(f"{path[:-3]}h5.chunked")

    return None


def resize_datasets(dataset_dict, dataset_names, dataset_size, dataset_dims):
    """ Resizes the input HDF datasets in `dataset_dict`. Originally a much
    longer dataset is created when creating the HDF dataset because it is
    impossible to precisely predict how many graphs will be equivalent
    beforehand, so the datasets are all made as long as the upper bound.
    """
    for dataset_name in dataset_names:
        try:
            dataset_dict[dataset_name].resize(
                (dataset_size, *dataset_dims[dataset_name]))
        except KeyError:  # `f_term` has no extra dims
            dataset_dict[dataset_name].resize((dataset_size,))

    return dataset_dict


def get_dataset_dims():
    """ Calculates the dimensions of the node features, edge features, and APD
    tensors. Returns a `dict` containing the "dimensions" (as a `list`) where
    they key is the dataset name.
    """
    dims = {}
    dims["nodes"] = C.dim_nodes
    dims["edges"] = C.dim_edges
    dims["APDs"] = [np.prod(C.dim_f_add) + np.prod(C.dim_f_conn) + 1]
    return dims


def get_graph(mol):
    """ Converts `rdkit.Chem.Mol` object to `PreprocessingGraph`.
    """
    if mol is not None:
        if not C.use_aromatic_bonds:
            rdkit.Chem.Kekulize(mol, clearAromaticFlags=True)
        molecular_graph = PreprocessingGraph(molecule=mol, constants=C)

        return molecular_graph


def get_molecule_subset(molecule_set, init_idx, n_molecules, subset_size):
    """ "Slices" the input set of molecules (`molecules_set`) into a subset of
    size `subset_size` (`int`), starting from `init_idx` (`int`). `n_molecules`
    (`int`) is the number of molecules in the full `molecule_set` (`list`).
    """
    molecule_subset = []
    max_idx = min(init_idx + subset_size, n_molecules)

    count = -1
    for mol in molecule_set:
        if mol is not None:
            count += 1
            if count < init_idx:
                continue
            elif count >= max_idx:
                return molecule_subset
            else:
                molecule_subset.append(mol)

    return molecule_subset


def get_n_subgraphs(molecule_set):
    """ Calculates the total number of subgraphs in the decoding route of all
    molecules in `molecule_set`. Loads training, testing, or validation set.
    First, the `PreprocessingGraph` for each molecule is obtained, and then the
    length of the decoding route is trivially calculated for each.

    Args:
      molecule_set (list) : Contains `rdkit.Chem.Mol` objects.

    Returns:
      n_subgraphs (int) : Sum of number of subgraphs in decoding routes of all
        molecules in `molecule_set`.
    """
    n_subgraphs = 0  # start the count

    # convert molecules in `molecule_set` to `PreprocessingGraph`s to loop over
    molecular_graph_generator = map(get_graph, molecule_set)

    for molecular_graph in molecular_graph_generator:

        # get the number of decoding graphs (i.e. the decoding route length)
        n_SGs = apd.get_decoding_route_length(molecular_graph=molecular_graph)

        # add the number of subgraphs to the running count
        n_subgraphs += n_SGs

    return n_subgraphs


def get_ts_properties(is_training_set, molecular_graphs, group_size, ts_properties_old):
    """ Gets molecular properties for group of molecular graphs iff it's the
    training set.

    Args:
      is_training_set (bool) : Indicates if the molecular graphs belong to the
        training set.
      molecular_graphs (list) : Contains `PreprocessingGraph`s.
      group_size (int) : Size of "group" (i.e. slice of graphs).
      ts_properties_old (None or dict) : If not `None`, contains the "old"
        training set properties (i.e. those of the previous group analyzed).

    Returns:
      ts_properties (None or dict) : If not `None`, contains the merged
        training set properties.
    """
    if is_training_set:

        ts_properties_new = anal.evaluate_training_set(
            preprocessing_graphs=molecular_graphs
        )

        # merge properties of current group with the previous group analyzed
        if ts_properties_old:
            ts_properties = anal.combine_ts_properties(
                prev_properties=ts_properties_old,
                next_properties=ts_properties_new,
                weight_next=group_size)
        else:
            ts_properties = ts_properties_new
    else:
        ts_properties = None

    return ts_properties


def load_datasets(hdf_file, dataset_name_list):
    """ Creates a dictionary of HDF datasets which have been previously created
    (for restart jobs only).

    Args:
      hdf_file (h5py._hl.files.File) : HDF5 file containing all the datasets.
      dataset_name_list (list) : Contains names (strings) of datasets.

    Returns:
      ds (dict) : Dictionary of datasets.
    """
    ds = {}  # initialize

    # use the name of the dataset as keys in the dictionary of datasets
    for ds_name in dataset_name_list:
        ds[ds_name] = hdf_file.get(ds_name)

    return ds


def save_group(dataset_dict, data_subgraphs, data_APDs, group_size, init_idx):
    """ Saves a group of padded subgraphs and their corresponding APDs to the
    existing HDF5 file as `numpy.ndarray`s.

    Args:
      dataset_dict (dict) : Contains HDF5 datasets.
      data_subgraphs (list) : Contains molecular subgraphs.
      data_APDs (list) : Contains APDs.
      group_size (int) : Size of HDF5 "slice".
      init_idx (int) : Index to begin slicing.
    """
    # convert to `np.ndarray`s
    nodes = np.array([graph_tuple[0] for graph_tuple in data_subgraphs])
    edges = np.array([graph_tuple[1] for graph_tuple in data_subgraphs])
    APDs = np.array(data_APDs)

    end_idx = init_idx + group_size  # idx to end slicing

    # once data is padded, save it to dataset slice
    dataset_dict["nodes"][init_idx:end_idx] = nodes
    dataset_dict["edges"][init_idx:end_idx] = edges
    dataset_dict["APDs"][init_idx:end_idx] = APDs


    return dataset_dict


def shuffle_datasets(dataset_dict, dataset_names, idx1, idx2):
    """ Shuffles two elements, indicated by `idx1` and `idx2`, between
    two datasets in `dataset_dict`.
    """
    for name in dataset_names:
        dataset_dict[name][idx1], dataset_dict[name][idx2] = \
            dataset_dict[name][idx2], dataset_dict[name][idx1]

    return dataset_dict
