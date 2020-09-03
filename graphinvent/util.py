# load general packages and functions
import csv
import numpy as np
import torch
import rdkit
from tensorboard_writer import writer as tb_writer

# load program-specific functions
from parameters.constants import constants as C

# contains miscellaneous useful functions



def get_feature_vector_indices():
    """ Gets the indices of the different segments of the feature vector. The
    indices are analogous to the lengths of the various segments.

    Returns:
      idc (list) : Contains the indices of the different one-hot encoded
        segments used in the feature vector representations of nodes in
        `MolecularGraph`s. These segments are, in order, atom type, formal
        charge, number of implicit Hs, and chirality.
    """
    idc = [C.n_atom_types, C.n_formal_charge]

    # indices corresponding to implicit H's and chirality are optional (below)
    if not C.use_explicit_H and not C.ignore_H:
        idc.append(C.n_imp_H)

    if C.use_chirality:
        idc.append(C.n_chirality)

    return np.cumsum(idc).tolist()


def get_last_epoch():
    """ Gets the epoch, learning rate, and loss of the previous training epoch.
    Reads it from file.
    """
    # define the path to the file containing desired information
    convergence_path = C.job_dir + "convergence.csv"

    try:
        epoch_key, lr, avg_loss = read_row(
            path=convergence_path, row=-1, col=(0, 1, 2)
        )
    except ValueError:
        epoch_key = "Epoch 1"

    generation_epoch = C.generation_epoch
    if C.job_type == "generate":
        epoch_key = f"Epoch GEN{generation_epoch}"
    elif C.job_type == "test":
        epoch_key = f"Epoch EVAL{generation_epoch}"

    return epoch_key


def normalize_evaluation_metrics(prop_dict, epoch_key):
    """ Normalizes histograms in `props_dict` (see below) and converts them
    to `list`s (from `torch.Tensor`s) and rounds the elements. This is done for
    clarity when saving the histograms to CSV.

    Returns:
      norm_n_nodes_hist (torch.Tensor) : Normalized histogram of the number of
        nodes per molecule.
      norm_atom_type_hist (torch.Tensor) : Normalized histogram of the atom
        types present in the molecules.
      norm_charge_hist (torch.Tensor) : Normalized histogram of the formal
        charges present in the molecules.
      norm_numh_hist (torch.Tensor) : Normalized histogram of the number of
        implicit hydrogens present in the molecules.
      norm_n_edges_hist (torch.Tensor) : Normalized histogram of the number of
        edges per node in the molecules.
      norm_edge_feature_hist (torch.Tensor) : Normalized histogram of the
        edge features (types of bonds) present in the molecules.
      norm_chirality_hist (torch.Tensor) : Normalized histogram of the
        chiral centers present in the molecules.
    """
    # compute histograms for non-optional features
    norm_n_nodes_hist = [
        round(i, 2) for i in
        norm(prop_dict[(epoch_key, "n_nodes_hist")]).tolist()
    ]

    norm_atom_type_hist = [
        round(i, 2) for i in
        norm(prop_dict[(epoch_key, "atom_type_hist")]).tolist()
    ]

    norm_charge_hist = [
        round(i, 2) for i in
        norm(prop_dict[(epoch_key, "formal_charge_hist")]).tolist()
    ]

    norm_n_edges_hist = [
        round(i, 2) for i in
        norm(prop_dict[(epoch_key, "n_edges_hist")]).tolist()
    ]

    norm_edge_feature_hist = [
        round(i, 2) for i in
        norm(prop_dict[(epoch_key, "edge_feature_hist")]).tolist()
    ]

    # compute histograms for optional features
    if not C.use_explicit_H and not C.ignore_H:
        norm_numh_hist = [
            round(i, 2) for i in
            norm(prop_dict[(epoch_key, "numh_hist")]).tolist()
        ]
    else:
        norm_numh_hist = [0] * len(C.imp_H)

    if C.use_chirality:
        norm_chirality_hist = [
            round(i, 2) for i in
            norm(prop_dict[(epoch_key, "chirality_hist")]).tolist()
        ]
    else:
        norm_chirality_hist = [1, 0, 0]

    return (
        norm_n_nodes_hist,
        norm_atom_type_hist,
        norm_charge_hist,
        norm_numh_hist,
        norm_n_edges_hist,
        norm_edge_feature_hist,
        norm_chirality_hist,
    )


def get_restart_epoch():
    """ Gets the restart epoch e.g. epoch for the last saved model state
    (`model_restart.pth`). Will simply return zero if called outside of a
    restart job.
    """
    if C.restart or C.job_type == "test":
        # define path to output file containing info on last saved state
        generation_path = C.job_dir + "generation.csv"

        epoch = "NA"
        row = -1

        while type(epoch) is not int:
            epoch_key = read_row(path=generation_path, row=row, col=0)
            try:
                epoch = int(epoch_key[6:])
            except ValueError:
                epoch = "NA"
            row -= 1
    else:
        epoch = 0

    return epoch


def load_ts_properties(csv_path):
    """ Loads training set properties from CSV, specified by the `csv_path`
    (`str`). Returns contents as `properties_dict` (`dict`).
    """
    print("* Loading training set properties.", flush=True)

    # read dictionary from CSV
    with open(csv_path, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=";")
        csv_dict = dict(reader)

    # create `properties_dict` from `csv_dict`, fix any bad filetypes
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


def norm(list_of_nums):
    """ Normalizes input `list_of_nums` (`list` of `float`s or `int`s)
    """
    try:
        norm_list_of_nums = list_of_nums / sum(list_of_nums)
    except:  # occurs if divide by zero
        norm_list_of_nums = list_of_nums

    return norm_list_of_nums


def one_of_k_encoding(x, allowable_set):
    """ Returns the one-of-k encoding of a value `x` having a range of possible
    values in `allowable_set`.

    Args:
      x (str, int) : Value to be one-hot encoded.
      allowable_set (list) : `list` of all possible values.

    Returns:
      one_hot_generator (generator) : One-hot encoding. A generator of `int`s.
    """
    if x not in set(allowable_set):  # use set for speedup over list
        raise Exception(
            f"Input {x} not in allowable set {allowable_set}. "
            f"Add {x} to allowable set in `features.py` and run again."
        )

    one_hot_generator = (int(x == s) for s in allowable_set)

    return one_hot_generator


def one_of_k_encoding_unk(x, allowable_set):
    """ Returns the one-of-k encoding of a value `x` having a range of possible
    values in `allowable_set`. If an input is not in the `allowable_set`, then
    it encodes is as "unknown" e.g. the last bin of `allowable_set`.

    Args:
      x (str, int) : Value to be one-hot encoded.
      allowable_set (list) : `list` of all possible values.

    Returns:
      one_hot_generator (generator) : One-hot encoding. A generator of `int`s.
    """
    if x not in set(allowable_set):
        x = allowable_set[-1]  # allowable_set[-1] should = "unknown"
    one_hot_generator = (int(x == s) for s in allowable_set)

    return one_hot_generator


def properties_to_csv(prop_dict, csv_filename, epoch_key, append=True):
    """ Writes a CSV summarizing how training is going by comparing the
    properties of the generated structures during evaluation to the
    training set.

    Args:
      prop_dict (dict) : Contains molecular properties.
      csv_filename (str) : Full path/filename to CSV file.
      epoch_key (str) : For example, "Training set" or "Epoch {n}".
      append (bool) : Indicates whether to append to the output file (if the
        file exists) or start a new one. Default `True`.
    """
    # get all the relevant properties from the dictionary
    frac_valid = prop_dict[(epoch_key, "fraction_valid")]
    avg_n_nodes = prop_dict[(epoch_key, "avg_n_nodes")]
    avg_n_edges = prop_dict[(epoch_key, "avg_n_edges")]
    frac_unique = prop_dict[(epoch_key, "fraction_unique")]

    # use the following properties if they exist e.g. for generation epochs, but
    # not for training set
    try:
        run_time = prop_dict[(epoch_key, "run_time")]
        frac_valid_pt = round(
            float(prop_dict[(epoch_key, "fraction_valid_properly_terminated")]), 5
        )
        frac_pt = round(
            float(prop_dict[(epoch_key, "fraction_properly_terminated")]), 5
        )
    except KeyError:
        run_time = "NA"
        frac_valid_pt = "NA"
        frac_pt = "NA"

    (
        norm_n_nodes_hist,
        norm_atom_type_hist,
        norm_formal_charge_hist,
        norm_numh_hist,
        norm_n_edges_hist,
        norm_edge_feature_hist,
        norm_chirality_hist,
    ) = normalize_evaluation_metrics(prop_dict, epoch_key)

    if not append:
        # file does not exist yet, create it
        with open(csv_filename, "w") as output_file:
            # write the file header
            output_file.write(
                "set, fraction_valid, fraction_valid_pt, fraction_pt, run_time, "
                "avg_n_nodes, avg_n_edges, fraction_unique, atom_type_hist, "
                "formal_charge_hist, numh_hist, chirality_hist, "
                "n_nodes_hist, n_edges_hist, edge_feature_hist\n"
            )

    # append the properties of interest to the CSV file
    with open(csv_filename, "a") as output_file:
        output_file.write(
            f"{epoch_key}, {frac_valid:.3f}, {frac_valid_pt}, {frac_pt}, {run_time}, "
            f"{avg_n_nodes:.3f}, {avg_n_edges:.3f}, {frac_unique:.3f}, "
            f"{norm_atom_type_hist}, {norm_formal_charge_hist}, "
            f"{norm_numh_hist}, {norm_chirality_hist}, {norm_n_nodes_hist}, "
            f"{norm_n_edges_hist}, {norm_edge_feature_hist}\n"
        )

    # write to tensorboard
    try:
        epoch = int(epoch_key.split()[1])
    except:
        pass
    else:
        # Scalars
        tb_writer.add_scalar("Evaluation/valid", frac_valid, epoch)
        tb_writer.add_scalar("Evaluation/valid_pt", frac_valid_pt, epoch)
        tb_writer.add_scalar("Evaluation/pt", frac_pt, epoch)
        tb_writer.add_scalar("Evaluation/n_nodes", avg_n_nodes, epoch)
        tb_writer.add_scalar("Evaluation/unique", frac_unique, epoch)

        # Histogram
        tb_writer.add_histogram("Distributions/atom_type",
                                np.array(norm_atom_type_hist), epoch)
        tb_writer.add_histogram("Distributions/form_charge",
                                np.array(norm_formal_charge_hist), epoch)
        tb_writer.add_histogram("Distributions/hydrogen",
                                np.array(norm_numh_hist), epoch)
        tb_writer.add_histogram("Distributions/n_nodes",
                                np.array(norm_n_nodes_hist), epoch)
        tb_writer.add_histogram("Distributions/n_edges",
                                np.array(norm_n_edges_hist), epoch)
        tb_writer.add_histogram("Distributions/edge_features",
                                np.array(norm_edge_feature_hist), epoch)


def read_column(path, column):
    """ Reads a `column` from CSV file specified in `path` and returns it as a
    `numpy.ndarray`. Removes "NA" missing values from `data` before returning.
    """
    with open(path, "r") as csv_file:
        data = np.genfromtxt(csv_file,
                             dtype=None,
                             delimiter=",",
                             skip_header=1,
                             usecols=column,
                             missing_values="NA")

    data = np.array(data)
    data = data[~np.isnan(data)]  # exclude `nan`

    return data


def read_last_molecule_idx(restart_file_path):
    """ Reads the index of the last preprocessed molecule from a file called
    "index.restart" located in the same directory as the data.
    """
    with open(restart_file_path + "index.restart", "r") as txt_file:

        last_molecule_idx = txt_file.read()

    return int(last_molecule_idx)


def read_row(path, row, col):
    """ Reads a row from CSV file specified in `path` and returns it as a
    `numpy.ndarray`. Removes "NA" missing values from `data` before returning.
    """
    with open(path, "r") as csv_file:
        data = np.genfromtxt(csv_file, dtype=str, delimiter=",", skip_header=1, usecols=col)

    data = np.array(data)

    return data[:][row]


def suppress_warnings():
    """ Suppresses unimportant warnings for a cleaner readout.
    """
    from rdkit import RDLogger
    from warnings import filterwarnings

    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    filterwarnings(action="ignore", category=UserWarning)
    filterwarnings(action="ignore", category=FutureWarning)
    # could instead suppress ALL warnings with:
    # `filterwarnings(action="ignore")`
    # but choosing not to do this


def turn_off_empty_axes(n_plots_y, n_plots_x, ax):
    """ Turns off empty axes in a `n_plots_y` by `n_plots_x` grid of plots.

    Args:
      n_plots_y (int) : See above.
      n_plots_x (int) : See above.
      ax (matplotlib.axes) : Matplotlib object containing grid of plots.
    """
    for vi in range(n_plots_y):
        for vj in range(n_plots_x):
            # if nothing plotted on ax, it will contain `inf`
            # in axes lims, so clean up (turn off)
            if "inf" in str(ax[vi, vj].dataLim):
                ax[vi, vj].axis("off")

    return ax


def update_lr(optimizer, scale_factor, minimum_lr=1e-8, maximum_lr=1e-2):
    """ Updates the learning rate by a constant scale factor.

    Args:
      optimizer (torch.optim) : Optimizer class (e.g. `torch.optim.Adam`).
      scale_factor (float) : Factor by which to scale the learning rate.
      minimum_lr (float) : Minimum allowed lr, after which will stop updating.
      maximum_lr (float) : Maximum allowed lr, after which will stop updating.
    """
    lr = optimizer.param_groups[0]["lr"]
    new_lr = lr * scale_factor

    if new_lr < minimum_lr:
        print("Learning rate has reached minimum lr.", flush=True)
        pass  # stop decaying the lr
    elif lr != maximum_lr and new_lr > maximum_lr:
        for param_group in optimizer.param_groups:
            param_group["lr"] = maximum_lr
        print(f"Updated learning rate to {maximum_lr} (maximum).", flush=True)
    elif lr == maximum_lr and new_lr > maximum_lr:
        pass  # do nothing
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        print(f"Updated learning rate to {new_lr}", flush=True)


def write_last_molecule_idx(last_molecule_idx, restart_file_path):
    """ Writes the index of the last preprocessed molecule (`last_molecule_idx`)
    to a file called "index.restart" to be located in the same directory as
    the data.
    """
    with open(restart_file_path + "index.restart", "w") as txt_file:
        txt_file.write(str(last_molecule_idx))


def write_job_parameters(params):
    """ Writes job parameters/hyperparameters in `params` (`namedtuple`) to
    CSV..
    """
    dict_path = params.job_dir + "params.csv"

    with open(dict_path, "w") as csv_file:

        writer = csv.writer(csv_file, delimiter=";")
        i = 0
        for key, value in enumerate(params._fields):
            writer.writerow([value, params[key]])
            i += 1


def write_preprocessing_parameters(params):
    """ Writes job parameters/hyperparameters in `params` (`namedtuple`) to
    CSV, so that parameters used during preprocessing can be referenced later.
    """
    dict_path = params.dataset_dir + "preprocessing_params.csv"
    keys_to_write = ["atom_types",
                     "formal_charge",
                     "imp_H",
                     "chirality",
                     "group_size",
                     "max_n_nodes",
                     "use_aromatic_bonds",
                     "use_chirality",
                     "use_explicit_H",
                     "ignore_H"]

    with open(dict_path, "w") as csv_file:

        writer = csv.writer(csv_file, delimiter=";")
        for key, value in enumerate(params._fields):
            if value in keys_to_write:
                writer.writerow([value, params[key]])


def write_graphs_to_smi(smi_filename, molecular_graphs_list):
    """ Writes the `TrainingGraph`s in `molecular_graphs_list` to a SMILES
    file, where the full path/filename is specified by `smi_filename`.
    """
    validity_tensor = torch.zeros(len(molecular_graphs_list), device="cuda")

    with open(smi_filename, "w") as smi_file:

        smi_writer = rdkit.Chem.rdmolfiles.SmilesWriter(smi_file)

        for idx, molecular_graph in enumerate(molecular_graphs_list):

            mol = molecular_graph.get_molecule()
            try:
                mol.UpdatePropertyCache(strict=False)
                rdkit.Chem.SanitizeMol(mol)
                smi_writer.write(mol)
                validity_tensor[idx] = 1
            except (ValueError, RuntimeError, AttributeError):
                # molecule cannot be written to file (likely contains unphysical
                # aromatic bond(s) or an unphysical valence), so put placeholder
                smi_writer.write(
                    rdkit.Chem.MolFromSmiles("[Xe]")
                )  # and `validity_tensor` remains 0

        smi_writer.close()

    fraction_valid = torch.sum(validity_tensor, dim=0) / len(validity_tensor)

    return fraction_valid, validity_tensor

def write_model_status(epoch=None, lr=None, loss=None, score=None, append=True):
    """ Writes the current epoch, loss, learning rate, and model score to CSV.
    """
    convergence_path = C.job_dir + "convergence.csv"

    if not append:  # create the file
        with open(convergence_path, "w") as output_file:
            # write the header
            output_file.write("epoch, lr, avg_loss, model_score\n")
    else:  # append to existing file
        if C.job_type == "train":  # only write a `convergence.csv` when training
            if score is None:
                with open(convergence_path, "a") as output_file:
                    output_file.write(f"Epoch {epoch}, {lr:.8f}, {loss:.8f}, ")
                # write to tensorboard
                tb_writer.add_scalar("Train/loss", loss, epoch)
                tb_writer.add_scalar("Train/lr", lr, epoch)

            elif score == "NA":
                with open(convergence_path, "a") as output_file:
                    output_file.write(f"{score}\n")

            elif score is not None:
                with open(convergence_path, "a") as output_file:
                    output_file.write(f"{score:.6f}\n")

            else:
                raise NotImplementedError



def write_molecules(molecules, final_nlls, epoch):
    """ Writes generated molecular graphs and their NLLs. In writing the
    structures to a SMILES file, determines if structures are valid and returns
    this information (to avoid recomputing later).

    Args:
      molecules (list) : Contains generated `MolecularGraph`s.
      final_nlls (torch.Tensor) : Contains final NLLs for the graphs.
      epoch (str) : Number corresponding to the current training epoch.

    Returns:
      fraction_valid (float) : Fraction of valid structures in the input set.
      validity_tensor (torch.Tensor) : Contains either a 0 or 1 at index
        corresponding to a graph in `molecules` to indicate if graph is valid.
    """
    # save molecules as SMILES
    smi_filename = C.job_dir + f"generation/epoch{epoch}.smi"
    fraction_valid, validity_tensor = write_graphs_to_smi(smi_filename=smi_filename,
                                                           molecular_graphs_list=molecules)

    # save the NLLs and validity status
    write_nlls(nll_filename=f"{smi_filename[:-3]}nll",
               nlls=final_nlls)
    write_validity(validity_file_path=f"{smi_filename[:-3]}valid",
                   validity_tensor=validity_tensor)

    return fraction_valid, validity_tensor


def write_nlls(nll_filename, nlls):
    """ Writes the final NLL of each molecule to a file in the same order as
    the molecules are written in the corresponding SMILES file.
    """
    with open(nll_filename, "w") as nll_file:
        for nll in nlls:
            nll_file.write(f"{nll}\n")


def write_ts_properties(ts_properties_dict):
    """ Writes the training set properties in `ts_properties_dict` to CSV.
    """
    training_set = C.training_set  # path to "train.smi"
    dict_path = f"{training_set[:-4]}.csv"

    with open(dict_path, "w") as csv_file:

        csv_writer = csv.writer(csv_file, delimiter=";")
        for key, value in ts_properties_dict.items():
            if "validity_tensor" in key:
                # skip writing the validity tensor here because it is really
                # long, instead it gets its own file elsewhere
                continue
            elif type(value) == np.ndarray:
                csv_writer.writerow([key, list(value)])
            elif type(value) == torch.Tensor:
                try:
                    csv_writer.writerow([key, float(value)])
                except ValueError:
                    csv_writer.writerow([key, [float(i) for i in value]])
            else:
                csv_writer.writerow([key, value])


def write_validation_scores(output_dir, epoch_key, model_scores, append=True):
    """ Writes a CSV with the model validation scores as a function of the epoch.

    Args:
      output_dir (str) : Full path/filename to CSV file.
      epoch_key (str) : For example, "Training set" or "Epoch {n}".
      model_scores (dict) : Contains the average NLL per molecule of
        {validation/training/generated} structures, and the average model score
        (weighted mean of above two scores).
      append (bool) : Indicates whether to append to the output file or
        start a new one. Default `True`.
    """
    validation_file_path = output_dir + "validation.csv"

    avg_nll_val = model_scores["avg_nll_val"]
    avg_nll_train = model_scores["avg_nll_train"]
    avg_nll_gen = model_scores["avg_nll_gen"]
    abs_nll_diff = model_scores["abs_nll_diff"]
    uc_jsd = model_scores["UC-JSD"]

    if not append:  # create file
        with open(validation_file_path, "w") as output_file:
            # write headeres
            output_file.write(
                f"set, avg_nll_per_molecule_val, avg_nll_per_molecule_train, "
                f"avg_nll_per_molecule_gen, abs_nll_diff, uc_jsd\n"
            )

    # append the properties of interest to the CSV file
    with open(validation_file_path, "a") as output_file:
        output_file.write(
            f"{epoch_key:}, {avg_nll_val:.5f}, {avg_nll_train:.5f}, "
            f"{avg_nll_gen:.5f}, {abs_nll_diff:.5f}, {uc_jsd:.7f}\n"
        )

    try:  # write to tensorboard
        epoch = int(epoch_key.split()[1])
        # scalars
        tb_writer.add_scalar("NLL/validation", avg_nll_val, epoch)
        tb_writer.add_scalar("NLL/training", avg_nll_train, epoch)
        tb_writer.add_scalar("NLL/generation", avg_nll_gen, epoch)
        tb_writer.add_scalar("NLL/diff", abs_nll_diff, epoch)
        tb_writer.add_scalar("NLL/uc_jsd", uc_jsd, epoch)
    except:
        pass


def write_validity(validity_file_path, validity_tensor):
    """ Writes the validity (0 or 1) of each molecule to a file in the same
    order as the molecules are written in the corresponding SMILES file.
    """
    with open(validity_file_path, "w") as valid_file:
        for valid in validity_tensor:
            valid_file.write(f"{valid}\n")
