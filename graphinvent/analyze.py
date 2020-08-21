# load general packages and functions
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import rdkit

# load program-specific functions
from parameters.constants import constants as C
import util

# functions for evaluating sets of structures, including
# sets of training, validation, and generation structures



def evaluate_model(valid_dataloader, train_dataloader, nll_per_action, model):
    """ Calculates the model score, which is the UC-JSD. Also calculates the mean
    NLL per action of the validation, training, and generated sets. Writes the
    scores to `validation.csv`.

    Args:
      valid_dataloader (torch.utils.data.dataloader.DataLoader) : Validation set
        data.
      train_dataloader (torch.utils.data.dataloader.DataLoader) : Training set
        data.
      nll_per_action (torch.Tensor) : Contains NLLs per action for one batch of
        generated structures.
      model (module.SummationMPNN) : Neural net model to evaluate.
    """
    epoch_key = util.get_last_epoch()

    # get NLL statistics of validation and training set
    print("-- Calculating NLL statistics for validation set.", flush=True)
    valid_nll_list, avg_valid_nll = get_validation_nll(dataloader=valid_dataloader,
                                                       model=model)
    print("-- Calculating NLL statistics for training set.", flush=True)
    train_nll_list, avg_train_nll = get_validation_nll(dataloader=train_dataloader,
                                                       model=model)

    # get average generated final NLL
    avg_gen_nll = torch.sum(nll_per_action) / C.n_samples

    # calculate absolute difference between all three sets
    abs_nll_diff = (abs(avg_valid_nll - avg_gen_nll) +
                    abs(avg_train_nll - avg_gen_nll) +
                    abs(avg_train_nll - avg_valid_nll))

    # initialize dictionary with NLL statistics
    model_scores = {
        "nll_val": valid_nll_list,
        "avg_nll_val": avg_valid_nll,
        "nll_train": train_nll_list,
        "avg_nll_train": avg_train_nll,
        "nll_gen": nll_per_action,
        "avg_nll_gen": avg_gen_nll,
        "abs_nll_diff": abs_nll_diff,
    }

    # get the UC-JSD and add it to the dictionary
    model_scores["UC-JSD"] = uc_jsd(nll_valid=model_scores["nll_val"],
                                    nll_train=model_scores["nll_train"],
                                    nll_sampled=model_scores["nll_gen"])

    # write results to disk
    util.write_validation_scores(output_dir=C.job_dir,
                                 epoch_key=epoch_key,
                                 model_scores=model_scores,
                                 append=bool("Epoch" in epoch_key))

    util.write_model_status(score=model_scores["UC-JSD"])


def evaluate_generated_graphs(generated_graphs, termination, nlls, start_time,
                              ts_properties, generation_batch_idx):
    """ Computes molecular properties for input set of generated graphs, saves
    results to CSV, and writes  `generated_mols` to disk as a SMILES file.
    Properties are expensive to calculate, so only done when
    `gen_batch_idx` == 0 (i.e. for the first batch of generated molecules).

    Args:
      generated_graphs (list) : Contains `GenerationGraph`s.
      termination (torch.Tensor) : Molecular termination details; contains 1 at
        index if graph from `generated_mols` was "properly" terminated, 0
        otherwise.
      nlls (torch.Tensor) : Contains final NLL of each item in `generated_mols`.
      start_time (time) : Program start time.
      ts_properties (dict) : Contains training set properties.
      gen_batch_idx (int) : Generation batch index.
    """
    epoch_key = util.get_last_epoch()

    if generation_batch_idx == 0:
        # calculate molecular properties of generated set
        prop_dict = get_molecular_properties(molecules=generated_graphs,
                                             epoch_key=epoch_key,
                                             termination=termination)
    else:
        prop_dict = {}  # initialize the property dictionary

    # add a few additional properties to the propery dictionary
    prop_dict[(epoch_key, "final_nll")] = nlls
    prop_dict[(epoch_key, "run_time")] = round(time.time() - start_time, 2)

    # output evaluation metrics to CSV
    output = C.job_dir

    # calculate validity list now, so as not to write to CSV in previous step

    epoch_id = epoch_key[6:] + "_" + str(generation_batch_idx)
    fraction_valid, validity_tensor = util.write_molecules(molecules=generated_graphs,
                                                           final_nlls=nlls,
                                                           epoch=epoch_id)

    # add these validity properties to the property dictionary
    prop_dict[(epoch_key, "fraction_valid")] = fraction_valid
    prop_dict[(epoch_key, "validity_tensor")] = validity_tensor

    # write these properties to disk, only for the first generation batch
    if generation_batch_idx == 0:
        util.properties_to_csv(prop_dict=prop_dict,
                               csv_filename=f"{output}generation.csv",
                               epoch_key=epoch_key,
                               append=True)

        # join ts properties with prop_dict for plotting
        merged_properties = {**prop_dict, **ts_properties}

        # plot properties for this epoch
        plot_filename = f"{output}generation/features{epoch_key[6:]}.png"
        plot_molecular_properties(properties_dict=merged_properties,
                                  plot_filename=plot_filename)


def evaluate_training_set(preprocessing_graphs):
    """
    Computes molecular properties for structures in training set.

    Args:
      training_graphs (list) : Contains `PreprocessingGraph`s.

    Returns:
      ts_prop_dict (dict) : Dictionary of training set molecular properties.
    """
    ts_prop_dict = get_molecular_properties(molecules=preprocessing_graphs,
                                            epoch_key="Training set")
    return ts_prop_dict


def get_edge_feature_distribution(molecular_graphs):
    """ Returns a histogram of edge features present in the input `molecular_graphs`
    (`list` of `MolecularGraph`s). The histogram is a `torch.Tensor` where
    the first item corresponds to the count of the first edge type, etc.
    The edge types correspond to those defined in `BONDTYPE_TO_INT`.
    """
    # initialize histogram
    edge_feature_hist = torch.zeros(C.n_edge_features, device="cuda")

    for molecular_graph in molecular_graphs:

        edges = molecular_graph.edge_features

        for edge in range(C.n_edge_features):
            try:  # `GenerationGraph`s
                edge_feature_hist[edge] += torch.sum(edges[:, :, edge])/2
            except TypeError:  # `PreprocessingGraph`s
                edge_feature_hist[edge] += np.sum(edges[:, :, edge])/2

    return edge_feature_hist


def get_fraction_unique(molecular_graphs):
    """ Returns the fraction (`float`) of unique graphs in `molecular_graphs`
    (`list` of `MolecularGraph`s) by comparing their canonical SMILES strings.
    """
    smiles_list = []

    for molecular_graph in molecular_graphs:

        smiles = molecular_graph.get_smiles()
        smiles_list.append(smiles)

    smiles_set = set(smiles_list)
    try:
        smiles_set.remove(None)  # remove placeholder for invalid SMILES
    except KeyError:  # no invalid SMILES in set!
        pass

    n_repeats = len(smiles_set)

    try:
        fraction_unique = n_repeats / len(smiles_list)
    except (ValueError, ZeroDivisionError):
        fraction_unique = 0

    return fraction_unique


def get_fraction_valid(molecular_graphs, termination):
    """ Determines which graphs in `molecular_graphs` (`list` of `MolecularGraph`s)
    correspond to valid molecular structures. Uses RDKit which admittedly isn't
    perfect. `termination` is a `torch.Tensor` containing 0s or 1s corresponding
    to the validity of the structures in `molecular_graphs`.

    Returns:
      fraction_valid (float) : Fraction of valid structures in the input set.
      fraction_valid_properly_terminated (float) : Fraction of valid structures
        in the input set, excluding structures which were improperly terminated.
      fraction_properly_terminated (float) : Fraction of generated structures
        which were properly terminated.
    """
    n_invalid = 0  # start counting
    n_valid_and_properly_terminated = 0  # start counting
    n_graphs = len(molecular_graphs)

    for idx, molecular_graph in enumerate(molecular_graphs):

        mol = molecular_graph.get_molecule()

        # determine if valid
        try:
            rdkit.Chem.SanitizeMol(mol)
            n_valid_and_properly_terminated += termination[idx]
        except:  # invalid molecule
            n_invalid += 1

    fraction_valid = (n_graphs - n_invalid) / n_graphs

    if 1 in termination:
        fraction_valid_properly_terminated = n_valid_and_properly_terminated / sum(termination)
    else:
        fraction_valid_properly_terminated = 0.0

    fraction_properly_terminated = sum(termination)/len(termination)

    return fraction_valid, fraction_valid_properly_terminated, fraction_properly_terminated


def get_molecular_properties(molecules, epoch_key, termination=None):
    """ Calculates properties for input `molecules` (`list` of `MolecularGraph`s).
    Properties include the distribution in number of nodes per molecule, the
    distribution of atom types, the distribution of edge features (bond types),
    the distribution of the chirality (if used), and the fraction of unique
    molecules.

    Args:
      molecules (list) : Contains either `PreprocessingGraph`s or
        `GenerationGraph`s, depending on job type.
      epoch_key (str) : For example, "Training set" or "Epoch {n}".
      termination (torch.Tensor) : If specified, molecular termination details
        for generated graphs; contains 1 at index if graph from `molecules`
        was "properly" terminated, 0 otherwise.

    Returns:
      properties_dict (dict) : Contains properties of generated and training set
        molecules. Keys are tuples of strings, e.g. ("Training set, {property}")
        or ("Epoch {n}, {property}").
    """
    # get the distribution of the number of atoms per graph
    n_nodes_hist, avg_n_nodes = get_n_nodes_distribution(molecular_graphs=molecules)

    # get the distributions of node features (e.g. atom types) in the graphs
    atom_type_hist, formal_charge_hist, numh_hist, chirality_hist = get_node_feature_distribution(molecular_graphs=molecules)

    # get the distribution of the number of edges per node and the average
    # number of edges per graph
    n_edges_hist, avg_n_edges = get_n_edges_distribution(molecular_graphs=molecules,
                                                         n_edges_to_bin=10)

    # get the distribution of bond types present in the graphs
    edge_feature_hist = get_edge_feature_distribution(molecular_graphs=molecules)

    # get the fraction of unique molecules in the input graphs
    fraction_unique = get_fraction_unique(molecular_graphs=molecules)

    if epoch_key == "Training set":
        fraction_valid = 1.0
        fraction_valid_pt = 1.0
        fraction_pt = 1.0
    else:
        # get the fraction of valid molecules in the graphs
        (
            fraction_valid,     # fraction valid
            fraction_valid_pt,  # fraction valid and properly terminated
            fraction_pt         # fraction properly terminated
        ) = get_fraction_valid(molecular_graphs=molecules, termination=termination)

    properties_dict = {
        (epoch_key, "n_nodes_hist"): n_nodes_hist,
        (epoch_key, "avg_n_nodes"): avg_n_nodes,
        (epoch_key, "atom_type_hist"): atom_type_hist,
        (epoch_key, "formal_charge_hist"): formal_charge_hist,
        (epoch_key, "n_edges_hist"): n_edges_hist,
        (epoch_key, "avg_n_edges"): avg_n_edges,
        (epoch_key, "edge_feature_hist"): edge_feature_hist,
        (epoch_key, "fraction_unique"): fraction_unique,
        (epoch_key, "fraction_valid"): fraction_valid,
        (epoch_key, "fraction_valid_properly_terminated"): fraction_valid_pt,
        (epoch_key, "fraction_properly_terminated"): fraction_pt,
        (epoch_key, "numh_hist"): numh_hist,
        (epoch_key, "chirality_hist"): chirality_hist
    }

    return properties_dict


def combine_ts_properties(prev_properties, next_properties, weight_next):
    """ Averages the properties of `prev_properties` and `next_properties` (both
    `dict`s). This is used when calculating the properties of the training set
    in separate "groups", as is done in `create_h5py_file()`.

    Args:
      prev_properties (dict) : Dictionary of old training set properties.
      next_properties (dict) : Dictionary of new training set properties.
      weight_next (int) : Weight given to `next_properties`, which is equal to
        the number of structures in the group used to calculate it (the weight
        is assumed to be `C.group_size` for `prev_properties`).

    Return:
      ts_properties (dict) : Averaged training set properties from the two
        input dictionaries.
    """
    # convert any CUDA (torch.Tensor)s to CPU
    for dictionary in [prev_properties, next_properties]:
        for key, value in dictionary.items():
            try:
                if value.is_cuda:
                    dictionary[key] = value.cpu()
            except AttributeError:
                pass

    # `weight_prev` says how much to weight the properties of the old structures
    # when calculating the average with the new structures
    weight_prev = C.group_size

    # bundle properties in a tuple for some readibility
    bundle_properties = (prev_properties, next_properties, weight_prev, weight_next)

    # take a weighted average of the "old properties" with the "new properties"
    n_nodes_hist = weighted_average(b=bundle_properties, key="n_nodes_hist")
    avg_n_nodes = weighted_average(b=bundle_properties, key="avg_n_nodes")
    atom_type_hist = weighted_average(b=bundle_properties, key="atom_type_hist")
    formal_charge_hist = weighted_average(b=bundle_properties, key="formal_charge_hist")
    n_edges_hist = weighted_average(b=bundle_properties, key="n_edges_hist")
    avg_n_edges = weighted_average(b=bundle_properties, key="avg_n_edges")
    edge_feature_hist = weighted_average(b=bundle_properties, key="edge_feature_hist")
    fraction_unique = weighted_average(b=bundle_properties, key="fraction_unique")
    fraction_valid = weighted_average(b=bundle_properties, key="fraction_valid")
    numh_hist = weighted_average(b=bundle_properties, key="numh_hist")
    chirality_hist = weighted_average(b=bundle_properties, key="chirality_hist")

    # return the weighted averages in a new dictionary
    ts_properties = {
        ("Training set", "n_nodes_hist"): n_nodes_hist,
        ("Training set", "avg_n_nodes"): avg_n_nodes,
        ("Training set", "atom_type_hist"): atom_type_hist,
        ("Training set", "formal_charge_hist"): formal_charge_hist,
        ("Training set", "n_edges_hist"): n_edges_hist,
        ("Training set", "avg_n_edges"): avg_n_edges,
        ("Training set", "edge_feature_hist"): edge_feature_hist,
        ("Training set", "fraction_unique"): fraction_unique,
        ("Training set", "fraction_valid"): fraction_valid,
        ("Training set", "numh_hist"): numh_hist,
        ("Training set", "chirality_hist"): chirality_hist
    }

    return ts_properties


def weighted_average(b, key):
    """Takes a weighted average of two training set property dictionaries.

    Args:
      b (tuple) : Bundle of the following four items:
        p (dict) : "Previous" dictionary.
        n (dict) : "Next" dictionary.
        wp (int) : Weight for `p`.
        wn (int) : Weight for `n`.
      key (str) : 2nd string in the tuple keys.

    Returns:
      weighted_average (dict) : Dictionary is weighted average of `p` and `n`.
    """
    (p, n, wp, wn) = b

    weighted_average = np.around((
        np.array(p[("Training set", key)]) * wp
        + np.array(n[("Training set", key)]) * wn
    ) / (wp + wn), decimals=3)

    return weighted_average


def get_n_edges_distribution(molecular_graphs, n_edges_to_bin=10):
    """ Returns a histogram of the number of edges per node present in the
    `molecular_graphs` (`list` of `MolecularGraph`s). The histogram is a `list`
    where the first item corresponds to the count of the number of nodes with one
    edge, the second item to the count of the number of nodes with two edges,
    etc, up until the count of the number of nodes with `n_edges_to_bin` edges
    (default 10). Also returns the average number of edges per node.
    """
    # initialize and populate histogram (last bin is for # num edges > `n_edges_to_bin`)
    n_edges_histogram = torch.zeros(n_edges_to_bin, device="cuda")

    for molecular_graph in molecular_graphs:

        edges = molecular_graph.edge_features

        for vi in range(molecular_graph.n_nodes):

            n_edges = 0
            for bond_type in range(C.n_edge_features):
                try:
                    n_edges += int(torch.sum(edges[vi, :, bond_type]))
                except TypeError:  # if edges is `np.ndarray`
                    n_edges += int(np.sum(edges[vi, :, bond_type]))

            if n_edges > n_edges_to_bin:
                n_edges = n_edges_to_bin
            n_edges_histogram[n_edges - 1] += 1

    # compute average number of edges per node
    sum_n_edges = 0
    for n_edges, count in enumerate(n_edges_histogram, start=1):
        sum_n_edges += n_edges * count

    try:
        avg_n_edges = sum_n_edges / torch.sum(n_edges_histogram, dim=0)
    except ValueError:
        avg_n_edges = 0

    return n_edges_histogram, avg_n_edges


def get_n_nodes_distribution(molecular_graphs):
    """ Returns a histogram of the number of nodes per graph present in the
    `molecular_graphs` (`list` of `MolecularGraph`s). The histogram is a `list`
    where the first item corresponds to the count of the number of graphs with
    one node, the second item corresponds to the count of the number of graphs
    with two nodes, etc, up until the count of the number of graphs with the
    largest number of nodes. Also returns the average number of nodes per graph.
    """
    # initialize and populate histogram
    n_nodes_histogram = torch.zeros(C.max_n_nodes + 1, device="cuda")

    for molecular_graph in molecular_graphs:
        n_nodes = molecular_graph.n_nodes
        n_nodes_histogram[n_nodes] += 1

    # compute the average number of nodes per graph
    sum_n_nodes = 0
    for key, count in enumerate(n_nodes_histogram):
        n_nodes = key
        sum_n_nodes += n_nodes * count

    avg_n_nodes = sum_n_nodes / len(molecular_graphs)

    return n_nodes_histogram, avg_n_nodes


def get_node_feature_distribution(molecular_graphs):
    """ Returns a `tuple` of histograms (`torch.Tensor`s) for atom types, formal
    charges, number of implicit Hs, and chiral states that are present in the
    input `molecular_graphs` (`list` of `MolecularGraph`s). Each histogram is a
    `list` where the nth item corresponds to the count of the nth property in
    `atom_types`, `formal_charge`, `imp_H`, and `chirality`, respectively.
    """
    # sum up all node feature vectors to get an un-normalized histogram
    if type(molecular_graphs[0].node_features) == torch.Tensor:
        nodes_hist = torch.zeros(C.n_node_features, device="cuda")
    else:
        nodes_hist = np.zeros(C.n_node_features)

    # loop over all the node feature matrices of the input `TrainingGraph`s
    for molecular_graph in molecular_graphs:
        try:
            nodes_hist += torch.sum(molecular_graph.node_features, dim=0)
        except TypeError:
            nodes_hist += np.sum(molecular_graph.node_features, axis=0)

    idc = util.get_feature_vector_indices()  # **note: "idc" == "indices"

    # split up `nodes_hist` into atom types hist, formal charge hist, etc
    # `atom_type_histogram` and `formal_charge_histogram` are calculated by
    # default, and if specified, also `numh_histogram` and `chirality_histogram`
    atom_type_histogram = nodes_hist[:idc[0]]

    formal_charge_histogram = nodes_hist[idc[0]:idc[1]]

    if not C.use_explicit_H and not C.ignore_H:
        numh_histogram = nodes_hist[idc[1]:idc[2]]
    else:
        numh_histogram = [0] * C.n_imp_H

    if C.use_chirality:
        correction = int(not C.use_explicit_H and not C.ignore_H)
        chirality_histogram = nodes_hist[idc[1 + correction]:idc[2 + correction]]
    else:
        chirality_histogram = [0] * C.n_chirality

    return (atom_type_histogram, formal_charge_histogram, numh_histogram, chirality_histogram)


def get_validation_nll(dataloader, model):
    """ Computes validation NLL (e.g. the NLL for taking the "correct" action
    for a specific fragment/atom) for graphs in the validation and training sets
    (whichever is specified by the `dataloader`). The subsets are equal in size
    to the number of structures generated per batch (`n_samples` below). Note:
    do not use for generation set structures, as there is no "correct" action!

    Returns:
      nlls (torch.Tensor) : Contains all NLLs per action for generating a set of
        molecules via the "correct" set of actions.
      avg_final_nll (torch.Tensor) : Contains average final NLLs for generating
        a set of molecules via the "correct" set of actions.
    """
    Softmax = torch.nn.Softmax(dim=1)
    n_samples = min(100000, C.n_samples)  #  number of dataloader structures to evaluate
    nlls = torch.zeros(n_samples*(C.max_n_nodes+5), device="cuda")
    n_structures = torch.zeros(1, device="cuda")

    # `batch` contains C.n_samples subgraphs during validation
    for idx, batch in enumerate(dataloader):

        # for really large dataloaders (like that of the training set), the line
        # below ensures that the validation NLL is only calculated until the
        # number of structures analyzed is roughly equivalent to the number of
        # structures generated, purely for speed
        if idx * C.batch_size > n_samples:
            break

        batch = [b.cuda(non_blocking=True) for b in batch]
        nodes, edges, target_output = batch

        # index [0] takes the first tensor output of torch.max (containing the
        # max values), and ignores the second tensor which contains the indices
        renormalized_target_output = target_output/(
            torch.max(target_output, dim=1)[0].unsqueeze(dim=1)
        )

        # return the output and normalize
        normalized_output = Softmax(model(nodes, edges))

        # multiplication with `target_output` zeros out the "incorrect" actions
        one_nonzero_elem_vector = torch.mul(renormalized_target_output, normalized_output)

        likelihood = torch.sum(one_nonzero_elem_vector, dim=1)

        nll = -1 * torch.log(likelihood)

        # len(nll) is always same as `C.batch_size`
        nlls[idx*C.batch_size:(idx*C.batch_size + len(nll))] = nll

        # in computing the number of structures, it is important to use
        # `target_output` and not `renormalized_target_output` (unnormalized
        # means the sum is the number of subgraphs)
        n_structures += torch.sum(target_output[:, -1])

    # below equivalent to avg NLL per molecule () are shuffled in training
    # dataloader so needs to be done this way)
    avg_final_nll = torch.sum(nlls, dim=0) / n_structures

    return nlls, avg_final_nll[0]  # only need the first item of tensor


def plot_molecular_properties(properties_dict, plot_filename):
    """ Plots a 3 by 3 grid of the histograms in `properties_dict` using
    separate colors for the training set and for each epoch.

    Args:
      properties_dict (dict) : Contains properties of generated and training
        set molecules. Only plots histogram properties, not averages.
      plot_filename (str) : Full path/filename for saving output PNG.
    """
    # start the grid
    n_plots_y = 3
    n_plots_x = 3
    matplotlib.rc("figure", figsize=(8.0, 7.0))
    fig, ax = plt.subplots(n_plots_y, n_plots_x, sharey="all")
    fig.subplots_adjust(hspace=0.6, wspace=0.4)

    ax_nn = ax[0, 0]  # number of nodes
    ax_at = ax[0, 1]  # atom types
    ax_fc = ax[0, 2]  # formal charges
    ax_nh = ax[1, 0]  # num implicit Hs
    ax_ne = ax[1, 1]  # number of edges
    ax_bt = ax[1, 2]  # here, bond type == edge feature
    ax_ct = ax[2, 0]  # chirality

    # get the keys of the properties to plot
    keys_to_plot = list(set([key[0] for key in properties_dict.keys()]))

    # plot the results for the training set and for each epoch
    for epoch_key in keys_to_plot:

        # set the plot labels
        if epoch_key == "Training set":
            m, c, ls = "*", "goldenrod", "-"
        else:
            m, c, ls = "o", "cadetblue", "--"

        # normalize so that all can share one y-axis
        (norm_n_nodes_hist, norm_atom_type_hist, norm_formal_charge_hist,
         norm_numh_hist, norm_n_edges_hist, norm_edge_feature_hist,
         norm_chirality_hist) = util.normalize_evaluation_metrics(prop_dict=properties_dict,
                                                                  epoch_key=epoch_key)

        # plot num nodes histogram
        ax_nn.plot(range(1, len(norm_n_nodes_hist) + 1), norm_n_nodes_hist,
                   color=c, label=epoch_key, linestyle=ls, marker=m)
        ax_nn.set(xlabel="Num nodes per graph")

        # plot atom type histogram
        ax_at.plot(range(1, len(norm_atom_type_hist) + 1), norm_atom_type_hist,
                   color=c, label=epoch_key, linestyle=ls, marker=m)
        xlabel_values = ", ".join(map(str, C.atom_types))
        ax_at.set(xlabel=f"Atom type ({xlabel_values})")

        # plot formal charge histogram
        ax_fc.plot(range(C.formal_charge[0], C.formal_charge[-1] + 1),
                   norm_formal_charge_hist,
                   color=c, label=epoch_key, linestyle=ls, marker=m)
        xlabel_values = ", ".join(map(str, C.formal_charge))
        ax_fc.set(xlabel=f"Formal charge ({xlabel_values})")

        # plot num H histogram
        ax_nh.plot(C.imp_H, norm_numh_hist,
                   color=c, label=epoch_key, linestyle=ls, marker=m)
        xlabel_values = ", ".join(map(str, C.imp_H))
        ax_nh.set(xlabel=f"Num implicit Hs ({xlabel_values})", ylabel="Fractional count")

        # plot n_edges histogram
        ax_ne.plot(range(1, len(norm_n_edges_hist) + 1), norm_n_edges_hist,
                   color=c, label=epoch_key, linestyle=ls, marker=m)
        ax_ne.set(xlabel="Num edges per node")

        # plot bond type/edge feature histogram
        ax_bt.plot(range(0, len(norm_edge_feature_hist)),
                   norm_edge_feature_hist,
                   color=c, label=epoch_key, linestyle=ls, marker=m)
        xlabel_values = ", ".join(map(str, C.int_to_bondtype))
        ax_bt.set(xlabel=f"Bond type ({xlabel_values})")

        # plot chirality histogram
        ax_ct.plot(range(1, len(norm_chirality_hist) + 1), norm_chirality_hist,
                   color=c, label=epoch_key, linestyle=ls, marker=m)
        xlabel_values = ", ".join(map(str, C.chirality))
        ax_ct.set(xlabel=f"Chirality ({xlabel_values})")

        # put the legend in the bottom right corner regardless
        ax_ct.legend(loc="upper right", prop={"size": 6})

    ax = util.turn_off_empty_axes(n_plots_y, n_plots_x, ax)

    fig.savefig(plot_filename)
    plt.close()


def uc_jsd(nll_valid, nll_train, nll_sampled):
    """ Computes the UC-JSD (metric used for the benchmark of generative models
    in Arús-Pous, J. et al., J. Chem. Inf., 2019, 1-13).

    Args:
      nll_valid (torch.Tensor) : Contains NLLs for sampling the correct action
        of structures in the validation set.
      nll_train (torch.Tensor) : Contains NLLs for sampling the correct action
        of structures in the training set.
      nll_sampled (torch.Tensor) : Contains NLLs for sampling the correct action
        of structures in the generated set.
    """
    min_len = min(len(nll_valid), len(nll_sampled), len(nll_train))

    # make all the distributions the same length (dim=0)
    nll_valid_norm = nll_valid[:min_len] / torch.sum(nll_valid[:min_len])
    nll_train_norm = nll_train[:min_len] / torch.sum(nll_train[:min_len])
    nll_sampled_norm = nll_sampled[:min_len] / torch.sum(nll_sampled[:min_len])

    nll_sum = (nll_valid_norm + nll_train_norm + nll_sampled_norm) / 3

    uc_jsd = (torch.nn.functional.kl_div(nll_valid_norm, nll_sum)
              + torch.nn.functional.kl_div(nll_train_norm, nll_sum)
              + torch.nn.functional.kl_div(nll_sampled_norm, nll_sum)) / 3

    return float(uc_jsd)
