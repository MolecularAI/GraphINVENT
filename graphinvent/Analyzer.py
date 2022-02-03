"""
The `Analyzer` contains functions for evaluating sets of structures, including
sets of training, validation, and sampled structures.
"""
# load general packages and functions
from typing import Union, Tuple
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import rdkit
from torch.utils.tensorboard import SummaryWriter

# load GraphINVENT-specific functions
from parameters.constants import constants
import util


class Analyzer:
    """
    Class for analyzing different datasets, including the training set,
    validation set, and sampled structures.
    """

    def __init__(self,
                 valid_dataloader : Union[torch.utils.data.DataLoader, None]=None,
                 train_dataloader : Union[torch.utils.data.DataLoader, None]=None,
                 start_time : Union[time.time, None]=None) -> None:

        self.valid_dataloader = valid_dataloader
        self.train_dataloader = train_dataloader
        self.start_time       = start_time
        self.tb_writer        = SummaryWriter(log_dir=constants.tensorboard_dir,
                                              flush_secs=10)

        self.model = None  # placeholder

    def evaluate_model(self, likelihood_per_action : torch.Tensor) -> None:
        """
        Calculates the model score, which is the UC-JSD. Also calculates the
        mean NLL per action of the validation, training, and generated sets.
        Writes the scores to `validation.log`.

        Args:
        ----
            likelihood_per_action (torch.Tensor) : Contains NLLs per action a
              batch of generated graphs.
        """
        def _uc_jsd(likelihood_valid : torch.Tensor,
                    likelihood_train : torch.Tensor,
                    likelihood_sampled : torch.Tensor) -> float:
            """
            Computes the UC-JSD (metric used for the benchmark of generative
            models in Arús-Pous, J. et al., J. Chem. Inf., 2019, 1-13).

            Args:
            ----
                likelihood_valid (torch.Tensor)   : NLLs for correct actions in
                                                    validation set.
                likelihood_train (torch.Tensor)   : NLLs for correct actions in
                                                    training set.
                likelihood_sampled (torch.Tensor) : NLLs for sampled actions in
                                                    the generated set.

            Returns:
            -------
                uc_jsd (float) : UC-JSD.
            """
            min_len = min(len(likelihood_valid),
                          len(likelihood_sampled),
                          len(likelihood_train))

            # make all the distributions the same length (dim=0)
            likelihood_valid_norm   = (likelihood_valid[:min_len] /
                                       torch.sum(likelihood_valid[:min_len]))
            likelihood_train_norm   = (likelihood_train[:min_len] /
                                       torch.sum(likelihood_train[:min_len]))
            likelihood_sampled_norm = (likelihood_sampled[:min_len] /
                                       torch.sum(likelihood_sampled[:min_len]))

            likelihood_sum = (
                (likelihood_valid_norm +
                 likelihood_train_norm +
                 likelihood_sampled_norm) / 3
            )

            uc_jsd = (
                torch.nn.functional.kl_div(likelihood_valid_norm, likelihood_sum)
                + torch.nn.functional.kl_div(likelihood_train_norm, likelihood_sum)
                + torch.nn.functional.kl_div(likelihood_sampled_norm, likelihood_sum)
            ) / 3

            return float(uc_jsd)

        epoch_key = util.get_last_epoch()
        if constants.job_type == "fine-tune":
            epoch_label = "Step"
        else:
            epoch_label = "Epoch"

        print("-- Calculating NLL statistics for validation set.", flush=True)
        valid_likelihood_list, avg_valid_likelihood = \
            self.get_validation_likelihood(dataset="validation")

        print("-- Calculating NLL statistics for training set.", flush=True)
        train_likelihood_list, avg_train_likelihood = \
            self.get_validation_likelihood(dataset="training")

        # get average final NLL for the generation set
        avg_gen_likelihood = (
            torch.sum(likelihood_per_action) / constants.n_samples
        )

        # initialize dictionary with NLL statistics
        model_scores = {
            "likelihood_val"      : valid_likelihood_list,
            "avg_likelihood_val"  : avg_valid_likelihood,
            "likelihood_train"    : train_likelihood_list,
            "avg_likelihood_train": avg_train_likelihood,
            "likelihood_gen"      : likelihood_per_action,
            "avg_likelihood_gen"  : avg_gen_likelihood,
        }

        # get the UC-JSD and add it to the dictionary
        model_scores["UC-JSD"] = _uc_jsd(
            likelihood_valid=model_scores["likelihood_val"],
            likelihood_train=model_scores["likelihood_train"],
            likelihood_sampled=model_scores["likelihood_gen"]
        )

        # write results to disk
        util.write_validation_scores(
            output_dir=constants.job_dir,
            epoch_key=epoch_key,
            model_scores=model_scores,
            append=bool(epoch_key != f"{epoch_label} {constants.sample_every}")
        )
        util.write_training_status(score=model_scores["UC-JSD"])

    def evaluate_generated_graphs(self, generated_graphs : list,
                                  termination : torch.Tensor,
                                  loglikelihoods : torch.Tensor,
                                  ts_properties : dict,
                                  generation_batch_idx : int) -> None:
        """
        Computes molecular properties for input set of generated graphs, saves
        results to CSV, and writes `generated_graphs` to disk as a SMILES file.
        Properties are expensive to calculate, so only done for the first batch
        of generated molecules.

        Args:
        ----
            generated_graphs (list)    : Contains `GenerationGraph`s.
            termination (torch.Tensor) : Molecular termination details; contains
                                         1 at index if graph from `generated_graphs`
                                         was "properly" terminated, 0 otherwise.
            likelihoods (torch.Tensor) : Contains final NLL of each item in
                                         `generated_graphs`.
            ts_properties (dict)       : Contains training set properties.
            generation_batch_idx (int) : Generation batch index.
        """
        epoch_key = util.get_last_epoch()
        if constants.job_type == "fine-tune":
            epoch_label = "Step"
        else:
            epoch_label = "Epoch"

        if generation_batch_idx == 0:
            # calculate molecular properties of generated set
            prop_dict = self.get_molecular_properties(molecules=generated_graphs,
                                                      epoch_key=epoch_key,
                                                      termination=termination)
        else:
            prop_dict = {}  # initialize the property dictionary

        # add a few additional properties to the propery dictionary
        prop_dict[(epoch_key, "final_likelihood")] = loglikelihoods
        prop_dict[(epoch_key, "run_time")]         = round(time.time() - self.start_time, 2)

        # calculate validity list now, so as not to write to CSV in previous step
        epoch_id = epoch_label + " " + str(generation_batch_idx)
        fraction_valid, validity_tensor, _ = util.write_molecules(
            molecules=generated_graphs,
            final_likelihoods=loglikelihoods,
            epoch=epoch_id,
            write=True,
            label="training",
        )
        prop_dict[(epoch_key, "fraction_valid")]  = fraction_valid
        prop_dict[(epoch_key, "validity_tensor")] = validity_tensor

        # write these properties to disk, only for the first generation batch
        if generation_batch_idx == 0:
            output = constants.job_dir  # shorthand
            util.properties_to_csv(prop_dict=prop_dict,
                                   csv_filename=f"{output}generation.log",
                                   epoch_key=epoch_key,
                                   append=True)

            # join ts properties with prop_dict for plotting
            merged_properties = {**prop_dict, **ts_properties}

            # plot properties for this epoch
            plot_filename = f"{output}generation/features{epoch_key[6:]}.png"
            self.plot_molecular_properties(properties=merged_properties,
                                           plot_filename=plot_filename)

    def evaluate_generated_graphs_rl(self, generated_graphs : list,
                                     termination : torch.Tensor,
                                     agent_loglikelihoods : torch.Tensor,
                                     prior_loglikelihoods : torch.Tensor,
                                     ts_properties : dict,
                                     step : int, is_agent : bool=False,
                                     label : str="") -> \
                                     Union[torch.Tensor, torch.Tensor]:
        """
        Computes molecular properties for input set of generated graphs, saves
        results to CSV, and writes `generated_graphs` to disk as a SMILES file.
        Properties are expensive to calculate, so only done for the first batch
        of generated molecules.

        Args:
        ----
            generated_graphs (list)             : Contains `GenerationGraph`s.
            termination (torch.Tensor)          : Molecular termination details;
                                                  contains 1 at index if graph from
                                                  `generated_graphs` was "properly"
                                                  terminated, 0 otherwise.
            agent_loglikelihoods (torch.Tensor) : Contains final NLL of each item
                                                  in `generated_graphs` (agent).
            prior_loglikelihoods (torch.Tensor) : Contains final NLL of each item
                                                  in `generated_graphs` (prior).
            ts_properties (dict)                : Contains training set properties.
            step (int)                          : Training step.
            is_agent (bool)                     : Indicates whether the `agent_loglikelihoods`
                                                  correspond to the agent.
            label (str)                         : Label to use for saving files.

        Returns:
        -------
            validity (torch.Tensor)   : Indicates the validity of the generated
                                        structures with a 1 for valid, 0 for invalid.
            uniqueness (torch.Tensor) : Indicates the uniqueness of the generated
                                        structures with a 1 for unique (and or first
                                        duplicate), and 0 for duplicate.
        """
        #epoch_key = util.get_last_epoch()
        epoch_key = f"Step {step} {label}"

        # calculate molecular properties of generated set
        prop_dict = self.get_molecular_properties(molecules=generated_graphs,
                                                  epoch_key=epoch_key,
                                                  termination=termination)

        # add a few additional properties to the propery dictionary
        prop_dict[(epoch_key, "final_agent_loglikelihood")] = agent_loglikelihoods
        prop_dict[(epoch_key, "final_prior_loglikelihood")] = prior_loglikelihoods
        prop_dict[(epoch_key, "run_time")]                  = round(time.time()
                                                                    - self.start_time, 2)

        # calculate validity list now, so as not to write to CSV in previous step
        fraction_valid, validity, uniqueness = util.write_molecules(
            molecules=generated_graphs,
            final_likelihoods=agent_loglikelihoods,
            epoch=epoch_key,
            write=is_agent,
            label=label
        )
        prop_dict[(epoch_key, "fraction_valid")]    = fraction_valid
        prop_dict[(epoch_key, "validity_tensor")]   = validity
        prop_dict[(epoch_key, "uniqueness_tensor")] = uniqueness

        # write these properties to disk, only for the first generation batch
        output            = constants.job_dir
        util.properties_to_csv(prop_dict=prop_dict,
                               csv_filename=f"{output}generation.log",
                               epoch_key=epoch_key,
                               append=True)

        # join ts properties with prop_dict for plotting
        merged_properties = {**prop_dict, **ts_properties}

        # plot properties for this epoch
        plot_label        = epoch_key[5:].replace(" ", "_")
        plot_filename     = f"{output}generation/features{plot_label}.png"
        self.plot_molecular_properties(properties=merged_properties,
                                       plot_filename=plot_filename)

        return validity, uniqueness

    def evaluate_training_set(self, preprocessing_graphs : list) -> dict:
        """
        Computes molecular properties for structures in training set.

        Args:
        ----
            preprocessing_graphs (list) : Contains `PreprocessingGraph`s.

        Returns:
        -------
            training_set_properties (dict) : Dictionary of training set
                                             molecular properties.
        """
        training_set_properties = self.get_molecular_properties(
            molecules=preprocessing_graphs,
            epoch_key="Training set"
        )
        return training_set_properties

    def get_molecular_properties(self, molecules : list, epoch_key : str,
                                 termination : Union[torch.Tensor, None]=None) \
                                 -> dict:
        """
        Calculates properties for input `molecules` (`list` of
        `MolecularGraph`s). Properties include the distribution in number of
        nodes per molecule, the distribution of atom types, the distribution of
        edge features (bond types), the distribution of the chirality (if used),
        and the fraction of unique molecules.

        Args:
        ----
            molecules (list)           : `PreprocessingGraph`s or `GenerationGraph`s,
                                          depending on job type.
            epoch_key (str)            : For example, "Training set" or "Epoch {n}".
            termination (torch.Tensor) : If specified, contains molecular termination
                                         details for generated graphs; contains 1
                                         at index if graph was "properly" terminated,
                                         0 otherwise.

        Returns:
        -------
            properties (dict) : Contains properties of generated and training
                                set molecules. Keys are string tuples, e.g. ("Training set",
                                "{property}") or ("Epoch {n}", "{property}").
        """
        def _get_n_edges_distribution(molecular_graphs : list,
                                      n_edges_to_bin : int=10) -> \
                                      Tuple[torch.Tensor, float]:
            """
            Returns a histogram of the number of edges per node present in the
            `molecular_graphs`. The histogram is a `list` where the first item
            corresponds to the count of the number of nodes with one edge, the
            second item to the count of the number of nodes with two edges, etc,
            up until the count of the number of nodes with `n_edges_to_bin`
            edges. Also returns the average number of edges per node.
            """
            # initialize and populate histogram (last bin is for # num edges >
            # `n_edges_to_bin`)
            n_edges_histogram = torch.zeros(n_edges_to_bin,
                                            device=constants.device)
            for molecular_graph in molecular_graphs:
                edges = molecular_graph.edge_features
                for node_idx in range(molecular_graph.n_nodes):
                    n_edges = 0
                    for bond_type in range(constants.n_edge_features):
                        try:
                            n_edges += int(
                                torch.sum(edges[node_idx, :, bond_type])
                            )
                        except TypeError:  # if edges is `np.ndarray`
                            n_edges += int(
                                np.sum(edges[node_idx, :, bond_type])
                            )
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

        def _get_n_nodes_distribution(molecular_graphs : list) -> \
                                      Tuple[torch.Tensor, float]:
            """
            Returns a histogram of the number of nodes per graph present in the
            `molecular_graphs`. The histogram is a `list` where the first item
            corresponds to the count of the number of graphs with one node, the
            second item corresponds to the count of the number of graphs with
            two nodes, etc, up until the count of the number of graphs with the
            largest number of nodes. Also returns the average number of nodes
            per graph.
            """
            # initialize histogram
            n_nodes_histogram = torch.zeros(constants.max_n_nodes + 1,
                                            device=constants.device)

            # populate histogram
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

        def _get_node_feature_distribution(molecular_graphs : list) -> \
                                           Tuple[Union[torch.Tensor, np.ndarray], ...]:
            """
            Returns a `tuple` of histograms (`torch.Tensor`s) for atom types,
            formal charges, number of implicit Hs, and chiral states that are
            present in the input `molecular_graphs`. Each histogram is a `list`
            where the nth item corresponds to the count of the nth property in
            `atom_types`, `formal_charge`, `imp_H`, and `chirality`.
            """
            # sum up all node feature vectors to get an un-normalized histogram
            if isinstance(molecular_graphs[0].node_features, torch.Tensor):
                nodes_hist = torch.zeros(constants.n_node_features,
                                         device=constants.device)
            else:
                nodes_hist = np.zeros(constants.n_node_features)

            # loop over all the node feature matrices of the input `TrainingGraph`s
            for molecular_graph in molecular_graphs:
                try:
                    nodes_hist += torch.sum(molecular_graph.node_features, dim=0)
                except TypeError:
                    nodes_hist += np.sum(molecular_graph.node_features, axis=0)

            idc = util.get_feature_vector_indices()  # **note: "idc" == "indices"

            # split up `nodes_hist` into atom types hist, formal charge hist, etc
            atom_type_histogram     = nodes_hist[:idc[0]]
            formal_charge_histogram = nodes_hist[idc[0]:idc[1]]
            if not constants.use_explicit_H and not constants.ignore_H:
                numh_histogram      = nodes_hist[idc[1]:idc[2]]
            else:
                numh_histogram      = [0] * constants.n_imp_H
            if constants.use_chirality:
                correction          = int(
                    not constants.use_explicit_H and not constants.ignore_H
                )
                chirality_histogram = nodes_hist[
                    idc[1 + correction]:idc[2 + correction]
                ]
            else:
                chirality_histogram = [0] * constants.n_chirality

            return (atom_type_histogram,
                    formal_charge_histogram,
                    numh_histogram,
                    chirality_histogram)

        def _get_edge_feature_distribution(molecular_graphs : list) -> \
            torch.Tensor:
            """
            Returns a histogram of edge features present in the input
            `molecular_graphs`. The histogram is a `torch.Tensor` where the
            first item corresponds to the count of the first edge type, etc. The
            edge types correspond to those defined in `BONDTYPE_TO_INT`.
            """
            # initialize and populate the histogram
            edge_feature_hist = torch.zeros(constants.n_edge_features,
                                            device=constants.device)

            for molecular_graph in molecular_graphs:
                edges = molecular_graph.edge_features
                for edge in range(constants.n_edge_features):
                    try:               # `GenerationGraph`s
                        edge_feature_hist[edge] += torch.sum(edges[:, :, edge])/2
                    except TypeError:  # `PreprocessingGraph`s
                        edge_feature_hist[edge] += np.sum(edges[:, :, edge])/2
            return edge_feature_hist

        def _get_fraction_unique(molecular_graphs : list) -> float:
            """
            Returns the fraction of unique graphs in `molecular_graphs`by
            comparing their canonical SMILES strings.
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

        def _get_fraction_valid(molecular_graphs : list,
                                termination : torch.Tensor) -> Tuple[float, ...]:
            """
            Determines which graphs in `molecular_graphs` correspond to valid
            molecular structures. Uses RDKit which admittedly isn't perfect.
            `termination` is a `torch.Tensor` containing 0s or 1s corresponding
            to the validity of the structures in `molecular_graphs`.

            Returns:
            -------
                fraction_valid (float)               : Fraction of valid structures
                                                       in the input set.
                fraction_valid_properly_terminated (float) : Fraction of valid structures
                                                             in the input set, excluding
                                                             structures which were
                                                             improperly terminated.
                fraction_properly_terminated (float) : Fraction of generated structures
                                                       which were properly terminated.
            """
            n_invalid                       = 0  # start counting
            n_valid_and_properly_terminated = 0  # start counting
            n_graphs                        = len(molecular_graphs)

            for idx, molecular_graph in enumerate(molecular_graphs):
                mol = molecular_graph.get_molecule()
                # determine if valid
                try:
                    rdkit.Chem.SanitizeMol(mol)
                    n_valid_and_properly_terminated += int(termination[idx])
                except:  # invalid molecule
                    n_invalid += 1
            fraction_valid = (n_graphs - n_invalid) / n_graphs
            if 1 in termination:
                fraction_valid_properly_terminated = (
                    n_valid_and_properly_terminated / torch.sum(termination)
                )
            else:
                fraction_valid_properly_terminated = 0.0
            fraction_properly_terminated = (
                torch.sum(termination)/len(termination)
            )
            return (fraction_valid,
                    fraction_valid_properly_terminated,
                    fraction_properly_terminated)

        # get the distribution of the number of atoms per graph
        n_nodes_hist, avg_n_nodes = _get_n_nodes_distribution(
            molecular_graphs=molecules
        )

        # get the distributions of node features (e.g. atom types) in the graphs
        atom_type_hist, formal_charge_hist, numh_hist, chirality_hist = \
            _get_node_feature_distribution(molecular_graphs=molecules)

        # get the distribution of the number of edges per node and the average
        # number of edges per graph
        n_edges_hist, avg_n_edges = _get_n_edges_distribution(
            molecular_graphs=molecules,
            n_edges_to_bin=10
        )

        # get the distribution of bond types present in the graphs
        edge_feature_hist         = _get_edge_feature_distribution(
            molecular_graphs=molecules
        )

        # get the fraction of unique molecules in the input graphs
        fraction_unique           = _get_fraction_unique(molecular_graphs=molecules)

        if epoch_key == "Training set":
            # for the training set, we assume everything is valid (otherwise,
            # what are you doing)
            fraction_valid, fraction_valid_pt, fraction_pt = 1.0, 1.0, 1.0
        else:
            # get the fraction of valid molecules in the graphs
            (
                fraction_valid,     # fraction valid
                fraction_valid_pt,  # fraction valid and properly terminated
                fraction_pt         # fraction properly terminated
            )                     = _get_fraction_valid(molecular_graphs=molecules,
                                                        termination=termination)

        properties = {
            (epoch_key, "n_nodes_hist")                      : n_nodes_hist,
            (epoch_key, "avg_n_nodes")                       : avg_n_nodes,
            (epoch_key, "atom_type_hist")                    : atom_type_hist,
            (epoch_key, "formal_charge_hist")                : formal_charge_hist,
            (epoch_key, "n_edges_hist")                      : n_edges_hist,
            (epoch_key, "avg_n_edges")                       : avg_n_edges,
            (epoch_key, "edge_feature_hist")                 : edge_feature_hist,
            (epoch_key, "fraction_unique")                   : fraction_unique,
            (epoch_key, "fraction_valid")                    : fraction_valid,
            (epoch_key, "fraction_valid_properly_terminated"): fraction_valid_pt,
            (epoch_key, "fraction_properly_terminated")      : fraction_pt,
            (epoch_key, "numh_hist")                         : numh_hist,
            (epoch_key, "chirality_hist")                    : chirality_hist
        }

        return properties

    def combine_ts_properties(self, prev_properties : dict,
                              next_properties : dict,
                              weight_next : int) -> dict:
        """
        Averages the properties of `prev_properties` and `next_properties` (both
        dictionaries). This is used when calculating the properties of the
        training set in separate "groups", as is done during preprocessing.

        Args:
        ----
            prev_properties (dict) : Dictionary of old training set properties.
            next_properties (dict) : Dictionary of new training set properties.
            weight_next (int)      : Weight given to `next_properties`, equal to
                                     the number of graphs in the group used to calculate
                                     it (the weight is assumed to be `constants.batch_size`
                                     for `prev_properties`).

        Returns:
        -------
            ts_properties (dict) : Averaged training set properties from the two
              input dictionaries.
        """
        # convert any CUDA (torch.Tensor)s to CPU tensors
        for dictionary in [prev_properties, next_properties]:
            for key, value in dictionary.items():
                try:
                    if value.is_cuda:
                        dictionary[key] = value.cpu()
                except AttributeError:
                    pass

        # `weight_prev` says how much to weight the properties of the old structures
        weight_prev = constants.batch_size

        # bundle properties in a tuple for some readibility
        bundle_properties  = (
            prev_properties, next_properties, weight_prev, weight_next
        )

        # take a weighted average of the "old properties" with the "new properties"
        n_nodes_hist       = self.weighted_average(b=bundle_properties,
                                                   key="n_nodes_hist")
        avg_n_nodes        = self.weighted_average(b=bundle_properties,
                                                   key="avg_n_nodes")
        atom_type_hist     = self.weighted_average(b=bundle_properties,
                                                   key="atom_type_hist")
        formal_charge_hist = self.weighted_average(b=bundle_properties,
                                                   key="formal_charge_hist")
        n_edges_hist       = self.weighted_average(b=bundle_properties,
                                                   key="n_edges_hist")
        avg_n_edges        = self.weighted_average(b=bundle_properties,
                                                   key="avg_n_edges")
        edge_feature_hist  = self.weighted_average(b=bundle_properties,
                                                   key="edge_feature_hist")
        fraction_unique    = self.weighted_average(b=bundle_properties,
                                                   key="fraction_unique")
        fraction_valid     = self.weighted_average(b=bundle_properties,
                                                   key="fraction_valid")
        numh_hist          = self.weighted_average(b=bundle_properties,
                                                   key="numh_hist")
        chirality_hist     = self.weighted_average(b=bundle_properties,
                                                   key="chirality_hist")

        # return the weighted averages in a new dictionary
        ts_properties = {
            ("Training set", "n_nodes_hist")      : n_nodes_hist,
            ("Training set", "avg_n_nodes")       : avg_n_nodes,
            ("Training set", "atom_type_hist")    : atom_type_hist,
            ("Training set", "formal_charge_hist"): formal_charge_hist,
            ("Training set", "n_edges_hist")      : n_edges_hist,
            ("Training set", "avg_n_edges")       : avg_n_edges,
            ("Training set", "edge_feature_hist") : edge_feature_hist,
            ("Training set", "fraction_unique")   : fraction_unique,
            ("Training set", "fraction_valid")    : fraction_valid,
            ("Training set", "numh_hist")         : numh_hist,
            ("Training set", "chirality_hist")    : chirality_hist
        }
        return ts_properties

    def weighted_average(self, b : Tuple[dict, dict, int, int], key : str) -> \
        np.ndarray:
        """
        Takes a weighted average of two training set property dictionaries.

        Args:
        ----
            b (tuple) : Bundle of the following four items:
              p (dict) : "Previous" dictionary.
              n (dict) : "Next" dictionary.
              wp (int) : Weight for `p`.
              wn (int) : Weight for `n`.
            key (str) : 2nd string in the tuple keys.

        Returns:
        -------
            weighted_average (dict) : Dictionary is weighted average of `p` and `n`.
        """
        (p, n, wp, wn)   = b

        weighted_average = np.around((
            np.array(p[("Training set", key)]) * wp
            + np.array(n[("Training set", key)]) * wn
        ) / (wp + wn), decimals=3)

        return weighted_average


    def get_validation_likelihood(self, dataset : str) -> \
        Tuple[torch.Tensor, float]:
        """
        Computes validation NLL (e.g. the NLL for taking the "correct" action
        for a specific fragment/atom) for graphs in the validation and training
        sets (whichever is specified by the `dataloader`). The subsets are equal
        in size to the number of structures generated per batch (`n_samples`
        below). Note: do not use for generation set structures, as there is no
        "correct" action!

        Returns:
        -------
            likelihoods (torch.Tensor) : Contains all NLLs per action for generating
                                         a set of molecules via the "correct" set
                                         of actions.
            avg_final_likelihood (torch.Tensor) : Contains average final NLLs for
                                                  generating a set of molecules
                                                  via the "correct" set of actions.
        """
        if dataset == "validation":
            dataloader = self.valid_dataloader
        elif dataset == "training":
            dataloader = self.train_dataloader
        else:
            raise ValueError("Invalid dataset entered.")

        Softmax      = torch.nn.Softmax(dim=1)
        n_samples    = min(100000, constants.n_samples)  # n graphs to evaluate
        likelihoods  = torch.zeros(n_samples * (constants.max_n_nodes+5),
                                   device=constants.device)
        n_structures = torch.zeros(1, device=constants.device)

        # `batch` contains constants.n_samples subgraphs during validation
        for idx, batch in enumerate(dataloader):

            # for really large dataloaders (like that of the training set), the
            # line below ensures that the validation NLL is only calculated
            # until the number of structures analyzed is roughly equivalent to
            # the number of structures generated, purely for speed
            if idx * constants.batch_size > n_samples:
                break

            if constants.device == "cuda":
                batch = [b.cuda(non_blocking=True) for b in batch]
            nodes, edges, target_output = batch

            renormalized_target_output = (
                target_output/torch.sum(target_output, dim=1, keepdim=True)
            )

            # return the output and normalize
            normalized_output = Softmax(self.model(nodes, edges))

            # multiplication with `target_output` zeros out the "incorrect" actions
            correct_action_probabilities = torch.mul(renormalized_target_output,
                                                     normalized_output)
            likelihood = torch.sum(correct_action_probabilities, dim=1)
            # line below removes NaN values; ~ inverts a boolean tensor
            likelihood = -1 * torch.log(likelihood[~torch.isnan(likelihood)])
            start_idx  = idx * constants.batch_size
            end_idx    = idx * constants.batch_size + len(likelihood)
            likelihoods[start_idx:end_idx] = likelihood

            # in computing the number of structures, important to use
            # `target_output` and not `renormalized_target_output` (unnormalized
            # means the sum is number of subgraphs)
            n_structures += torch.sum(target_output[:, -1]).unsqueeze(dim=0)

        avg_final_likelihood = torch.sum(likelihoods, dim=0) / n_structures[0]

        return likelihoods, avg_final_likelihood


    def plot_molecular_properties(self, properties : dict,
                                  plot_filename : str) -> None:
        """
        Plots a 3 by 3 grid of the histograms in `properties` using separate
        colors for the training set and for each epoch.

        Args:
        ----
            properties (dict) : Contains properties of generated and training
              set molecules. Only plots histogram properties, not averages.
            plot_filename (str) : Full path/filename for saving output PNG.
        """
        # start the grid
        matplotlib.rc("figure", figsize=(8.0, 7.0))
        n_plots_y, n_plots_x = 3, 3
        fig, ax              = plt.subplots(n_plots_y, n_plots_x, sharey="all")
        fig.subplots_adjust(hspace=0.6, wspace=0.4)

        ax_nn = ax[0, 0]  # number of nodes
        ax_at = ax[0, 1]  # atom types
        ax_fc = ax[0, 2]  # formal charges
        ax_nh = ax[1, 0]  # num implicit Hs
        ax_ne = ax[1, 1]  # number of edges
        ax_bt = ax[1, 2]  # here, bond type == edge feature
        ax_ct = ax[2, 0]  # chirality

        # get the keys of the properties to plot
        keys_to_plot = list(set([key[0] for key in properties.keys()]))

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
             norm_chirality_hist) = util.normalize_evaluation_metrics(
                property_histograms=properties,
                epoch_key=epoch_key
            )

            # plot num nodes histogram
            ax_nn.plot(range(1, len(norm_n_nodes_hist) + 1), norm_n_nodes_hist,
                       color=c, label=epoch_key, linestyle=ls, marker=m)
            ax_nn.set(xlabel="Num nodes per graph")

            # plot atom type histogram
            ax_at.plot(range(1, len(norm_atom_type_hist) + 1), norm_atom_type_hist,
                       color=c, label=epoch_key, linestyle=ls, marker=m)
            xlabel_values = ", ".join(map(str, constants.atom_types))
            ax_at.set(xlabel=f"Atom type ({xlabel_values})")

            # plot formal charge histogram
            ax_fc.plot(range(constants.formal_charge[0], constants.formal_charge[-1] + 1),
                       norm_formal_charge_hist,
                       color=c, label=epoch_key, linestyle=ls, marker=m)
            xlabel_values = ", ".join(map(str, constants.formal_charge))
            ax_fc.set(xlabel=f"Formal charge ({xlabel_values})")

            # plot num H histogram
            ax_nh.plot(constants.imp_H, norm_numh_hist,
                       color=c, label=epoch_key, linestyle=ls, marker=m)
            xlabel_values = ", ".join(map(str, constants.imp_H))
            ax_nh.set(xlabel=f"Num implicit Hs ({xlabel_values})", ylabel="Fractional count")

            # plot n_edges histogram
            ax_ne.plot(range(1, len(norm_n_edges_hist) + 1), norm_n_edges_hist,
                       color=c, label=epoch_key, linestyle=ls, marker=m)
            ax_ne.set(xlabel="Num edges per node")

            # plot bond type/edge feature histogram
            ax_bt.plot(range(0, len(norm_edge_feature_hist)),
                       norm_edge_feature_hist,
                       color=c, label=epoch_key, linestyle=ls, marker=m)
            xlabel_values = ", ".join(map(str, constants.int_to_bondtype))
            ax_bt.set(xlabel=f"Bond type ({xlabel_values})")

            # plot chirality histogram
            ax_ct.plot(range(1, len(norm_chirality_hist) + 1), norm_chirality_hist,
                       color=c, label=epoch_key, linestyle=ls, marker=m)
            xlabel_values = ", ".join(map(str, constants.chirality))
            ax_ct.set(xlabel=f"Chirality ({xlabel_values})")

            # put the legend in the bottom right corner regardless
            ax_ct.legend(loc="upper right", prop={"size": 6})

        ax = util.turn_off_empty_axes(n_plots_y, n_plots_x, ax)

        fig.savefig(plot_filename)
        plt.close()

    def save_metrics(self, step, score, append=True) -> None:
        """
        Save the fine-tuning metrics, such as the model score, to a log file.

        Args:
        ----
            step (int)              : The fine-tuning step.
            score (float)           : Model score.
            append (bool, optional) : Indicates whether to append to an existing
                                      file, or create a new file. Defaults to True.
        """
        if not append:
            with open(constants.job_dir + "fine-tuning.log", "w") as output_file:
                output_file.write("Step, Score\n")
                output_file.write(f"Step {step}, {score:.8f}\n")
        else:
            with open(constants.job_dir + "fine-tuning.log", "a") as output_file:
                output_file.write(f"Step {step}, {score:.8f}\n")

        self.tb_writer.add_scalar("Evaluation/score", score, step)
