"""
`MolecularGraph.py` defines the parent MolecularGraph class and three sub-classes.
"""
# load general packages and functions
from collections import namedtuple
import itertools
import random
from copy import deepcopy
from typing import Union, Tuple
import numpy as np
import torch
import rdkit
from rdkit.Chem.rdmolfiles import MolToSmiles

# load GraphINVENT-specific functions
import util


class MolecularGraph:
    """
    Parent class for all molecular graphs.

    This class is then inherited by three subclasses:
      `PreprocessingGraph`, which is used when preprocessing training data, and
      `TrainingGraph`, which is used when training structures.
      `GenerationGraph`, which is used when generating structures.

    The reason for the two classes is that `np.ndarray`s are needed to save
    test/train/valid sets to HDF file format when preprocessing, but for
    training it is more efficient to use `torch.Tensor`s since these can be
    easily used on the GPU for training/generation.
    """
    def __init__(self, constants : namedtuple,
                 molecule : rdkit.Chem.Mol ,
                 node_features : Union[np.ndarray, torch.Tensor],
                 edge_features : Union[np.ndarray, torch.Tensor],
                 atom_feature_vector : torch.Tensor) -> None:
        """
        Args:
        ----
            constants (namedtuple)             : Contains job parameters as
                                                 well as global constants.
            molecule (rdkit.Chem.Mol)          : Input used for creating
                                                 `PreprocessingGraph`.
            atom_feature_vector (torch.Tensor) : Input used for creating
                                                 `TrainingGraph`.
        """
        self.constants = constants

        # placeholders (these are set in the respective sub-classes)
        self.molecule      = None
        self.node_features = None
        self.edge_features = None
        self.n_nodes       = None

    def get_graph_state(self) -> None:
        """
        This function is implemented in each subclass, since preprocessing
        graphs use `np.ndarray`s and training/generation graphs use
        `torch.Tensor`s.
        """
        raise NotImplementedError

    def get_n_edges(self) -> int:
        """
        Gets the number of edges in the `MolecularGraph`.
        """
        # divide by 2 to avoid double-counting edges
        n_edges = self.edge_features.sum() // 2
        return n_edges

    def get_molecule(self) -> rdkit.Chem.Mol:
        """
        Gets the molecule representation of the current `MolecularGraph`.
        """
        if self.molecule is False:
            pass
        else:
            self.molecule = self.graph_to_mol()

        return self.molecule

    def get_smiles(self) -> str:
        """
        Gets the SMILES representation of the current `MolecularGraph`.
        """
        try:
            smiles = MolToSmiles(mol=self.molecule, kekuleSmiles=False)
        except:
            # if molecule is invalid, set SMILES to `None`
            smiles = None
        return smiles

    def graph_to_mol(self) -> rdkit.Chem.Mol:
        """
        Generates the `rdkit.Chem.Mol` object corresponding to the graph.

        The function uses for a given graph:
          * `n_nodes`       : number of nodes in graph
          * `node_features` : node feature matrix, a |V|x|NF| matrix
          * `edge_features` : edge feature tensor, a |V|x|V|x|B| tensor

        Above, V is the set of all nodes in a graph, NF is the set of node
        features, and B is the set of available bond types.
        """
        # create empty editable `rdkit.Chem.Mol` object
        molecule    = rdkit.Chem.RWMol()

        # add atoms to `rdkit.Chem.Mol` and keep track of idx
        node_to_idx = {}

        for v in range(0, self.n_nodes):
            atom_to_add    = self.features_to_atom(node_idx=v)
            molecule_idx   = molecule.AddAtom(atom_to_add)
            node_to_idx[v] = molecule_idx

        # add bonds between adjacent atoms
        for bond_type in range(self.constants.n_edge_features):
            # `self.edge_features[:, :, bond_type]` is an adjacency matrix
            #  for that specific `bond_type`
            for bond_idx1, row in enumerate(
                self.edge_features[:self.n_nodes, :self.n_nodes, bond_type]
                ):
                # traverse only half adjacency matrix to not duplicate bonds
                for bond_idx2 in range(bond_idx1):
                    bond = row[bond_idx2]
                    if bond:  # if `bond_idx1` and `bond_idx2` are bonded
                        try:  # try adding the bond to `rdkit.Chem.Mol` object
                            molecule.AddBond(
                                node_to_idx[bond_idx1],
                                node_to_idx[bond_idx2],
                                self.constants.int_to_bondtype[bond_type]
                            )
                        except (TypeError, RuntimeError, AttributeError):
                            # errors occur if the above `AddBond()` action tries
                            # to add multiple bonds to a node pair (should not
                            # happen, but kept here as a safety)
                            raise ValueError("MolecularGraphError: Multiple "
                                             "edges connecting a single pair "
                                             "of nodes in graph.")

        try:  # convert from `rdkit.Chem.RWMol` to Mol object
            molecule.GetMol()
        except AttributeError:  # raised if molecules is `None`
            pass

        # if `ignore_H` flag is used, "sanitize" the structure to correct
        # the number of implicit hydrogens (otherwise, they will all stay at 0)
        if self.constants.ignore_H and molecule:
            try:
                rdkit.Chem.SanitizeMol(molecule)
            except ValueError:
                # raised if `molecule` is False, None, or too ugly to sanitize
                pass

        return molecule

    def features_to_atom(self, node_idx : int) -> rdkit.Chem.Atom:
        """
        Converts the atomic feature vector corresponding to the atom indicated
        by input `node_idx` into an `rdkit.Atom` object.

        The atomic feature vector describes a unique node on a graph using
        concatenated one-hot encoded vectors for the features of interest (e.g.
        atom type, formal charge), and is a row of `self.node_features`. Note
        that if `ignore_H` flag is used, will assign a placeholder of 0 to the
        number of implicit hydrogens in the atom (to be corrected for later via
        kekulization).

        Args:
        ----
            node_idx (int) : Index for a node feature vector (i.e. a row of
                             `self.node_features`).

        Returns:
        -------
            new_atom (rdkit.Atom) : Atom object.
        """
        # determine the nonzero indices of the feature vector
        feature_vector = self.node_features[node_idx]
        try:  # if `feature_vector` is a `torch.Tensor`
            nonzero_idc = torch.nonzero(feature_vector)
        except TypeError:  # if `feature_vector` is a `numpy.ndarray`
            nonzero_idc = np.nonzero(feature_vector)[0]

        # determine atom symbol
        atom_idx  = nonzero_idc[0]
        atom_type = self.constants.atom_types[atom_idx]
        new_atom  = rdkit.Chem.Atom(atom_type)

        # determine formal charge
        fc_idx        = nonzero_idc[1] - self.constants.n_atom_types
        formal_charge = self.constants.formal_charge[fc_idx]

        new_atom.SetFormalCharge(formal_charge)

        # determine number of implicit Hs
        if not self.constants.use_explicit_H and not self.constants.ignore_H:
            total_num_h_idx = (
                nonzero_idc[2]
                - self.constants.n_atom_types
                - self.constants.n_formal_charge
            )
            total_num_h = self.constants.imp_H[total_num_h_idx]

            new_atom.SetUnsignedProp("_TotalNumHs", total_num_h)

        elif self.constants.ignore_H:
            # Hs will be set with structure is sanitized later
            pass

        # determine chirality
        if self.constants.use_chirality:
            cip_code_idx = (
                    nonzero_idc[-1]
                    - self.constants.n_atom_types
                    - self.constants.n_formal_charge
                    - bool(not self.constants.use_explicit_H and
                           not self.constants.ignore_H)
                    * self.constants.n_imp_H
            )
            cip_code = self.constants.chirality[cip_code_idx]
            new_atom.SetProp("_CIPCode", cip_code)

        return new_atom

    def mol_to_graph(self, molecule : rdkit.Chem.Mol) -> None:
        """
        Generates the graph representation (`self.node_features` and
        `self.edge_features`) when creating a new `PreprocessingGraph`.
        """
        n_atoms = self.n_nodes
        atoms   = map(molecule.GetAtomWithIdx, range(n_atoms))

        # build the node features matrix using a Numpy array
        node_features = np.array(list(map(self.atom_features, atoms)),
                                 dtype=np.int32)

        # build the edge features tensor using a Numpy array
        edge_features = np.zeros(
            [n_atoms, n_atoms, self.constants.n_edge_features],
            dtype=np.int32
        )
        for bond in molecule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = self.constants.bondtype_to_int[bond.GetBondType()]
            edge_features[i, j, bond_type] = 1
            edge_features[j, i, bond_type] = 1

        # define the number of nodes
        self.n_nodes = n_atoms

        self.node_features = node_features  # not padded!
        self.edge_features = edge_features  # not padded!


class PreprocessingGraph(MolecularGraph):
    """
    Class for molecular graphs to be used during the data preprocessing phase.
    Uses `np.ndarray`s for the graph attributes, so they can be stored as HDF5
    datasets. These are never loaded onto the GPU.
    """
    def __init__(self, constants : namedtuple,
                 molecule : rdkit.Chem.Mol) -> None:
        super().__init__(constants, molecule=False, node_features=False,
                         edge_features=False, atom_feature_vector=False)

        # define values previously set to `None` or undefined
        self.node_ordering = None  # to be defined in `self.node_remap()`

        if self.constants.use_explicit_H and not self.constants.ignore_H:
            molecule = rdkit.Chem.AddHs(molecule)

        self.n_nodes = molecule.GetNumAtoms()

        # get the graph attributes from the `rdkit.Chem.Mol()` object
        self.mol_to_graph(molecule=molecule)

        # remap the nodes using either a canonical or random node ordering
        self.node_remap(molecule=molecule)

        # pad up to size of largest graph in dataset (`self.constants.max_n_nodes`)
        self.pad_graph_representation()

    def atom_features(self, atom : rdkit.Chem.Atom) -> np.ndarray:
        """
        Generates the feature vector for a given node on a molecular graph. Uses
        the following descriptors to recreate the molecular graph: atom type,
        formal charge, and, if specified, number of implicit Hs and chirality.
        The first two descriptors are the bare minimum needed.

        Args:
        ----
            atom (rdkit.Chem.Atom) : Atom in molecule for which to get feature
                                     vector.

        Returns:
        -------
            feature_vector (numpy.ndarray) : Corresponding feature vector.
        """
        feature_vector_generator = itertools.chain(
            util.one_of_k_encoding(atom.GetSymbol(), self.constants.atom_types),
            util.one_of_k_encoding(atom.GetFormalCharge(),
                                   self.constants.formal_charge)
        )
        if not self.constants.use_explicit_H and not self.constants.ignore_H:
            feature_vector_generator = itertools.chain(
                feature_vector_generator,
                util.one_of_k_encoding(atom.GetTotalNumHs(),
                                       self.constants.imp_H)
            )
        if self.constants.use_chirality:
            try:
                chiral_state = atom.GetProp("_CIPCode")
            except KeyError:
                chiral_state = self.constants.chirality[0]  # "None"

            feature_vector_generator = itertools.chain(
                feature_vector_generator,
                util.one_of_k_encoding(chiral_state, self.constants.chirality)
            )

        feature_vector = np.fromiter(feature_vector_generator, int)

        return feature_vector

    def breadth_first_search(self, node_ranking : list,
                             node_init : int=0) -> list:
        """
        Starting from the specified `node_init` in the graph, uses a breadth-
        first search (BFS) algorithm to find all adjacent nodes, returning an
        ordered list of these nodes. Prioritizes the nodes based on the input
        `node_ranking`. The function uses the edge feature tensor to find
        adjacent nodes.

        Args:
        ----
            node_ranking (list) : Contains the ranking of all the nodes in the
                                  graph (e.g. the canonical RDKit node ranking,
                                  or a random ranking).
            node_init (int)     : Index of node to start the BFS from. Default 0.

        Returns:
        -------
            nodes_visited (list) : BFS ordering for nodes in the molecular graph.
        """
        nodes_visited      = [node_init]
        last_nodes_visited = [node_init]

        # loop until all nodes have been visited
        while len(nodes_visited) < self.n_nodes:
            neighboring_nodes = []

            for node in last_nodes_visited:
                neighbor_nodes = []
                for bond_type in range(self.constants.n_edge_features):
                    neighbor_nodes.extend(list(
                        np.nonzero(self.edge_features[node, :, bond_type])[0]
                    ))
                new_neighbor_nodes = list(
                    set(neighbor_nodes) - (set(neighbor_nodes) & set(nodes_visited))
                )
                node_importance    = [node_ranking[neighbor_node] for
                                      neighbor_node in new_neighbor_nodes]

                # check all neighboring nodes and sort in order of importance
                while sum(node_importance) != -len(node_importance):
                    next_node = node_importance.index(max(node_importance))
                    neighboring_nodes.append(new_neighbor_nodes[next_node])
                    node_importance[next_node] = -1

            # append the new, sorted neighboring nodes to list of visited nodes
            nodes_visited.extend(set(neighboring_nodes))

            # update the list of most recently visited nodes
            last_nodes_visited = set(neighboring_nodes)

        return nodes_visited

    def depth_first_search(self, node_ranking : list, node_init : int=0) -> list:
        """
        Starting from the specified `node_init` in the graph, uses a depth-first
        search (DFS) algorithm to find the longest branch nodes, returning an
        ordered list of the nodes traversed. Prioritizes the nodes based on the
        input `node_ranking`. The function uses the edge feature tensor to find
        adjacent nodes.

        Args:
        ----
            node_ranking (list) : Contains the ranking of all the nodes in the
                                  graph (e.g. the canonical RDKit node ranking,
                                  or a random ranking).
            node_init (int)     : Index of node to start the DFS from. Default 0.

        Returns:
        -------
            nodes_visited (list) : DFS ordering for nodes in the molecular graph.
        """
        nodes_visited     = [node_init]
        last_node_visited = node_init

        # loop until all nodes have been visited
        while len(nodes_visited) < self.n_nodes:

            neighbor_nodes = []
            for bond_type in range(self.constants.n_edge_features):
                neighbor_nodes.extend(list(
                    np.nonzero(self.edge_features[last_node_visited, :, bond_type])[0]
                ))
            new_neighbor_nodes = list(
                set(neighbor_nodes) - (set(neighbor_nodes) & set(nodes_visited))
            )

            if not new_neighbor_nodes:  # list is empty
                # backtrack if there are no "new" neighbor nodes i.e. reached end of branch
                current_node_idx  = nodes_visited.index(last_node_visited)
                last_node_visited = nodes_visited[current_node_idx - 1]
                continue

            node_importance = [node_ranking[neighbor_node] for
                               neighbor_node in new_neighbor_nodes]

            # get the most important of the neighboring nodes
            most_important_neighbor_node = node_importance.index(max(node_importance))

            # append the new, sorted neighboring nodes to list of visited nodes
            nodes_visited.extend([new_neighbor_nodes[most_important_neighbor_node]])

            # update the most recently visited node
            last_node_visited = new_neighbor_nodes[most_important_neighbor_node]

        return nodes_visited

    def node_remap(self, molecule : rdkit.Chem.Mol) -> None:
        """
        Remaps nodes in `rdkit.Chem.Mol` object (`molecule`) either randomly, or
        using RDKit's canonical node ordering. This depends on if `use_canon` is
        specified or not.
        """
        if not self.constants.use_canon:
            # get a *random* node ranking
            atom_ranking = list(range(self.n_nodes))
            random.shuffle(atom_ranking)
        else:
            # get RDKit canonical ranking
            atom_ranking = list(
                rdkit.Chem.CanonicalRankAtoms(molecule, breakTies=True)
            )

        # using a random node as a starting point, get a new node ranking that
        # does not leave isolated fragments in graph traversal
        if self.constants.decoding_route == "bfs":
            self.node_ordering = self.breadth_first_search(node_ranking=atom_ranking,
                                                           node_init=atom_ranking[0])
        elif self.constants.decoding_route == "dfs":
            self.node_ordering = self.depth_first_search(node_ranking=atom_ranking,
                                                         node_init=atom_ranking[0])

        # reorder all nodes according to new node ranking
        self.reorder_nodes()

    def get_decoding_APD(self) -> np.ndarray:
        """
        For a given subgraph along a decoding route for a `PreprocessingGraph`,
        computes the target decoding APD that would take you to the next
        subgraph (adding one edge/node). Used when generating the training data.

        Returns:
        -------
            The graph decoding APD, comprised of the following probability values:

            f_add (numpy.ndarray)  : Add APD. Size Mx|A|x|F|x|H|x|B| tensor whose
                                     elements are the probabilities of adding a new
                                     atom of type a with formal charge f and implicit
                                     Hs h to existing atom v with a new bond of type
                                     b. If `use_chirality`==True, it is instead a
                                     size Mx|A|x|F|x|H|x|C|x|B| tensor, whose elements
                                     are the probabilities of adding such an atom
                                     with chiral state c.
            f_conn (numpy.ndarray) : Connect APD. Size |V|x|B| matrix, whose
                                     elements are the probability of connecting
                                     the last appended atom with existing atom v
                                     using a new bond of type b.
            f_term (int)           : Terminate APD. Scalar indicating probability
                                     of terminating the graph.

            M is the maximum number of nodes in a graph in any set (train, test,
            val), A is the set of atom types, F is the set of formal charges,
            H is the set of implicit Hs, C is the set of chiral states, and B
            is the set of bond types.
        """
        # determine the indices of the atom descriptors (e.g. atom type)
        last_node_idx  = self.n_nodes - 1  # zero-indexing
        fv_nonzero_idc = self.get_nonzero_feature_indices(node_idx=last_node_idx)

        # initialize action probability distribution (APD)
        f_add  = np.zeros(self.constants.dim_f_add, dtype=np.int32)
        f_conn = np.zeros(self.constants.dim_f_conn, dtype=np.int32)

        # determine which nodes are bonded
        bonded_nodes = []
        for bond_type in range(self.constants.n_edge_features):
            bonded_nodes.extend(list(
                np.nonzero(self.edge_features[:, last_node_idx, bond_type])[0]
            ))

        if bonded_nodes:
            degree            = len(bonded_nodes)
            v_idx             = bonded_nodes[-1]  # idx of node to form bond with
            bond_type_forming = int(
                np.nonzero(self.edge_features[v_idx, last_node_idx, :])[0]
            )

            if degree > 1:
                # if multiple bonds to one node first add bonds one by one
                # (modify `f_conn`)
                f_conn[v_idx, bond_type_forming] = 1
            else:
                # if only bound to one node, bond and node addition is one move
                # (modify `f_add`)
                f_add[tuple([v_idx] + fv_nonzero_idc + [bond_type_forming])] = 1
        else:
            # if it is the last node in the graph, node addition occurs in one
            # move (modify `f_add`); uses a dummy edge to "connect" to node 0
            f_add[tuple([0] + fv_nonzero_idc + [0])] = 1

        # concatenate `f_add`, `f_conn`, and `f_term` (`f_term`==0)
        apd = np.concatenate((f_add.ravel(), f_conn.ravel(), np.array([0])))
        return apd

    def get_final_decoding_APD(self) -> np.ndarray:
        """
        For a given subgraph along a decoding route for a `PreprocessingGraph`,
        computes the target decoding APD that would indicate termination. Used
        when generating the training data.

        Returns:
        -------
            The graph decoding APD, comprised of the following probability
            values (see `get_decoding_APD()` docstring above):

            f_add (numpy.ndarray)  : Add APD.
            f_conn (numpy.ndarray) : Connect APD.
            f_term (int)           : Terminate APD. Scalar (1, since terminating)
                                     indicating the probability of terminating
                                     the graph generation.
        """
        # initialize action probability distribution (APD)
        f_add  = np.zeros(self.constants.dim_f_add, dtype=np.int32)
        f_conn = np.zeros(self.constants.dim_f_conn, dtype=np.int32)

        # concatenate `f_add`, `f_conn`, and `f_term` (`f_term`==0)
        apd = np.concatenate((f_add.ravel(), f_conn.ravel(), np.array([1])))
        return apd

    def get_graph_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the graph representation of the current `PreprocessingGraph`.
        """
        return self.node_features, self.edge_features

    def get_nonzero_feature_indices(self, node_idx : int) -> list:
        """
        Gets indices of the nonzero values in a one-hot encoded atomic feature
        vector (for converting a feature vector into an `rdkit.Chem.Atom`
        object).

        Args:
        ----
            node_idx (int) : Index for a specific node in the `PreprocessingGraph`.

        Returns:
        -------
            segment_idc (list) : Contains the nonzero indices of the atom type,
                                 formal charge, number of implicit Hs, and chirality
                                 that describe a specific node in a feature vector.
                                 The indices are "corrected" for each one-hot encoded
                                 segment.
        """
        fv_idc = util.get_feature_vector_indices()
        idc    = np.nonzero(self.node_features[node_idx])[0]

        # correct for the concatenation of the different segments of each node
        # feature vector
        segment_idc = [idc[0]]
        for idx, value in enumerate(idc[1:]):
            segment_idc.append(value - fv_idc[idx])

        return segment_idc

    def reorder_nodes(self) -> None:
        """
        Remaps the numerical ordering of nodes in the graph as specified by the
        `self.node_ordering`. Modifies the `PreprocessingGraph` in place.
        """
        # first remap the node features matrix
        node_features_remapped = np.array(
            [self.node_features[node] for node in self.node_ordering],
            dtype=np.int32
        )

        # then remap the edge features tensor
        edge_features_rows_done = np.array(
            [self.edge_features[node, :, :] for node in self.node_ordering],
            dtype=np.int32
        )
        edge_features_remapped = np.array(
            [edge_features_rows_done[:, node, :] for node in self.node_ordering],
            dtype=np.int32
        )

        self.node_features = node_features_remapped
        self.edge_features = edge_features_remapped

    def pad_graph_representation(self) -> None:
        """
        Pads arrays to size corresponding to largest graph in training, testing,
        and validation datasets.
        """
        # initialize the padded graph representation arrays
        node_features_padded = np.zeros((self.constants.max_n_nodes,
                                         self.constants.n_node_features))
        edge_features_padded = np.zeros((self.constants.max_n_nodes,
                                         self.constants.max_n_nodes,
                                         self.constants.n_edge_features))

        # pad up to size of largest graph
        node_features_padded[:self.n_nodes, :]                = self.node_features
        edge_features_padded[:self.n_nodes, :self.n_nodes, :] = self.edge_features

        self.node_features = node_features_padded
        self.edge_features = edge_features_padded

    def truncate_graph(self) -> None:
        """
        Truncates a molecule by editing its molecular graph (`self.node_features`
        and `self.edge_features`) in place. By default deletes the last node.

        If the last atom is bound to multiple atoms on the graph (i.e. a ring
        atom), then only deletes the least "important" bond, as determined from
        the breadth-first ordering. This is so as to allow the APD to be broken
        up into multiple steps (add, connect, terminate).
        """
        last_atom_idx = self.n_nodes - 1

        if self.n_nodes == 1:
            # remove the last atom
            self.node_features[last_atom_idx, :] = 0
            self.n_nodes                        -= 1
        else:
            # determine how many bonds on the least important atom
            bond_idc = []
            for bond_type in range(self.constants.n_edge_features):
                bond_idc.extend(
                    list(
                        np.nonzero(self.edge_features[:, last_atom_idx, bond_type])[0]
                    )
                )

            degree = len(bond_idc)

            if degree == 1:
                # delete atom from node features
                self.node_features[last_atom_idx, :] = 0
                self.n_nodes -= 1
            else:  # if degree > 1
                # if the last atom is bound to multiple atoms, only delete the
                # least important bond, but leave the atom and remaining bonds
                bond_idc = bond_idc[-1]  # mark bond for deletion (below)

            # delete bond from row feature tensor (first row, then column)
            self.edge_features[bond_idc, last_atom_idx, :] = 0
            self.edge_features[last_atom_idx, bond_idc, :] = 0

    def get_decoding_route_length(self) -> int:
        """
        Returns the number of subgraphs in the graph's decoding route, which is
        how many subgraphs would be formed in the process of deleting the last
        atom/bond in the molecule stepwise until only a single atom is left.
        Note that this is simply the number of edges plus two, since each action
        adds an edge, plus we have the initial and final actions.

        Returns:
        -------
            n_decoding_graphs (int) : Number of subgraphs in the input graph's
                                      decoding route.
        """
        return int(self.get_n_edges() + 2)

    def get_decoding_route_state(self, subgraph_idx : int) -> \
                                 Tuple[list, np.ndarray]:
        """
        Starting from the specified graph, returns the state (subgraph and
        decoding APD) indicated by `subgraph_idx` along the decoding route.

        Args:
        ----
            subgraph_idx (int) : Index of subgraph along decoding route.

        Returns:
        -------
            decoding_graph (list)      : Graph representation, structured as [X, E].
            decoding_APDs (np.ndarray) : Contains the decoding APD, structured as
                                         a concatenation of flattened (f_add, f_conn,
                                         f_term).
        """
        molecular_graph = deepcopy(self)

        if subgraph_idx != 0:
            # find which subgraph is indicated by the index by progressively
            # truncating the input molecular graph
            for _ in range(1, subgraph_idx):
                molecular_graph.truncate_graph()

            # get the APD before the last truncation (since APD says how to get
            # to the next graph, need to truncate once more after obtaining APD)
            decoding_APD = molecular_graph.get_decoding_APD()
            molecular_graph.truncate_graph()
            X, E         = molecular_graph.get_graph_state()

        elif subgraph_idx == 0:
            # return the first subgraph
            decoding_APD = molecular_graph.get_final_decoding_APD()
            X, E         = molecular_graph.get_graph_state()

        else:
            raise ValueError("`subgraph_idx` not a valid value.")

        decoding_graph = [X, E]

        return decoding_graph, decoding_APD


class TrainingGraph(MolecularGraph):
    """
    Class for molecular graphs to be used during model training. Uses
    `torch.Tensor`s for the graph attributes, so they can be conveniently used
    on the GPU.
    """
    def __init__(self, constants : namedtuple,
                 atom_feature_vector : torch.Tensor) -> None:
        super().__init__(constants, molecule=False, node_features=False,
                         edge_features=False, atom_feature_vector=False)

        # define values previously set to `None` or undefined
        self.n_nodes = int(bool(1 in atom_feature_vector))

        # define graph attributes
        self.node_features = atom_feature_vector.unsqueeze(dim=0)

        self.edge_features = torch.Tensor([[[0] * self.constants.n_edge_features]],
                                          device=self.constants.device)

        # initialize the padded graph representation arrays
        node_features_padded = torch.zeros((self.constants.max_n_nodes,
                                            self.constants.n_node_features),
                                           device=self.constants.device)
        edge_features_padded = torch.zeros((self.constants.max_n_nodes,
                                            self.constants.max_n_nodes,
                                            self.constants.n_edge_features),
                                           device=self.constants.device)

        # pad up to size of largest graph
        node_features_padded[:self.n_nodes, :]                = self.node_features
        edge_features_padded[:self.n_nodes, :self.n_nodes, :] = self.edge_features

        self.node_features = node_features_padded
        self.edge_features = edge_features_padded

    def get_graph_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the graph representation of the current `TrainingGraph`. Note
        that it will pad up all the graph attributes to those of the size of the
        largest graph.
        """
        # convert to torch.Tensors
        node_features_tensor = torch.Tensor(self.node_features)
        adjacency_tensor     = torch.Tensor(self.edge_features)
        return node_features_tensor, adjacency_tensor


class GenerationGraph(MolecularGraph):
    """
    Class for molecular graphs to be used during graph generation. Uses
    `torch.Tensor`s for the graph attributes, so they can be conveniently used
    on the GPU.
    """
    def __init__(self, constants : namedtuple,
                 molecule : rdkit.Chem.Mol,
                 node_features : torch.Tensor,
                 edge_features : torch.Tensor) -> None:
        super().__init__(constants, molecule=False, node_features=False,
                         edge_features=False, atom_feature_vector=False)

        try:
            self.n_nodes = molecule.GetNumAtoms()
        except AttributeError:
            self.n_nodes = 0

        self.molecule      = molecule
        self.node_features = node_features
        self.edge_features = edge_features

    def get_molecule(self) -> rdkit.Chem.Mol:
        """
        Returns the molecule representation of the current `MolecularGraph`.
        """
        return self.molecule
