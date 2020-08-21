# load general packages and functions
import numpy as np
import torch
import rdkit
from tqdm import tqdm
import time

# load program-specific functions
from parameters.constants import constants as C
from MolecularGraph import GenerationGraph

# defines how to build molecular graphs using the following actions:
# * "add" a node to graph
# * "connect" existing nodes in graph
# * "terminate" graph



def allocate_graph_tensors(n_total, batch_size):
    """ Allocates tensors for the node features, edge features, NLLs, and
    termination status for all graphs to be generated (**not only one batch**).
    These then get filled in during the graph generation process.

    Args:
      n_total (int) : Total number of graphs to generate (**not only batch**).
      batch_size (int) : Batch size.

    Returns:
      nodes (torch.Tensor) : Node features tensor for all graphs.
      edges (torch.Tensor) : Edge features tensor for all graphs.
      n_nodes (torch.Tensor) : Number of nodes per graph in all graphs.
      nlls (torch.Tensor) : Sampled NLL per action for all graphs.
      generated_nlls (torch.Tensor) : Sampled NLLs per action for all finished
        graphs.
      properly_terminated (torch.Tensor) : Indicates if graph was properly
        terminated or not using a 0 or 1 at the corresponding index.
    """
    # define tensor shapes
    node_shape = (batch_size, *C.dim_nodes)
    edge_shape = (batch_size, *C.dim_edges)
    nlls_shape = (C.max_n_nodes + 7)*int(n_total / batch_size)  # the + 7 is buffer

    n_allocate = n_total + batch_size

    # create the placeholder tensors
    nodes = torch.zeros((n_allocate, *node_shape[1:]), dtype=torch.float32, device="cuda")
    edges = torch.zeros((n_allocate, *edge_shape[1:]), dtype=torch.float32, device="cuda")
    n_nodes = torch.zeros(n_allocate, dtype=torch.int8, device="cuda")
    nlls = torch.zeros((batch_size + 1, nlls_shape), device="cuda")
    generated_nlls = torch.zeros((n_allocate, nlls_shape), device="cuda")
    properly_terminated = torch.zeros(n_allocate, device="cuda")

    return nodes, edges, n_nodes, nlls, generated_nlls, properly_terminated


def apply_actions(add, conn, nodes, edges, n_nodes, generation_round, nlls, nlls_sampled):
    """ Applies the batch of sampled actions (specified by `add` and `conn`) to
    the batch of graphs under construction. Also adds the NLLs for the newly
    sampled actions (`nlls_sampled`) to the running list of NLLs (`nlls`).

    Args:
      add (torch.Tensor) : Indices indicating which "add" actions were
        sampled for a batch of graphs.
      conn (torch.Tensor) : Indices indicating which "connect" actions were
        sampled for a batch of graphs.
      nodes (torch.Tensor) : Node features tensor (batch).
      edges (torch.Tensor) : Edge features tensor (batch).
      n_nodes (torch.Tensor) : Number of nodes per graph in `nodes` and `edges`
        (batch.)
      generation_round (int) : Indicates the current generation round (running
        count).
      nlls (torch.Tensor) : Sampled NLL per action for graphs in `nodes` and
        `edges` (batch).
      nlls_sampled (torch.Tensor) : NLL per action sampled for the most recent
        set of actions.

    Returns:
      nodes (torch.Tensor) : Updated node features tensor (batch).
      edges (torch.Tensor) : Updated edge features tensor (batch).
      n_nodes (torch.Tensor) : Updated number of nodes per graph in `nodes` and
        `edges` (batch.)
      nlls (torch.Tensor) : Updated ampled NLL per action for graphs in `nodes`
        and `edges` (batch).
    """
    # first applies the "add" action to all graphs in batch (note: does nothing
    # if a graph did not sample "add")
    nodes, edges, n_nodes, nlls = add_nodes(add,
                                            nodes,
                                            edges,
                                            n_nodes,
                                            generation_round,
                                            nlls,
                                            nlls_sampled)
    # then applies the "connect" action to all graphs in batch (note: does
    # nothing if a graph did not sample "connect")
    edges, nlls = conn_nodes(conn,
                             edges,
                             n_nodes,
                             generation_round,
                             nlls,
                             nlls_sampled)

    return nodes, edges, n_nodes, nlls


def add_nodes(add, nodes, edges, n_nodes, generation_round, nlls, nlls_sampled):
    """ Adds new nodes to graphs which sampled the "add" action.
    """
    # get the action indices
    batch, to, t, ch, b, fr = add  # t = type, ch = charge

    n_node_features = [C.n_atom_types, C.n_formal_charge]

    # add the new nodes to the node features tensors
    nodes[batch, fr, t] = 1
    nodes[batch, fr, ch + n_node_features[0]] = 1


    # mask dummy edges (self-loops) introduced from adding node to empty graph
    batch_masked = batch[torch.nonzero(n_nodes[batch] != 0)]
    to_masked = to[torch.nonzero(n_nodes[batch] != 0)]
    fr_masked = fr[torch.nonzero(n_nodes[batch] != 0)]
    b_masked = b[torch.nonzero(n_nodes[batch] != 0)]

    # connect newly added nodes to the graphs
    edges[batch_masked, to_masked, fr_masked, b_masked] = 1
    edges[batch_masked, fr_masked, to_masked, b_masked] = 1

    # keep track of the newly added node
    n_nodes[batch] += 1

    # include the NLLs for the add actions for this generation round
    nlls[batch, generation_round] = nlls_sampled[batch]

    return nodes, edges, n_nodes, nlls


def conn_nodes(conn, edges, n_nodes, generation_round, nlls, nlls_sampled):
    """ Connects nodes in graphs which sampled the "connect" action.
    """
    # get the action indices
    batch, to, b, fr = conn

    # apply the connect actions
    edges[batch, fr, to, b] = 1
    edges[batch, to, fr, b] = 1

    # include the NLLs for the connect actions for this generation round
    nlls[batch, generation_round] = nlls_sampled[batch]

    return edges, nlls


def copy_terminated_graphs(terminate_idc, n_graphs_generated, nodes, edges, n_nodes,
                           generated_nodes, generated_edges, generated_n_nodes,
                           generation_round, nlls, nlls_sampled, generated_nlls):
    """ Copies terminated graphs (either because "terminate" action sampled, or
    invalid action sampled) to `generated_nodes` and `generated_edges` before
    they are removed from the running batch of graphs being generated.

    Args:
      terminate_idc (torch.Tensor) : Indices corresponding to graphs that will
        terminate this round.
      n_graphs_generated (int) : Number of graphs generated thus far (not
        including those about to be copied).
      nodes (torch.Tensor) : Node features tensors for a batch of graphs.
      edges (torch.Tensor) : Edge features tensors for a batch of graphs.
      n_nodes (torch.Tensor) : Number of nodes in each graph for a batch of
        graphs.
      generated_nodes (torch.Tensor) : Node features tensors for completed
        graphs thus far (not including those about to be copied).
      generated_edges (torch.Tensor) : Edge features tensors for completed
        graphs thus far (not including those about to be copied).
      generated_n_nodes (torch.Tensor) : Number of nodes in each completed
        graph thus far (not including those about to be copied).
      generation_round (int) : Indicates the current generation round (running
        count).
      nlls (torch.Tensor) : NLLs per action for each graph in a batch of graphs.
      nlls_sampled (torch.Tensor) : NLLs for the newest sampled action for each
        graph in a batch of graphs (not yet included in `nlls`).
      generated_nlls (torch.Tensor) : Sampled NLLs per action for completed
        graphs thus far (not including those about to be copied).

    Returns:
      n_graphs_generated (int) : Number of graphs generated thus far.
      generated_nodes (torch.Tensor) : Node features tensors for completed
        graphs thus far.
      generated_edges (torch.Tensor) : Edge features tensors for completed
        graphs thus far.
      generated_n_nodes (torch.Tensor) : Number of nodes in each completed
        graph thus far.
      generated_nlls (torch.Tensor) : Sampled NLLs per action for completed
        graphs thus far.
    """
    # number of graphs to be terminated
    nlls[terminate_idc, generation_round] = nlls_sampled[terminate_idc]

    # number of graphs to be terminated
    n = len(terminate_idc)

    # copy the new graphs to the finished tensors
    nodes_local = nodes[terminate_idc]
    edges_local = edges[terminate_idc]
    n_nodes_local = n_nodes[terminate_idc]
    nlls_local = nlls[terminate_idc]

    generated_nodes[n_graphs_generated : n_graphs_generated + n] = nodes_local
    generated_edges[n_graphs_generated : n_graphs_generated + n] = edges_local
    generated_n_nodes[n_graphs_generated : n_graphs_generated + n] = n_nodes_local
    generated_nlls[n_graphs_generated : n_graphs_generated + n] = nlls_local

    n_graphs_generated += n

    return (
        n_graphs_generated,
        generated_nodes,
        generated_edges,
        generated_n_nodes,
        generated_nlls
    )


def initialize_graph_batch(batch_size):
    """ Initialize a batch of empty graphs to begin the generation process.

    Args:
      batch_size (int) : Batch size.

    Returns:
      nodes (torch.Tensor) : Empty node features tensor (batch).
      edges (torch.Tensor) : Empty edge features tensor (batch).
      n_nodes (torch.Tensor) : Number of nodes per graph in `nodes` and `edges`
        (batch), currently all 0.
    """
    # define tensor shapes
    node_shape = ([batch_size + 1] + C.dim_nodes)
    edge_shape = ([batch_size + 1] + C.dim_edges)

    # initialize tensors
    nodes = torch.zeros(node_shape, dtype=torch.float32, device="cuda")
    edges = torch.zeros(edge_shape, dtype=torch.float32, device="cuda")
    n_nodes = torch.zeros(batch_size + 1, dtype=torch.int64, device="cuda")

    # add a dummy non-empty graph at top, since models cannot receive as input
    # purely empty graphs
    nodes[0] = torch.ones(([1] + C.dim_nodes), device="cuda")
    edges[0, 0, 0, 0] = 1
    n_nodes[0] = 1

    return nodes, edges, n_nodes


def reset_graphs(n_samples, idx, nodes, edges, n_nodes, nlls, batch_size):
    """Resets the `nodes` and `edges` tensors by reseting graphs which sampled
    invalid actions (indicated by `idx`).

    Args:
      n_samples (int) : Number of graphs to generate in one batch.
      idx (int) : Indices corresponding to graphs to reset.
      nodes (torch.Tensor) : Node features tensor (batch).
      edges (torch.Tensor) : Edge features tensor (batch).
      n_nodes (torch.Tensor) : Number of nodes per graph in `nodes` and `edges`
        (batch).
      nlls (torch.Tensor) : Sampled NLL per action for graphs in `nodes` and
        `edges` (batch).
      batch_size (int) : Batch size.

    Returns:
      nodes_reset (torch.Tensor) : Reset node features tensor (batch).
      edges_reset (torch.Tensor) : Reset edge features tensor (batch).
      n_nodes_reset (torch.Tensor) : Reset number of nodes per graph in `nodes`
        and `edges` (batch).
      nlls_reset (torch.Tensor) : Reset sampled NLL per action for graphs in
        `nodes` and `edges` (batch).
    """
    # define constants
    n_total = n_samples
    n_batches = int(n_total / batch_size)
    node_shape = ([batch_size + 1] + C.dim_nodes)
    edge_shape = ([batch_size + 1] + C.dim_edges)
    nlls_shape = ([batch_size + 1] + [(C.max_n_nodes + 7)*n_batches])  # the + 7 is buffer for connect actions

    # initialize placeholders
    nodes_reset = torch.zeros(node_shape, dtype=torch.float32, device="cuda")
    edges_reset = torch.zeros(edge_shape, dtype=torch.float32, device="cuda")
    n_nodes_reset = torch.zeros(batch_size + 1, dtype=torch.int64, device="cuda")
    nlls_reset = torch.zeros(nlls_shape, dtype=torch.float32, device="cuda")

    # reset the "bad" graphs with zero tensors
    if len(idx) > 0:
        nodes[idx] = torch.zeros(
            (len(idx), *node_shape[1:]), dtype=torch.float32, device="cuda"
        )
        edges[idx] = torch.zeros(
            (len(idx), *edge_shape[1:]), dtype=torch.float32, device="cuda"
        )
        n_nodes[idx] = torch.zeros(
            len(idx), dtype=torch.int64, device="cuda"
        )
        nlls[idx] = torch.zeros(
            (len(idx), *nlls_shape[1:]), dtype=torch.float32, device="cuda"
        )

    # fill in the placeholder tensors with the respective tensors
    nodes_reset[1:] = nodes
    edges_reset[1:] = edges
    n_nodes_reset[1:] = n_nodes
    nlls_reset[1:] = nlls

    # add a dummy non-empty graph
    nodes_reset[0] = torch.ones(([1] + C.dim_nodes), device="cuda")
    edges_reset[0, 0, 0, 0] = 1
    n_nodes_reset[0] = 1

    return nodes_reset, edges_reset, n_nodes_reset, nlls_reset


def get_actions(apds, edges, n_nodes, batch_size):
    """ Samples the input batch of APDs and separates the action indices.

    Args:
        apds (torch.Tensor) : APDs for a batch of graphs.
        edges (torch.Tensor) : Edge features tensor for a batch of graphs.
        n_nodes (torch.Tensor) : Number of nodes corresponding to graphs in
          `edges`.
        batch_size (int) : Batch size.

    Returns:
      f_add_idc (torch.Tensor) : Indices corresponding to "add" action.
      f_conn_idc (torch.Tensor) : Indices corresponding to "connect" action.
      f_term_idc (torch.Tensor) : Indices corresponding to "terminate"
        action.
      invalid_idc (torch.Tensor) : Indices corresponding graphs which
        sampled an invalid action.
      nlls (torch.Tensor) : NLLs per action corresponding to graphs in batch.
    """
    # sample the APD for all graphs in the batch for action indices
    f_add_idc, f_conn_idc, f_term_idc, nlls = sample_apd(apds, batch_size)

    # get indices for the "add" action
    f_add_from = n_nodes[f_add_idc[0]]
    f_add_idc = (*f_add_idc, f_add_from)

    # get indices for the "connect" action
    f_conn_from = n_nodes[f_conn_idc[0]] - 1
    f_conn_idc = (*f_conn_idc, f_conn_from)

    # get indices for the invalid add and connect actions
    invalid_idc, max_node_idc = get_invalid_actions(f_add_idc,
                                                    f_conn_idc,
                                                    edges,
                                                    n_nodes)

    # change "connect to" index for graphs trying to add more than max num nodes
    f_add_idc[-1][max_node_idc] = 0

    return f_add_idc, f_conn_idc, f_term_idc, invalid_idc, nlls


def get_invalid_actions(f_add_idc, f_conn_idc, edges, n_nodes):
    """ Gets the indices corresponding to any invalid sampled actions.

    Args:
      f_add_idc (torch.Tensor) : Indices for "add" actions for batch of graphs.
      f_conn_idc (torch.Tensor) : Indices for the "connect" actions for batch
        of graphs.
      edges (torch.Tensor) : Edge features tensors for batch of graphs.
      n_nodes (torch.Tensor) : Number of nodes for graphs in a batch.

    Returns:
      invalid_action_idc (torch.Tensor) : Indices corresponding to all invalid
        actions (include the indices below).
      invalid_action_idc_needing_reset (torch.Tensor) : Indices corresponding to
        add actions attempting to add more than the maximum number of nodes.
        These must be treated separately because the "connect to" index needs
        to be reset.
    """
    n_max_nodes = C.dim_nodes[0]

    # empty graphs for which "add" action sampled
    f_add_empty_graphs = torch.nonzero(n_nodes[f_add_idc[0]] == 0)

    # get invalid indices for when adding a new node to a non-empty graph
    invalid_add_idx_tmp = torch.nonzero(f_add_idc[1] >= n_nodes[f_add_idc[0]])
    combined = torch.cat((invalid_add_idx_tmp, f_add_empty_graphs))
    uniques, counts = combined.unique(return_counts=True)
    invalid_add_idc = uniques[counts == 1].unsqueeze(dim=1)  # set difference

    # get invalid indices for when adding a new node to an empty graph
    invalid_add_empty_idc = torch.nonzero(f_add_idc[1] != n_nodes[f_add_idc[0]])
    combined = torch.cat((invalid_add_empty_idc, f_add_empty_graphs))
    uniques, counts = combined.unique(return_counts=True)
    invalid_add_empty_idc = uniques[counts > 1].unsqueeze(dim=1)  # set intersection

    # get invalid indices for when adding more nodes than possible
    invalid_madd_idc = torch.nonzero(f_add_idc[-1] >= n_max_nodes)

    # get invalid indices for when connecting a node to nonexisting node
    invalid_conn_idc = torch.nonzero(f_conn_idc[1] >= n_nodes[f_conn_idc[0]])

    # get invalid indices for when creating self-loops
    invalid_sconn_idc = torch.nonzero(f_conn_idc[1] == f_conn_idc[-1])

    # get invalid indices for when attemting to add multiple edges
    invalid_dconn_idc = torch.nonzero(
        torch.sum(edges, dim=-1)[f_conn_idc[0], f_conn_idc[1], f_conn_idc[-1]] > 0,
    )

    # only need one invalid index per graph
    invalid_action_idc =torch.unique(
        torch.cat(
            (f_add_idc[0][invalid_add_idc],
             f_add_idc[0][invalid_add_empty_idc],
             f_conn_idc[0][invalid_conn_idc],
             f_conn_idc[0][invalid_sconn_idc],
             f_conn_idc[0][invalid_dconn_idc],
             f_add_idc[0][invalid_madd_idc])
        )
    )
    invalid_action_idc_needing_reset = torch.unique(
        torch.cat(
            (invalid_madd_idc, f_add_empty_graphs)
        )
    )

    return invalid_action_idc, invalid_action_idc_needing_reset


def sample_apd(apds, batch_size):
    """ Samples the input APDs for all graphs in the batch.

    Args:
      apds (torch.Tensor) : APDs for a batch of graphs.
      batch_size (int) : Batch size.

    Returns:
      nonzero elements in f_add (torch.Tensor) :
      nonzero elements in f_conn (torch.Tensor) :
      nonzero elements in f_term (torch.Tensor) :
      nlls (torch.Tensor) : Contains NLLs for samples actions.
    """
    m = torch.distributions.Multinomial(1, probs=apds)
    apd_one_hot = m.sample()
    f_add, f_conn, f_term = reshape_apd(apd_one_hot, batch_size)
    nlls = apds[apd_one_hot == 1]

    return (
        torch.nonzero(f_add, as_tuple=True),
        torch.nonzero(f_conn, as_tuple=True),
        torch.nonzero(f_term).view(-1),
        nlls
    )


def reshape_apd(apds, batch_size):
    """ Reshapes the input batch of APDs (inverse to flattening).

    Args:
      apds (torch.Tensor) : APDs for a batch of graphs.
      batch_size (int) : Batch size.

    Returns:
      f_add (torch.Tensor) : Reshaped APD segment for "add" action.
      f_conn (torch.Tensor) : Reshaped APD segment for "connect" action.
      f_term (torch.Tensor) : Reshaped APD segment for "terminate" action.
    """
    # get shapes of "add" and "connect" actions
    f_add_shape = (batch_size, *C.dim_f_add)
    f_conn_shape = (batch_size, *C.dim_f_conn)

    # get ilength of flattened segment of APD corresponding to "add" action
    f_add_size = np.prod(C.dim_f_add)

    # reshape the various APD components
    f_add = torch.reshape(apds[:, :f_add_size], f_add_shape)
    f_conn = torch.reshape(apds[:, f_add_size:-1], f_conn_shape)
    f_term = apds[:, -1]

    return f_add, f_conn, f_term


def graph_to_graph(idx, generated_nodes, generated_edges, generated_n_nodes):
    """ Converts a molecular graph representation into `GenerationGraph` objects.

    Args:
      idx (int) : Index for the molecular graph to convert.
      generated_nodes (torch.Tensor) : Node features tensors for all generated graphs.
      generated_edges (torch.Tensor) : Edge features tensors for all generated graphs.
      generated_n_nodes (torch.Tensor) : Number of nodes for all generated graphs.

    Returns :
      graph (GenerationGraph) :
    """
    try:
        # first get the `rdkit.Mol` object corresponding to the selected graph
        mol = graph_to_mol(generated_nodes[idx],
                           generated_edges[idx],
                           generated_n_nodes[idx])
    except (IndexError, AttributeError):  # raised when graph is empty
        mol = None

    # use the `rdkit.Mol` object, and node and edge features tensors, to get
    # the `GenerationGraph` object
    graph = GenerationGraph(constants=C,
                            molecule=mol,
                            node_features=generated_nodes[idx],
                            edge_features=generated_edges[idx])
    return graph


def graph_to_mol(node_features, edge_features, n_nodes):
    """ Converts input graph represenetation (node and edge features) into an
    `rdkit.Mol` object.

    Args:
      node_features (torch.Tensor) : Node features tensor.
      edge_features (torch.Tensor) : Edge features tensor.
      n_nodes (int) : Number of nodes in the graph representation.

    Returns:
      molecule (rdkit.Chem.Mol) : Molecule object.
    """
    # create empty editable `rdkit.Chem.Mol` object
    molecule = rdkit.Chem.RWMol()
    node_to_idx = {}

    # add atoms to editable mol object
    for v in range(n_nodes):
        atom_to_add = features_to_atom(v, node_features)
        molecule_idx = molecule.AddAtom(atom_to_add)
        node_to_idx[v] = molecule_idx

    # add bonds to atoms in editable mol object; to not add the same bond twice
    # (which leads to an error), mask half of the edge features beyond diagonal
    n_max_nodes = C.dim_nodes[0]
    edge_mask = torch.triu(
        torch.ones((n_max_nodes, n_max_nodes), device="cuda"), diagonal=1
    )
    edge_mask = edge_mask.view(n_max_nodes, n_max_nodes, 1)
    edges_idc = torch.nonzero(edge_features * edge_mask)

    for vi, vj, b in edges_idc:
        molecule.AddBond(
            node_to_idx[vi.item()],
            node_to_idx[vj.item()],
            C.int_to_bondtype[b.item()],
        )

    # convert editable mol object to non-editable mol object
    try:
        molecule.GetMol()
    except AttributeError:  # will throw an error if molecule is `None`
        pass

    # correct for ignored Hs
    if C.ignore_H and molecule:
        try:
            rdkit.Chem.SanitizeMol(molecule)
        except ValueError:
            # throws 1st exception if "molecule" is too ugly to be corrected
            pass

    return molecule

def features_to_atom(node_idx, node_features):
    """ Converts the node feature vector corresponding to the specified node
    into an atom object.

    Args:
      node_idx (int) : Index denoting the specific node on the graph to convert.
      node_features (torch.Tensor) : Node features tensor for one graph.

    Returns:
      new_atom (rdkit.Atom) : Atom object corresponding to specified node
        features.
    """
    # get all the nonzero indices in the specified node feature vector
    nonzero_idc = torch.nonzero(node_features[node_idx])

    # determine atom symbol
    atom_idx = nonzero_idc[0]
    atom_type = C.atom_types[atom_idx]

    # initialize atom
    new_atom = rdkit.Chem.Atom(atom_type)

    # determine formal charge
    fc_idx = nonzero_idc[1] - C.n_atom_types
    formal_charge = C.formal_charge[fc_idx]

    new_atom.SetFormalCharge(formal_charge)  # set property

    # determine number of implicit Hs (if used)
    if not C.use_explicit_H and not C.ignore_H:
        total_num_h_idx = nonzero_idc[2] - C.n_atom_types - C.n_formal_charge
        total_num_h = C.imp_H[total_num_h_idx]

        new_atom.SetUnsignedProp("_TotalNumHs", total_num_h)  # set property
    elif C.ignore_H:
        # these will be set with structure is "sanitized" (corrected) later
        # in `mol_to_graph()`.
        pass

    # determine chirality (if used)
    if C.use_chirality:
        cip_code_idx = (
            nonzero_idc[-1]
            - C.n_atom_types
            - C.n_formal_charge
            - bool(not C.use_explicit_H and not C.ignore_H) * C.n_imp_H
        )
        cip_code = C.chirality[cip_code_idx]
        new_atom.SetProp("_CIPCode", cip_code)  # set property

    return new_atom


def build_graphs(model, n_graphs_to_generate, batch_size):
    """ Generates molecular graphs in batches.

    Args:
      model (modules.SummationMPNN or modules.AggregationMPNN or
        modules.EdgeMPNN) : Neural net model.
      n_graphs_to_generate (int) : Total number of graphs to generate.
      batch_size (int) : Size of batches to use for graph generation.

    Returns:
      graphs (`list` of `GenerationGraph`s) : Generated molecular graphs.
      generated_nlls (torch.Tensor) : Sampled NLLs per action for the
        generated graphs.
      final_nlls (torch.Tensor) : Final total NLLs (sum) for the generated
        graphs.
      properly_terminated_graphs (torch.Tensor) : Indicates if graphs were
        properly terminated or not using a 0 or 1.
    """
    # start the timer
    t = time.time()

    # define the softmax for use later
    softmax = torch.nn.Softmax(dim=1)

    # initialize node and edge features tensors for batch of graphs, as well
    # as a tensor to keep track of the number of nodes per graph
    nodes, edges, n_nodes = initialize_graph_batch(batch_size=batch_size)

    # allocate tensors for finished graphs; these will get filled in gradually
    # as graphs terminate
    (
        generated_nodes, 
        generated_edges,
        generated_n_nodes, 
        nlls, 
        generated_nlls, 
        properly_terminated_graphs
    ) = allocate_graph_tensors(n_graphs_to_generate,
                               batch_size)

    # keep track of a few things...
    n_generated_so_far = 0
    t_bar = tqdm(total=n_graphs_to_generate)
    generation_round = 0

    # generate graphs in a batch until the total number of graphs is reached
    while n_generated_so_far < n_graphs_to_generate:

        # skip dummy node after calling model (only need it for predicting APDs)
        apd = softmax(model(nodes, edges))[1:]
        nodes = nodes[1:]
        edges = edges[1:]
        n_nodes = n_nodes[1:]
        nlls = nlls[1:]

        # get the actions from the predicted APDs
        add, conn, term, invalid, nlls_just_sampled = get_actions(apd,
                                                                  edges,
                                                                  n_nodes,
                                                                  batch_size)

        # indicate (with a 1) the structures which have been properly terminated
        properly_terminated_graphs[n_generated_so_far : n_generated_so_far + len(term)] = 1
        termination_idc = torch.cat((term, invalid))

        # copy the graphs to be terminated (indicated by `terminated_idc`) to
        # the tensors for finished graphs (e.g. `generated_nodes`, etc)
        (
            n_generated_so_far, 
            generated_nodes,
            generated_edges, 
            generated_n_nodes,
            generated_nlls
        ) = copy_terminated_graphs(termination_idc,
                                   n_generated_so_far,
                                   nodes,
                                   edges,
                                   n_nodes,
                                   generated_nodes,
                                   generated_edges,
                                   generated_n_nodes,
                                   generation_round,
                                   nlls,
                                   nlls_just_sampled,
                                   generated_nlls)

        # apply actions to all graphs (note: applies actions to terminated
        # graphs too to keep on GPU, as this makes generation faster and graphs
        # will be removed anyway)
        nodes, edges, n_nodes, nlls = apply_actions(add,
                                                    conn,
                                                    nodes,
                                                    edges,
                                                    n_nodes,
                                                    generation_round,
                                                    nlls,
                                                    nlls_just_sampled)

        # after actions are applied, reset graphs which were set to terminate
        # this round
        nodes, edges, n_nodes, nlls = reset_graphs(n_graphs_to_generate,
                                                   termination_idc,
                                                   nodes,
                                                   edges,
                                                   n_nodes,
                                                   nlls,
                                                   batch_size)

        # update variables that are being kept track of
        t_bar.update(len(termination_idc))
        generation_round += 1

    # done generating
    t_bar.close()

    # get the time it took to generate graphs
    t = time.time() - t
    print(f"Generated {n_generated_so_far} molecules in {t:.4} s")
    print(f"--{n_generated_so_far/t:4.5} molecules/s")

    # convert the molecular graphs (currently separate node and edge features
    # tensors) into `GenerationGraph` objects
    graphs = []
    for graph_idx in range(n_graphs_to_generate):
        graphs.append(
            graph_to_graph(graph_idx, generated_nodes, generated_edges, generated_n_nodes)
        )

    # sum NLLs over all the actions to get the total NLLs for each structurei
    # and remove extra zero padding
    final_nlls = torch.sum(generated_nlls, dim=1)[:len(graphs)]

    # remove extra zero padding from `generated_nlls` and `properly_terminated_graphs`
    generated_nlls = generated_nlls[generated_nlls != 0]
    properly_terminated_graphs = properly_terminated_graphs[:len(graphs)]


    return graphs, generated_nlls, final_nlls, properly_terminated_graphs
