# load general packages and functions
import math
import torch

# load program-specific functions
import gnn.aggregation_mpnn
import gnn.edge_mpnn
import gnn.summation_mpnn
import gnn.modules

# defines specific MPNN implementations


# some constants
BIG_NEGATIVE = -1e6
BIG_POSITIVE = 1e6

class MNN(gnn.summation_mpnn.SummationMPNN):
    """ The "message neural network" model.

    Args:
      *edge_features (int) : Number of edge features.
      *f_add_elems (int) : Number of elements PER NODE in `f_add` (e.g.
        `n_atom_types` * `n_formal_charge` * `n_edge_features`).
      mlp1_depth (int) : Num layers in first-tier MLP in APD readout.
      mlp1_dropout_p (float) : Dropout probability in first-tier MLP in APD readout.
      mlp1_hidden_dim (int) : Number of weights (layer width) in first-tier MLP
        in APD readout.
      mlp2_depth (int) : Num layers in second-tier MLP in APD readout.
      mlp2_dropout_p (float) : Dropout probability in second-tier MLP in APD readout.
      mlp2_hidden_dim (int) : Number of weights (layer width) in second-tier MLP
        in APD readout.
      hidden_node_features (int) : Indicates length of node hidden states.
      *initialization (str) : Initialization scheme for weights in feed-forward
        networks ('none', 'uniform', or 'normal').
      message_passes (int) : Number of message passing steps.
      message_size (int) : Size of message passed ('enn' MLP output size).
      *n_nodes_largest_graph (int) : Number of nodes in the largest graph.
      *node_features (int) : Number of node features (e.g. `n_atom_types` +
        `n_formal_charge`).
    """
    def __init__(self, edge_features, f_add_elems, mlp1_depth, mlp1_dropout_p,
                 mlp1_hidden_dim, mlp2_depth, mlp2_dropout_p, mlp2_hidden_dim,
                 hidden_node_features, initialization, message_passes,
                 message_size, n_nodes_largest_graph, node_features):

        super(MNN, self).__init__(node_features, hidden_node_features, edge_features, message_size, message_passes)

        message_weights = torch.Tensor(message_size, hidden_node_features, edge_features)

        message_weights = message_weights.to("cuda", non_blocking=True)

        self.message_weights = torch.nn.Parameter(message_weights)

        self.gru = torch.nn.GRUCell(
            input_size=message_size, hidden_size=hidden_node_features, bias=True
        )

        self.APDReadout = gnn.modules.GlobalReadout(
            node_emb_size=hidden_node_features,
            graph_emb_size=hidden_node_features,
            mlp1_hidden_dim=mlp1_hidden_dim,
            mlp1_depth=mlp1_depth,
            mlp1_dropout_p=mlp1_dropout_p,
            mlp2_hidden_dim=mlp2_hidden_dim,
            mlp2_depth=mlp2_depth,
            mlp2_dropout_p=mlp2_dropout_p,
            init=initialization,
            f_add_elems=f_add_elems,
            f_conn_elems=edge_features,
            f_term_elems=1,
            max_n_nodes=n_nodes_largest_graph
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdev = 1.0 / math.sqrt(self.message_weights.size(1))

        self.message_weights.data.uniform_(-stdev, stdev)

    def message_terms(self, nodes, node_neighbours, edges):
        edges_view = edges.view(-1, 1, 1, self.edge_features)
        weights_for_each_edge = (edges_view * self.message_weights.unsqueeze(0)).sum(3)
        return torch.matmul(
            weights_for_each_edge, node_neighbours.unsqueeze(-1)
        ).squeeze()

    def update(self, nodes, messages):
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = torch.sum(hidden_nodes, dim=1)
        output = self.APDReadout(hidden_nodes, graph_embeddings)

        return output


class S2V(gnn.summation_mpnn.SummationMPNN):
    """ The "set2vec" model.

    Args:
      *edge_features (int) : Number of edge features.
      enn_depth (int) : Num layers in 'enn' MLP.
      enn_dropout_p (float) : Dropout probability in 'enn' MLP.
      enn_hidden_dim (int) : Number of weights (layer width) in 'enn' MLP.
      *f_add_elems (int) : Number of elements PER NODE in `f_add` (e.g.
        `n_atom_types` * `n_formal_charge` * `n_edge_features`).
      mlp1_depth (int) : Num layers in first-tier MLP in APD readout function.
      mlp1_dropout_p (float) : Dropout probability in first-tier MLP in APD
        readout function.
      mlp1_hidden_dim (int) : Number of weights (layer width) in first-tier MLP
        in APD readout function.
      mlp2_depth (int) : Num layers in second-tier MLP in APD readout function.
      mlp2_dropout_p (float) : Dropout probability in second-tier MLP in APD
        readout function.
      mlp2_hidden_dim (int) : Number of weights (layer width) in second-tier MLP
        in APD readout function.
      hidden_node_features (int) : Indicates length of node hidden states.
      *initialization (str) : Initialization scheme for weights in feed-forward
        networks ('none', 'uniform', or 'normal').
      message_passes (int) : Number of message passing steps.
      message_size (int) : Size of message passed (input size to `GRU`).
      *n_nodes_largest_graph (int) : Number of nodes in the largest graph.
      *node_features (int) : Number of node features (e.g. `n_atom_types` +
        `n_formal_charge`).
      s2v_lstm_computations (int) : Number of LSTM computations (loop) in S2V readout.
      s2v_memory_size (int) : Number of input features and hidden state size in
        LSTM cell in S2V readout.
    """
    def __init__(self, edge_features, enn_depth, enn_dropout_p, enn_hidden_dim,
                 f_add_elems, mlp1_depth, mlp1_dropout_p, mlp1_hidden_dim,
                 mlp2_dropout_p, mlp2_depth, mlp2_hidden_dim, hidden_node_features,
                 initialization, message_passes, message_size, n_nodes_largest_graph,
                 node_features, s2v_lstm_computations, s2v_memory_size):

        super(S2V, self).__init__(node_features, hidden_node_features, edge_features, message_size, message_passes)

        self.n_nodes_largest_graph = n_nodes_largest_graph

        self.enn = gnn.modules.MLP(
            in_features=edge_features,
            hidden_layer_sizes=[enn_hidden_dim] * enn_depth,
            out_features=hidden_node_features * message_size,
            init=initialization,
            dropout_p=enn_dropout_p
        )

        self.gru = torch.nn.GRUCell(
            input_size=message_size, hidden_size=hidden_node_features, bias=True
        )

        self.s2v = gnn.modules.Set2Vec(
            node_features=node_features,
            hidden_node_features=hidden_node_features,
            lstm_computations=s2v_lstm_computations,
            memory_size=s2v_memory_size
        )

        self.APDReadout = gnn.modules.GlobalReadout(
            node_emb_size=hidden_node_features,
            graph_emb_size=s2v_memory_size * 2,
            mlp1_hidden_dim=mlp1_hidden_dim,
            mlp1_depth=mlp1_depth,
            mlp1_dropout_p=mlp1_dropout_p,
            mlp2_hidden_dim=mlp2_hidden_dim,
            mlp2_depth=mlp2_depth,
            mlp2_dropout_p=mlp2_dropout_p,
            init=initialization,
            f_add_elems=f_add_elems,
            f_conn_elems=edge_features,
            f_term_elems=1,
            max_n_nodes=n_nodes_largest_graph
        )

    def message_terms(self, nodes, node_neighbours, edges):
        enn_output = self.enn(edges)
        matrices = enn_output.view(-1, self.message_size, self.hidden_node_features)
        msg_terms = torch.matmul(matrices, node_neighbours.unsqueeze(-1)).squeeze(-1)

        return msg_terms

    def update(self, nodes, messages):
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.s2v(hidden_nodes, input_nodes, node_mask)
        output = self.APDReadout(hidden_nodes, graph_embeddings)

        return output


class AttentionS2V(gnn.aggregation_mpnn.AggregationMPNN):
    """ The "set2vec with attention" model.

    Args:
      att_depth (int) : Num layers in 'att_enn' MLP.
      att_dropout_p (float) : Dropout probability in 'att_enn' MLP.
      att_hidden_dim (int) : Number of weights (layer width) in 'att_enn' MLP.
      *edge_features (int) : Number of edge features.
      enn_depth (int) : Num layers in 'enn' MLP.
      enn_dropout_p (float) : Dropout probability in 'enn' MLP.
      enn_hidden_dim (int) : Number of weights (layer width) in 'enn' MLP.
      *f_add_elems (int) : Number of elements PER NODE in `f_add` (e.g.
        `n_atom_types` * `n_formal_charge` * `n_edge_features`).
      mlp1_depth (int) : Num layers in first-tier MLP in APD readout function.
      mlp1_dropout_p (float) : Dropout probability in first-tier MLP in APD
        readout function.
      mlp1_hidden_dim (int) : Number of weights (layer width) in first-tier MLP
        in APD readout function.
      mlp2_depth (int) : Num layers in second-tier MLP in APD readout function.
      mlp2_dropout_p (float) : Dropout probability in second-tier MLP in APD
        readout function.
      mlp2_hidden_dim (int) : Number of weights (layer width) in second-tier MLP
        in APD readout function.
      hidden_node_features (int) : Indicates length of node hidden states.
      *initialization (str) : Initialization scheme for weights in feed-forward
        networks ('none', 'uniform', or 'normal').
      message_passes (int) : Number of message passing steps.
      message_size (int) : Size of message passed (output size of 'att_enn' MLP,
        input size to `GRU`).
      *n_nodes_largest_graph (int) : Number of nodes in the largest graph.
      *node_features (int) : Number of node features (e.g. `n_atom_types` +
        `n_formal_charge`).
      s2v_lstm_computations (int) : Number of LSTM computations (loop) in S2V readout.
      s2v_memory_size (int) : Number of input features and hidden state size in
        LSTM cell in S2V readout.
    """
    def __init__(self, att_depth, att_dropout_p, att_hidden_dim, edge_features,
                 enn_depth, enn_dropout_p, enn_hidden_dim, f_add_elems, mlp1_depth,
                 mlp1_dropout_p, mlp1_hidden_dim, mlp2_depth, mlp2_dropout_p,
                 mlp2_hidden_dim, hidden_node_features, initialization,
                 message_passes, message_size, n_nodes_largest_graph,
                 node_features, s2v_lstm_computations, s2v_memory_size):

        super(AttentionS2V, self).__init__(node_features, hidden_node_features, edge_features, message_size, message_passes)

        self.n_nodes_largest_graph = n_nodes_largest_graph
        self.message_size = message_size
        self.enn = gnn.modules.MLP(
            in_features=edge_features,
            hidden_layer_sizes=[enn_hidden_dim] * enn_depth,
            out_features=hidden_node_features * message_size,
            init=initialization,
            dropout_p=enn_dropout_p
        )

        self.att_enn = gnn.modules.MLP(
            in_features=hidden_node_features + edge_features,
            hidden_layer_sizes=[att_hidden_dim] * att_depth,
            out_features=message_size,
            init=initialization,
            dropout_p=att_dropout_p
        )

        self.gru = torch.nn.GRUCell(
            input_size=message_size, hidden_size=hidden_node_features, bias=True
        )

        self.s2v = gnn.modules.Set2Vec(
            node_features=node_features,
            hidden_node_features=hidden_node_features,
            lstm_computations=s2v_lstm_computations,
            memory_size=s2v_memory_size,
        )

        self.APDReadout = gnn.modules.GlobalReadout(
            node_emb_size=hidden_node_features,
            graph_emb_size=s2v_memory_size * 2,
            mlp1_hidden_dim=mlp1_hidden_dim,
            mlp1_depth=mlp1_depth,
            mlp1_dropout_p=mlp1_dropout_p,
            mlp2_hidden_dim=mlp2_hidden_dim,
            mlp2_depth=mlp2_depth,
            mlp2_dropout_p=mlp2_dropout_p,
            init=initialization,
            f_add_elems=f_add_elems,
            f_conn_elems=edge_features,
            f_term_elems=1,
            max_n_nodes=n_nodes_largest_graph,
        )

    def aggregate_message(self, nodes, node_neighbours, edges, mask):
        Softmax = torch.nn.Softmax(dim=1)

        max_node_degree = node_neighbours.shape[1]

        enn_output = self.enn(edges)
        matrices = enn_output.view(
            -1, max_node_degree, self.message_size, self.hidden_node_features
        )
        message_terms = torch.matmul(matrices, node_neighbours.unsqueeze(-1)).squeeze()

        att_enn_output = self.att_enn(torch.cat((edges, node_neighbours), dim=2))
        energies = att_enn_output.view(-1, max_node_degree, self.message_size)
        energy_mask = (1 - mask).float() * BIG_NEGATIVE
        weights = Softmax(energies + energy_mask.unsqueeze(-1))

        return (weights * message_terms).sum(1)

    def update(self, nodes, messages):
        messages = messages + torch.zeros(self.message_size, device="cuda")
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.s2v(hidden_nodes, input_nodes, node_mask)
        output = self.APDReadout(hidden_nodes, graph_embeddings)

        return output


class GGNN(gnn.summation_mpnn.SummationMPNN):
    """ The "gated-graph neural network" model.

    Args:
      *edge_features (int) : Number of edge features.
      enn_depth (int) : Num layers in 'enn' MLP.
      enn_dropout_p (float) : Dropout probability in 'enn' MLP.
      enn_hidden_dim (int) : Number of weights (layer width) in 'enn' MLP.
      *f_add_elems (int) : Number of elements PER NODE in `f_add` (e.g.
        `n_atom_types` * `n_formal_charge` * `n_edge_features`).
      mlp1_depth (int) : Num layers in first-tier MLP in APD readout function.
      mlp1_dropout_p (float) : Dropout probability in first-tier MLP in APD
        readout function.
      mlp1_hidden_dim (int) : Number of weights (layer width) in first-tier MLP
        in APD readout function.
      mlp2_depth (int) : Num layers in second-tier MLP in APD readout function.
      mlp2_dropout_p (float) : Dropout probability in second-tier MLP in APD
        readout function.
      mlp2_hidden_dim (int) : Number of weights (layer width) in second-tier MLP
        in APD readout function.
      gather_att_depth (int) : Num layers in 'gather_att' MLP in graph gather block.
      gather_att_dropout_p (float) : Dropout probability in 'gather_att' MLP in
        graph gather block.
      gather_att_hidden_dim (int) : Number of weights (layer width) in
        'gather_att' MLP in graph gather block.
      gather_emb_depth (int) : Num layers in 'gather_emb' MLP in graph gather block.
      gather_emb_dropout_p (float) : Dropout probability in 'gather_emb' MLP in
        graph gather block.
      gather_emb_hidden_dim (int) : Number of weights (layer width) in
        'gather_emb' MLP in graph gather block.
      gather_width (int) : Output size of graph gather block block.
      hidden_node_features (int) : Indicates length of node hidden states.
      *initialization (str) : Initialization scheme for weights in feed-forward
        networks ('none', 'uniform', or 'normal').
      message_passes (int) : Number of message passing steps.
      message_size (int) : Size of message passed (output size of all MLPs in
        message aggregation step, input size to `GRU`).
      *n_nodes_largest_graph (int) : Number of nodes in the largest graph.
      *node_features (int) : Number of node features (e.g. `n_atom_types` +
        `n_formal_charge`).
    """
    def __init__(self, edge_features, enn_depth, enn_dropout_p, enn_hidden_dim,
                 f_add_elems, mlp1_depth, mlp1_dropout_p, mlp1_hidden_dim,
                 mlp2_depth, mlp2_dropout_p, mlp2_hidden_dim, gather_att_depth,
                 gather_att_dropout_p, gather_att_hidden_dim, gather_width,
                 gather_emb_depth, gather_emb_dropout_p, gather_emb_hidden_dim,
                 hidden_node_features, initialization, message_passes,
                 message_size, n_nodes_largest_graph, node_features):

        super(GGNN, self).__init__(node_features, hidden_node_features, edge_features, message_size, message_passes)

        self.n_nodes_largest_graph = n_nodes_largest_graph

        self.msg_nns = torch.nn.ModuleList()
        for _ in range(edge_features):
            self.msg_nns.append(
                gnn.modules.MLP(
                    in_features=hidden_node_features,
                    hidden_layer_sizes=[enn_hidden_dim] * enn_depth,
                    out_features=message_size,
                    init=initialization,
                    dropout_p=enn_dropout_p,
                )
            )

        self.gru = torch.nn.GRUCell(
            input_size=message_size, hidden_size=hidden_node_features, bias=True
        )

        self.gather = gnn.modules.GraphGather(
            node_features=node_features,
            hidden_node_features=hidden_node_features,
            out_features=gather_width,
            att_depth=gather_att_depth,
            att_hidden_dim=gather_att_hidden_dim,
            att_dropout_p=gather_att_dropout_p,
            emb_depth=gather_emb_depth,
            emb_hidden_dim=gather_emb_hidden_dim,
            emb_dropout_p=gather_emb_dropout_p,
            init=initialization,
        )

        self.APDReadout = gnn.modules.GlobalReadout(
            node_emb_size=hidden_node_features,
            graph_emb_size=gather_width,
            mlp1_hidden_dim=mlp1_hidden_dim,
            mlp1_depth=mlp1_depth,
            mlp1_dropout_p=mlp1_dropout_p,
            mlp2_hidden_dim=mlp2_hidden_dim,
            mlp2_depth=mlp2_depth,
            mlp2_dropout_p=mlp2_dropout_p,
            init=initialization,
            f_add_elems=f_add_elems,
            f_conn_elems=edge_features,
            f_term_elems=1,
            max_n_nodes=n_nodes_largest_graph,
        )

    def message_terms(self, nodes, node_neighbours, edges):
        edges_v = edges.view(-1, self.edge_features, 1)
        node_neighbours_v = edges_v * node_neighbours.view(-1, 1, self.hidden_node_features)
        terms_masked_per_edge = [
            edges_v[:, i, :] * self.msg_nns[i](node_neighbours_v[:, i, :])
            for i in range(self.edge_features)
        ]
        return sum(terms_masked_per_edge)

    def update(self, nodes, messages):
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        output = self.APDReadout(hidden_nodes, graph_embeddings)

        return output



class AttentionGGNN(gnn.aggregation_mpnn.AggregationMPNN):
    """ The "GGNN with attention" model.

    Args:
      att_depth (int) : Num layers in 'att_nns' MLP (message aggregation step).
      att_dropout_p (float) : Dropout probability in 'att_nns' MLP (message
        aggregation step).
      att_hidden_dim (int) : Number of weights (layer width) in 'att_nns' MLP
        (message aggregation step).
      *edge_features (int) : Number of edge features.
      *f_add_elems (int) : Number of elements PER NODE in `f_add` (e.g.
        `n_atom_types` * `n_formal_charge` * `n_edge_features`).
      mlp1_depth (int) : Num layers in first-tier MLP in APD readout function.
      mlp1_dropout_p (float) : Dropout probability in first-tier MLP in APD
        readout function.
      mlp1_hidden_dim (int) : Number of weights (layer width) in first-tier MLP
        in APD readout function.
      mlp2_depth (int) : Num layers in second-tier MLP in APD readout function.
      mlp2_dropout_p (float) : Dropout probability in second-tier MLP in APD
        readout function.
      mlp2_hidden_dim (int) : Number of weights (layer width) in second-tier MLP
        in APD readout function.
      gather_att_depth (int) : Num layers in 'gather_att' MLP in graph gather block.
      gather_att_dropout_p (float) : Dropout probability in 'gather_att' MLP in
        graph gather block.
      gather_att_hidden_dim (int) : Number of weights (layer width) in
        'gather_att' MLP in graph gather block.
      gather_emb_depth (int) : Num layers in 'gather_emb' MLP in graph gather block.
      gather_emb_dropout_p (float) : Dropout probability in 'gather_emb' MLP in
        graph gather block.
      gather_emb_hidden_dim (int) : Number of weights (layer width) in
        'gather_emb' MLP in graph gather block.
      gather_width (int) : Output size of graph gather block block.
      hidden_node_features (int) : Indicates length of node hidden states.
      *initialization (str) : Initialization scheme for weights in feed-forward
        networks ('none', 'uniform', or 'normal').
      message_passes (int) : Number of message passing steps.
      message_size (int) : Size of message passed (output size of all MLPs in
        message aggregation step, input size to `GRU`).
      msg_depth (int) : Num layers in 'msg_nns' MLP (message aggregation step).
      msg_dropout_p (float) : Dropout probability in 'msg_nns' MLP (message
        aggregation step).
      msg_hidden_dim (int) : Number of weights (layer width) in 'msg_nns' MLP
        (message aggregation step).
      *n_nodes_largest_graph (int) : Number of nodes in the largest graph.
      *node_features (int) : Number of node features (e.g. `n_atom_types` +
        `n_formal_charge`).
    """
    def __init__(self, att_depth, att_dropout_p, att_hidden_dim, edge_features,
                 f_add_elems, mlp1_depth, mlp1_dropout_p, mlp1_hidden_dim,
                 mlp2_depth, mlp2_dropout_p, mlp2_hidden_dim, gather_att_depth,
                 gather_att_dropout_p, gather_att_hidden_dim, gather_emb_depth,
                 gather_emb_dropout_p, gather_emb_hidden_dim, gather_width,
                 hidden_node_features, initialization, message_passes,
                 message_size, msg_depth, msg_dropout_p, msg_hidden_dim,
                 n_nodes_largest_graph, node_features):

        super(AttentionGGNN, self).__init__(node_features, hidden_node_features, edge_features, message_size, message_passes)

        self.n_nodes_largest_graph = n_nodes_largest_graph
        self.msg_nns = torch.nn.ModuleList()
        self.att_nns = torch.nn.ModuleList()

        for _ in range(edge_features):
            self.msg_nns.append(
                gnn.modules.MLP(
                  in_features=hidden_node_features,
                  hidden_layer_sizes=[msg_hidden_dim] * msg_depth,
                  out_features=message_size,
                  init=initialization,
                  dropout_p=msg_dropout_p,
                )
            )
            self.att_nns.append(
                gnn.modules.MLP(
                  in_features=hidden_node_features,
                  hidden_layer_sizes=[att_hidden_dim] * att_depth,
                  out_features=message_size,
                  init=initialization,
                  dropout_p=att_dropout_p,
                )
            )
        self.gru = torch.nn.GRUCell(
            input_size=message_size, hidden_size=hidden_node_features, bias=True
        )

        self.gather = gnn.modules.GraphGather(
            node_features=node_features,
            hidden_node_features=hidden_node_features,
            out_features=gather_width,
            att_depth=gather_att_depth,
            att_hidden_dim=gather_att_hidden_dim,
            att_dropout_p=gather_att_dropout_p,
            emb_depth=gather_emb_depth,
            emb_hidden_dim=gather_emb_hidden_dim,
            emb_dropout_p=gather_emb_dropout_p,
            init=initialization,
        )

        self.APDReadout = gnn.modules.GlobalReadout(
            node_emb_size=hidden_node_features,
            graph_emb_size=gather_width,
            mlp1_hidden_dim=mlp1_hidden_dim,
            mlp1_depth=mlp1_depth,
            mlp1_dropout_p=mlp1_dropout_p,
            mlp2_hidden_dim=mlp2_hidden_dim,
            mlp2_depth=mlp2_depth,
            mlp2_dropout_p=mlp2_dropout_p,
            init=initialization,
            f_add_elems=f_add_elems,
            f_conn_elems=edge_features,
            f_term_elems=1,
            max_n_nodes=n_nodes_largest_graph,
        )

    def aggregate_message(self, nodes, node_neighbours, edges, mask):
        Softmax = torch.nn.Softmax(dim=1)

        energy_mask = (mask == 0).float() * BIG_POSITIVE

        embeddings_masked_per_edge = [
            edges[:, :, i].unsqueeze(-1) * self.msg_nns[i](node_neighbours)
            for i in range(self.edge_features)
        ]
        energies_masked_per_edge = [
            edges[:, :, i].unsqueeze(-1) * self.att_nns[i](node_neighbours)
            for i in range(self.edge_features)
        ]

        embedding = sum(embeddings_masked_per_edge)
        energies = sum(energies_masked_per_edge) - energy_mask.unsqueeze(-1)

        attention = Softmax(energies)

        return torch.sum(attention * embedding, dim=1)

    def update(self, nodes, messages):
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        output = self.APDReadout(hidden_nodes, graph_embeddings)

        return output


class EMN(gnn.edge_mpnn.EdgeMPNN):
    """ The "edge memory network" model.

    Args:
      att_depth (int) : Num layers in 'att_msg_nn' MLP (edge propagation step).
      att_dropout_p (float) : Dropout probability in 'att_msg_nn' MLP (edge
        propagation step).
      att_hidden_dim (int) : Number of weights (layer width) in 'att_msg_nn' MLP
        (edge propagation step).
      edge_emb_depth (int) : Num layers in 'embedding_nn' MLP (edge processing step).
      edge_emb_dropout_p (float) : Dropout probability in 'embedding_nn' MLP
        (edge processing step).
      edge_emb_hidden_dim (int) : Number of weights (layer width) in
        'embedding_nn' MLP (edge processing step).
      edge_emb_size (int) : Output size of all MLPs in edge propagation and
        processing steps (input size to graph gather block).
      *edge_features (int) : Number of edge features.
      *f_add_elems (int) : Number of elements PER NODE in `f_add` (e.g.
        `n_atom_types` * `n_formal_charge` * `n_edge_features`).
      mlp1_depth (int) : Num layers in first-tier MLP in APD readout function.
      mlp1_dropout_p (float) : Dropout probability in first-tier MLP in APD
        readout function.
      mlp1_hidden_dim (int) : Number of weights (layer width) in first-tier MLP
        in APD readout function.
      mlp2_depth (int) : Num layers in second-tier MLP in APD readout function.
      mlp2_dropout_p (float) : Dropout probability in second-tier MLP in APD
        readout function.
      mlp2_hidden_dim (int) : Number of weights (layer width) in second-tier MLP
        in APD readout function.
      gather_att_depth (int) : Num layers in 'gather_att' MLP in graph gather block.
      gather_att_dropout_p (float) : Dropout probability in 'gather_att' MLP in
        graph gather block.
      gather_att_hidden_dim (int) : Number of weights (layer width) in
        'gather_att' MLP in graph gather block.
      gather_emb_depth (int) : Num layers in 'gather_emb' MLP in graph gather block.
      gather_emb_dropout_p (float) : Dropout probability in 'gather_emb' MLP in
        graph gather block.
      gather_emb_hidden_dim (int) : Number of weights (layer width) in
        'gather_emb' MLP in graph gather block.
      gather_width (int) : Output size of graph gather block block.
      *initialization (str) : Initialization scheme for weights in feed-forward
        networks ('none', 'uniform', or 'normal').
      message_passes (int) : Number of message passing steps.
      msg_depth (int) : Num layers in 'emb_msg_nn' MLP (edge propagation step).
      msg_dropout_p (float) : Dropout probability in 'emb_msg_n' MLP (edge
        propagation step).
      msg_hidden_dim (int) : Number of weights (layer width) in 'emb_msg_nn' MLP
        (edge propagation step).
      *n_nodes_largest_graph (int) : Number of nodes in the largest graph.
      *node_features (int) : Number of node features (e.g. `n_atom_types` +
        `n_formal_charge`).
    """
    def __init__(self, att_depth, att_dropout_p, att_hidden_dim, edge_emb_depth,
                 edge_emb_dropout_p, edge_emb_hidden_dim, edge_emb_size,
                 edge_features, f_add_elems, mlp1_depth, mlp1_dropout_p,
                 mlp1_hidden_dim, mlp2_depth, mlp2_dropout_p, mlp2_hidden_dim,
                 gather_att_depth, gather_att_dropout_p, gather_att_hidden_dim,
                 gather_emb_depth, gather_emb_dropout_p, gather_emb_hidden_dim,
                 gather_width, initialization, message_passes, msg_depth,
                 msg_dropout_p, msg_hidden_dim, n_nodes_largest_graph, node_features):

        super(EMN, self).__init__(edge_features, edge_emb_size, message_passes, n_nodes_largest_graph)

        self.n_nodes_largest_graph = n_nodes_largest_graph

        self.embedding_nn = gnn.modules.MLP(
            in_features=node_features * 2 + edge_features,
            hidden_layer_sizes=[edge_emb_hidden_dim] * edge_emb_depth,
            out_features=edge_emb_size,
            init=initialization,
            dropout_p=edge_emb_dropout_p,
        )

        self.emb_msg_nn = gnn.modules.MLP(
            in_features=edge_emb_size,
            hidden_layer_sizes=[msg_hidden_dim] * msg_depth,
            out_features=edge_emb_size,
            init=initialization,
            dropout_p=msg_dropout_p,
        )

        self.att_msg_nn = gnn.modules.MLP(
            in_features=edge_emb_size,
            hidden_layer_sizes=[att_hidden_dim] * att_depth,
            out_features=edge_emb_size,
            init=initialization,
            dropout_p=att_dropout_p,
        )

        self.gru = torch.nn.GRUCell(
            input_size=edge_emb_size, hidden_size=edge_emb_size, bias=True
        )

        self.gather = gnn.modules.GraphGather(
            node_features=edge_emb_size,
            hidden_node_features=edge_emb_size,
            out_features=gather_width,
            att_depth=gather_att_depth,
            att_hidden_dim=gather_att_hidden_dim,
            att_dropout_p=gather_att_dropout_p,
            emb_depth=gather_emb_depth,
            emb_hidden_dim=gather_emb_hidden_dim,
            emb_dropout_p=gather_emb_dropout_p,
            init=initialization,
        )

        self.APDReadout = gnn.modules.GlobalReadout(
            node_emb_size=edge_emb_size,
            graph_emb_size=gather_width,
            mlp1_hidden_dim=mlp1_hidden_dim,
            mlp1_depth=mlp1_depth,
            mlp1_dropout_p=mlp1_dropout_p,
            mlp2_hidden_dim=mlp2_hidden_dim,
            mlp2_depth=mlp2_depth,
            mlp2_dropout_p=mlp2_dropout_p,
            init=initialization,
            f_add_elems=f_add_elems,
            f_conn_elems=edge_features,
            f_term_elems=1,
            max_n_nodes=n_nodes_largest_graph,
        )

    def preprocess_edges(self, nodes, node_neighbours, edges):
        cat = torch.cat((nodes, node_neighbours, edges), dim=1)

        return torch.tanh(self.embedding_nn(cat))

    def propagate_edges(self, edges, ingoing_edge_memories, ingoing_edges_mask):
        Softmax = torch.nn.Softmax(dim=1)

        energy_mask = ((1 - ingoing_edges_mask).float() * BIG_NEGATIVE).unsqueeze(-1)

        cat = torch.cat((edges.unsqueeze(1), ingoing_edge_memories), dim=1)
        embeddings = self.emb_msg_nn(cat)

        edge_energy = self.att_msg_nn(edges)
        ing_memory_energies = self.att_msg_nn(ingoing_edge_memories) + energy_mask
        energies = torch.cat((edge_energy.unsqueeze(1), ing_memory_energies), dim=1)
        attention = Softmax(energies)

        # set aggregation of set of given edge feature and ingoing edge memories
        message = (attention * embeddings).sum(dim=1)

        return self.gru(message)  # return hidden state

    def readout(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        output = self.APDReadout(hidden_nodes, graph_embeddings)

        return output
