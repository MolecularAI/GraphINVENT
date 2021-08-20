"""
Defines specific MPNN implementations.
"""
# load general packages and functions
from collections import namedtuple
import math
import torch

# load GraphINVENT-specific functions
import gnn.aggregation_mpnn
import gnn.edge_mpnn
import gnn.summation_mpnn
import gnn.modules


class MNN(gnn.summation_mpnn.SummationMPNN):
    """
    The "message neural network" model.
    """
    def __init__(self, constants : namedtuple) -> None:
        super().__init__(constants)

        self.constants       = constants
        message_weights      = torch.Tensor(self.constants.message_size,
                                            self.constants.hidden_node_features,
                                            self.constants.n_edge_features)
        if self.constants.device == "cuda":
            message_weights = message_weights.to("cuda", non_blocking=True)

        self.message_weights = torch.nn.Parameter(message_weights)

        self.gru             = torch.nn.GRUCell(
            input_size=self.constants.message_size,
            hidden_size=self.constants.hidden_node_features,
            bias=True
        )

        self.APDReadout      = gnn.modules.GlobalReadout(
            node_emb_size=self.constants.hidden_node_features,
            graph_emb_size=self.constants.hidden_node_features,
            mlp1_hidden_dim=self.constants.mlp1_hidden_dim,
            mlp1_depth=self.constants.mlp1_depth,
            mlp1_dropout_p=self.constants.mlp1_dropout_p,
            mlp2_hidden_dim=self.constants.mlp2_hidden_dim,
            mlp2_depth=self.constants.mlp2_depth,
            mlp2_dropout_p=self.constants.mlp2_dropout_p,
            f_add_elems=self.constants.len_f_add_per_node,
            f_conn_elems=self.constants.len_f_conn_per_node,
            f_term_elems=1,
            max_n_nodes=self.constants.max_n_nodes,
            device=self.constants.device,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdev = 1.0 / math.sqrt(self.message_weights.size(1))
        self.message_weights.data.uniform_(-stdev, stdev)

    def message_terms(self, nodes : torch.Tensor, node_neighbours : torch.Tensor,
                      edges : torch.Tensor) -> torch.Tensor:
        edges_view            = edges.view(-1, 1, 1, self.constants.n_edge_features)
        weights_for_each_edge = (edges_view * self.message_weights.unsqueeze(0)).sum(3)
        return torch.matmul(weights_for_each_edge,
                            node_neighbours.unsqueeze(-1)).squeeze()

    def update(self, nodes : torch.Tensor, messages : torch.Tensor) -> torch.Tensor:
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes : torch.Tensor, input_nodes : torch.Tensor,
                node_mask : torch.Tensor) -> torch.Tensor:
        graph_embeddings = torch.sum(hidden_nodes, dim=1)
        output           = self.APDReadout(hidden_nodes, graph_embeddings)
        return output


class S2V(gnn.summation_mpnn.SummationMPNN):
    """
    The "set2vec" model.
    """
    def __init__(self, constants : namedtuple) -> None:
        super().__init__(constants)

        self.constants  = constants

        self.enn        = gnn.modules.MLP(
            in_features=self.constants.n_edge_features,
            hidden_layer_sizes=[self.constants.enn_hidden_dim] * self.constants.enn_depth,
            out_features=self.constants.hidden_node_features * self.constants.message_size,
            dropout_p=self.constants.enn_dropout_p
        )

        self.gru        = torch.nn.GRUCell(
            input_size=self.constants.message_size,
            hidden_size=self.constants.hidden_node_features,
            bias=True
        )

        self.s2v        = gnn.modules.Set2Vec(
            node_features=self.constants.n_node_features,
            hidden_node_features=self.constants.hidden_node_features,
            lstm_computations=self.constants.s2v_lstm_computations,
            memory_size=self.constants.s2v_memory_size
        )

        self.APDReadout = gnn.modules.GlobalReadout(
            node_emb_size=self.constants.hidden_node_features,
            graph_emb_size=self.constants.s2v_memory_size * 2,
            mlp1_hidden_dim=self.constants.mlp1_hidden_dim,
            mlp1_depth=self.constants.mlp1_depth,
            mlp1_dropout_p=self.constants.mlp1_dropout_p,
            mlp2_hidden_dim=self.constants.mlp2_hidden_dim,
            mlp2_depth=self.constants.mlp2_depth,
            mlp2_dropout_p=self.constants.mlp2_dropout_p,
            f_add_elems=self.constants.len_f_add_per_node,
            f_conn_elems=self.constants.len_f_conn_per_node,
            f_term_elems=1,
            max_n_nodes=self.constants.max_n_nodes,
            device=self.constants.device,
        )

    def message_terms(self, nodes : torch.Tensor, node_neighbours : torch.Tensor,
                      edges : torch.Tensor) -> torch.Tensor:
        enn_output = self.enn(edges)
        matrices   = enn_output.view(-1,
                                     self.constants.message_size,
                                     self.constants.hidden_node_features)
        msg_terms  = torch.matmul(matrices,
                                  node_neighbours.unsqueeze(-1)).squeeze(-1)
        return msg_terms

    def update(self, nodes : torch.Tensor, messages : torch.Tensor) -> torch.Tensor:
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes : torch.Tensor, input_nodes : torch.Tensor,
                node_mask : torch.Tensor) -> torch.Tensor:
        graph_embeddings = self.s2v(hidden_nodes, input_nodes, node_mask)
        output           = self.APDReadout(hidden_nodes, graph_embeddings)
        return output


class AttentionS2V(gnn.aggregation_mpnn.AggregationMPNN):
    """
    The "set2vec with attention" model.
    """
    def __init__(self, constants : namedtuple) -> None:

        super().__init__(constants)

        self.constants  = constants

        self.enn        = gnn.modules.MLP(
            in_features=self.constants.n_edge_features,
            hidden_layer_sizes=[self.constants.enn_hidden_dim] * self.constants.enn_depth,
            out_features=self.constants.hidden_node_features * self.constants.message_size,
            dropout_p=self.constants.enn_dropout_p
        )

        self.att_enn    = gnn.modules.MLP(
            in_features=self.constants.hidden_node_features + self.constants.n_edge_features,
            hidden_layer_sizes=[self.constants.att_hidden_dim] * self.constants.att_depth,
            out_features=self.constants.message_size,
            dropout_p=self.constants.att_dropout_p
        )

        self.gru        = torch.nn.GRUCell(
            input_size=self.constants.message_size,
            hidden_size=self.constants.hidden_node_features,
            bias=True
        )

        self.s2v        = gnn.modules.Set2Vec(
            node_features=self.constants.n_node_features,
            hidden_node_features=self.constants.hidden_node_features,
            lstm_computations=self.constants.s2v_lstm_computations,
            memory_size=self.constants.s2v_memory_size,
        )

        self.APDReadout = gnn.modules.GlobalReadout(
            node_emb_size=self.constants.hidden_node_features,
            graph_emb_size=self.constants.s2v_memory_size * 2,
            mlp1_hidden_dim=self.constants.mlp1_hidden_dim,
            mlp1_depth=self.constants.mlp1_depth,
            mlp1_dropout_p=self.constants.mlp1_dropout_p,
            mlp2_hidden_dim=self.constants.mlp2_hidden_dim,
            mlp2_depth=self.constants.mlp2_depth,
            mlp2_dropout_p=self.constants.mlp2_dropout_p,
            f_add_elems=self.constants.len_f_add_per_node,
            f_conn_elems=self.constants.len_f_conn_per_node,
            f_term_elems=1,
            max_n_nodes=self.constants.max_n_nodes,
            device=self.constants.device,
        )

    def aggregate_message(self, nodes : torch.Tensor,
                          node_neighbours : torch.Tensor,
                          edges : torch.Tensor,
                          mask : torch.Tensor) -> torch.Tensor:
        Softmax         = torch.nn.Softmax(dim=1)
        max_node_degree = node_neighbours.shape[1]

        enn_output      = self.enn(edges)
        matrices        = enn_output.view(-1,
                                          max_node_degree,
                                          self.constants.message_size,
                                          self.constants.hidden_node_features)
        message_terms   = torch.matmul(matrices, node_neighbours.unsqueeze(-1)).squeeze()

        att_enn_output  = self.att_enn(torch.cat((edges, node_neighbours), dim=2))
        energies        = att_enn_output.view(-1, max_node_degree, self.constants.message_size)
        energy_mask     = (1 - mask).float() * self.constants.big_negative
        weights         = Softmax(energies + energy_mask.unsqueeze(-1))

        return (weights * message_terms).sum(1)

    def update(self, nodes : torch.Tensor, messages : torch.Tensor) -> torch.Tensor:
        if self.constants.device == "cuda":
            messages = messages + torch.zeros(self.constants.message_size, device="cuda")
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes : torch.Tensor,
                input_nodes : torch.Tensor,
                node_mask : torch.Tensor) -> torch.Tensor:
        graph_embeddings = self.s2v(hidden_nodes, input_nodes, node_mask)
        output           = self.APDReadout(hidden_nodes, graph_embeddings)
        return output


class GGNN(gnn.summation_mpnn.SummationMPNN):
    """
    The "gated-graph neural network" model.
    """
    def __init__(self, constants : namedtuple) -> None:
        super().__init__(constants)

        self.constants  = constants

        self.msg_nns    = torch.nn.ModuleList()
        for _ in range(self.constants.n_edge_features):
            self.msg_nns.append(
                gnn.modules.MLP(
                    in_features=self.constants.hidden_node_features,
                    hidden_layer_sizes=[self.constants.enn_hidden_dim] * self.constants.enn_depth,
                    out_features=self.constants.message_size,
                    dropout_p=self.constants.enn_dropout_p,
                )
            )

        self.gru        = torch.nn.GRUCell(
            input_size=self.constants.message_size,
            hidden_size=self.constants.hidden_node_features,
            bias=True
        )

        self.gather     = gnn.modules.GraphGather(
            node_features=self.constants.n_node_features,
            hidden_node_features=self.constants.hidden_node_features,
            out_features=self.constants.gather_width,
            att_depth=self.constants.gather_att_depth,
            att_hidden_dim=self.constants.gather_att_hidden_dim,
            att_dropout_p=self.constants.gather_att_dropout_p,
            emb_depth=self.constants.gather_emb_depth,
            emb_hidden_dim=self.constants.gather_emb_hidden_dim,
            emb_dropout_p=self.constants.gather_emb_dropout_p,
            big_positive=self.constants.big_positive
        )

        self.APDReadout = gnn.modules.GlobalReadout(
            node_emb_size=self.constants.hidden_node_features,
            graph_emb_size=self.constants.gather_width,
            mlp1_hidden_dim=self.constants.mlp1_hidden_dim,
            mlp1_depth=self.constants.mlp1_depth,
            mlp1_dropout_p=self.constants.mlp1_dropout_p,
            mlp2_hidden_dim=self.constants.mlp2_hidden_dim,
            mlp2_depth=self.constants.mlp2_depth,
            mlp2_dropout_p=self.constants.mlp2_dropout_p,
            f_add_elems=self.constants.len_f_add_per_node,
            f_conn_elems=self.constants.len_f_conn_per_node,
            f_term_elems=1,
            max_n_nodes=self.constants.max_n_nodes,
            device=self.constants.device,
        )

    def message_terms(self, nodes : torch.Tensor, node_neighbours : torch.Tensor,
                      edges : torch.Tensor) -> torch.Tensor:
        edges_v               = edges.view(-1, self.constants.n_edge_features, 1)
        node_neighbours_v     = edges_v * node_neighbours.view(-1,
                                                               1,
                                                               self.constants.hidden_node_features)
        terms_masked_per_edge = [
            edges_v[:, i, :] * self.msg_nns[i](node_neighbours_v[:, i, :])
            for i in range(self.constants.n_edge_features)
        ]
        return sum(terms_masked_per_edge)

    def update(self, nodes : torch.Tensor, messages : torch.Tensor) -> torch.Tensor:
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes : torch.Tensor, input_nodes : torch.Tensor,
                node_mask : torch.Tensor) -> torch.Tensor:
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        output           = self.APDReadout(hidden_nodes, graph_embeddings)
        return output


class AttentionGGNN(gnn.aggregation_mpnn.AggregationMPNN):
    """
    The "GGNN with attention" model.
    """
    def __init__(self, constants : namedtuple) -> None:
        super().__init__(constants)

        self.constants = constants
        self.msg_nns   = torch.nn.ModuleList()
        self.att_nns   = torch.nn.ModuleList()

        for _ in range(self.constants.n_edge_features):
            self.msg_nns.append(
                gnn.modules.MLP(
                  in_features=self.constants.hidden_node_features,
                  hidden_layer_sizes=[self.constants.msg_hidden_dim] * self.constants.msg_depth,
                  out_features=self.constants.message_size,
                  dropout_p=self.constants.msg_dropout_p,
                )
            )
            self.att_nns.append(
                gnn.modules.MLP(
                  in_features=self.constants.hidden_node_features,
                  hidden_layer_sizes=[self.constants.att_hidden_dim] * self.constants.att_depth,
                  out_features=self.constants.message_size,
                  dropout_p=self.constants.att_dropout_p,
                )
            )

        self.gru = torch.nn.GRUCell(
            input_size=self.constants.message_size,
            hidden_size=self.constants.hidden_node_features,
            bias=True
        )

        self.gather = gnn.modules.GraphGather(
            node_features=self.constants.n_node_features,
            hidden_node_features=self.constants.hidden_node_features,
            out_features=self.constants.gather_width,
            att_depth=self.constants.gather_att_depth,
            att_hidden_dim=self.constants.gather_att_hidden_dim,
            att_dropout_p=self.constants.gather_att_dropout_p,
            emb_depth=self.constants.gather_emb_depth,
            emb_hidden_dim=self.constants.gather_emb_hidden_dim,
            emb_dropout_p=self.constants.gather_emb_dropout_p,
            big_positive=self.constants.big_positive
        )

        self.APDReadout = gnn.modules.GlobalReadout(
            node_emb_size=self.constants.hidden_node_features,
            graph_emb_size=self.constants.gather_width,
            mlp1_hidden_dim=self.constants.mlp1_hidden_dim,
            mlp1_depth=self.constants.mlp1_depth,
            mlp1_dropout_p=self.constants.mlp1_dropout_p,
            mlp2_hidden_dim=self.constants.mlp2_hidden_dim,
            mlp2_depth=self.constants.mlp2_depth,
            mlp2_dropout_p=self.constants.mlp2_dropout_p,
            f_add_elems=self.constants.len_f_add_per_node,
            f_conn_elems=self.constants.len_f_conn_per_node,
            f_term_elems=1,
            max_n_nodes=self.constants.max_n_nodes,
            device=self.constants.device,
        )

    def aggregate_message(self, nodes : torch.Tensor, node_neighbours : torch.Tensor,
                          edges : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
        Softmax = torch.nn.Softmax(dim=1)

        energy_mask = (mask == 0).float() * self.constants.big_positive

        embeddings_masked_per_edge = [
            edges[:, :, i].unsqueeze(-1) * self.msg_nns[i](node_neighbours)
            for i in range(self.constants.n_edge_features)
        ]
        energies_masked_per_edge = [
            edges[:, :, i].unsqueeze(-1) * self.att_nns[i](node_neighbours)
            for i in range(self.constants.n_edge_features)
        ]

        embedding   = sum(embeddings_masked_per_edge)
        energies    = sum(energies_masked_per_edge) - energy_mask.unsqueeze(-1)
        attention   = Softmax(energies)

        return torch.sum(attention * embedding, dim=1)

    def update(self, nodes : torch.Tensor, messages : torch.Tensor) -> torch.Tensor:
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes : torch.Tensor, input_nodes : torch.Tensor,
                node_mask : torch.Tensor) -> torch.Tensor:
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        output           = self.APDReadout(hidden_nodes, graph_embeddings)
        return output


class EMN(gnn.edge_mpnn.EdgeMPNN):
    """
    The "edge memory network" model.
    """
    def __init__(self, constants : namedtuple) -> None:
        super().__init__(constants)

        self.constants    = constants

        self.embedding_nn = gnn.modules.MLP(
            in_features=self.constants.n_node_features * 2 + self.constants.n_edge_features,
            hidden_layer_sizes=[self.constants.edge_emb_hidden_dim] *self.constants.edge_emb_depth,
            out_features=self.constants.edge_emb_size,
            dropout_p=self.constants.edge_emb_dropout_p,
        )

        self.emb_msg_nn   = gnn.modules.MLP(
            in_features=self.constants.edge_emb_size,
            hidden_layer_sizes=[self.constants.msg_hidden_dim] * self.constants.msg_depth,
            out_features=self.constants.edge_emb_size,
            dropout_p=self.constants.msg_dropout_p,
        )

        self.att_msg_nn   = gnn.modules.MLP(
            in_features=self.constants.edge_emb_size,
            hidden_layer_sizes=[self.constants.att_hidden_dim] * self.constants.att_depth,
            out_features=self.constants.edge_emb_size,
            dropout_p=self.constants.att_dropout_p,
        )

        self.gru          = torch.nn.GRUCell(
            input_size=self.constants.edge_emb_size,
            hidden_size=self.constants.edge_emb_size,
            bias=True
        )

        self.gather       = gnn.modules.GraphGather(
            node_features=self.constants.edge_emb_size,
            hidden_node_features=self.constants.edge_emb_size,
            out_features=self.constants.gather_width,
            att_depth=self.constants.gather_att_depth,
            att_hidden_dim=self.constants.gather_att_hidden_dim,
            att_dropout_p=self.constants.gather_att_dropout_p,
            emb_depth=self.constants.gather_emb_depth,
            emb_hidden_dim=self.constants.gather_emb_hidden_dim,
            emb_dropout_p=self.constants.gather_emb_dropout_p,
            big_positive=self.constants.big_positive
        )

        self.APDReadout   = gnn.modules.GlobalReadout(
            node_emb_size=self.constants.edge_emb_size,
            graph_emb_size=self.constants.gather_width,
            mlp1_hidden_dim=self.constants.mlp1_hidden_dim,
            mlp1_depth=self.constants.mlp1_depth,
            mlp1_dropout_p=self.constants.mlp1_dropout_p,
            mlp2_hidden_dim=self.constants.mlp2_hidden_dim,
            mlp2_depth=self.constants.mlp2_depth,
            mlp2_dropout_p=self.constants.mlp2_dropout_p,
            f_add_elems=self.constants.len_f_add_per_node,
            f_conn_elems=self.constants.len_f_conn_per_node,
            f_term_elems=1,
            max_n_nodes=self.constants.max_n_nodes,
            device=self.constants.device,
        )

    def preprocess_edges(self, nodes : torch.Tensor, node_neighbours : torch.Tensor,
                         edges : torch.Tensor) -> torch.Tensor:
        cat = torch.cat((nodes, node_neighbours, edges), dim=1)
        return torch.tanh(self.embedding_nn(cat))

    def propagate_edges(self, edges : torch.Tensor, ingoing_edge_memories : torch.Tensor,
                        ingoing_edges_mask : torch.Tensor) -> torch.Tensor:
        Softmax             = torch.nn.Softmax(dim=1)

        energy_mask         = (
            (1 - ingoing_edges_mask).float() * self.constants.big_negative
        ).unsqueeze(-1)
        cat                 = torch.cat((edges.unsqueeze(1), ingoing_edge_memories), dim=1)
        embeddings          = self.emb_msg_nn(cat)
        edge_energy         = self.att_msg_nn(edges)
        ing_memory_energies = self.att_msg_nn(ingoing_edge_memories) + energy_mask
        energies            = torch.cat((edge_energy.unsqueeze(1), ing_memory_energies), dim=1)
        attention           = Softmax(energies)

        # set aggregation of set of given edge feature and ingoing edge memories
        message = (attention * embeddings).sum(dim=1)

        return self.gru(message)  # return hidden state

    def readout(self, hidden_nodes : torch.Tensor, input_nodes : torch.Tensor,
                node_mask : torch.Tensor) -> torch.Tensor:
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        output           = self.APDReadout(hidden_nodes, graph_embeddings)
        return output
