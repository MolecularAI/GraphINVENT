"""
Defines MPNN modules and readout functions, and APD readout functions.
"""
# load general packages and functions
from collections import namedtuple
import torch

# load GraphINVENT-specific functions
# (none)


class GraphGather(torch.nn.Module):
    """
    GGNN readout function.
    """
    def __init__(self, node_features : int, hidden_node_features : int,
                 out_features : int, att_depth : int, att_hidden_dim : int,
                 att_dropout_p : float, emb_depth : int, emb_hidden_dim : int,
                 emb_dropout_p : float, big_positive : float) -> None:

        super().__init__()

        self.big_positive = big_positive

        self.att_nn = MLP(
            in_features=node_features + hidden_node_features,
            hidden_layer_sizes=[att_hidden_dim] * att_depth,
            out_features=out_features,
            dropout_p=att_dropout_p
        )

        self.emb_nn = MLP(
            in_features=hidden_node_features,
            hidden_layer_sizes=[emb_hidden_dim] * emb_depth,
            out_features=out_features,
            dropout_p=emb_dropout_p
        )

    def forward(self, hidden_nodes : torch.Tensor, input_nodes : torch.Tensor,
                node_mask : torch.Tensor) -> torch.Tensor:
        """
        Defines forward pass.
        """
        Softmax     = torch.nn.Softmax(dim=1)

        cat         = torch.cat((hidden_nodes, input_nodes), dim=2)
        energy_mask = (node_mask == 0).float() * self.big_positive
        energies    = self.att_nn(cat) - energy_mask.unsqueeze(-1)
        attention   = Softmax(energies)
        embedding   = self.emb_nn(hidden_nodes)

        return torch.sum(attention * embedding, dim=1)


class Set2Vec(torch.nn.Module):
    """
    S2V readout function.
    """
    def __init__(self, node_features : int, hidden_node_features : int,
                 lstm_computations : int, memory_size : int,
                 constants : namedtuple) -> None:

        super().__init__()

        self.constants         = constants
        self.lstm_computations = lstm_computations
        self.memory_size       = memory_size

        self.embedding_matrix = torch.nn.Linear(
            in_features=node_features + hidden_node_features,
            out_features=self.memory_size,
            bias=True
        )

        self.lstm = torch.nn.LSTMCell(
            input_size=self.memory_size,
            hidden_size=self.memory_size,
            bias=True
        )

    def forward(self, hidden_output_nodes : torch.Tensor, input_nodes : torch.Tensor,
                node_mask : torch.Tensor) -> torch.Tensor:
        """
        Defines forward pass.
        """
        Softmax      = torch.nn.Softmax(dim=1)

        batch_size   = input_nodes.shape[0]
        energy_mask  = torch.bitwise_not(node_mask).float() * self.C.big_negative
        lstm_input   = torch.zeros(batch_size, self.memory_size, device=self.constants.device)
        cat          = torch.cat((hidden_output_nodes, input_nodes), dim=2)
        memory       = self.embedding_matrix(cat)
        hidden_state = torch.zeros(batch_size, self.memory_size, device=self.constants.device)
        cell_state   = torch.zeros(batch_size, self.memory_size, device=self.constants.device)

        for _ in range(self.lstm_computations):
            query, cell_state = self.lstm(lstm_input, (hidden_state, cell_state))

            # dot product query x memory
            energies  = (query.view(batch_size, 1, self.memory_size) * memory).sum(dim=-1)
            attention = Softmax(energies + energy_mask)
            read      = (attention.unsqueeze(-1) * memory).sum(dim=1)

            hidden_state = query
            lstm_input   = read

        cat = torch.cat((query, read), dim=1)
        return cat


class MLP(torch.nn.Module):
    """
    Multi-layer perceptron. Applies SELU after every linear layer.

    Args:
    ----
        in_features (int)         : Size of each input sample.
        hidden_layer_sizes (list) : Hidden layer sizes.
        out_features (int)        : Size of each output sample.
        dropout_p (float)         : Probability of dropping a weight.
    """

    def __init__(self, in_features : int, hidden_layer_sizes : list, out_features : int,
                 dropout_p : float) -> None:
        super().__init__()

        activation_function = torch.nn.SELU

        # create list of all layer feature sizes
        fs = [in_features, *hidden_layer_sizes, out_features]

        # create list of linear_blocks
        layers = [self._linear_block(in_f, out_f,
                                     activation_function,
                                     dropout_p)
                  for in_f, out_f in zip(fs, fs[1:])]

        # concatenate modules in all sequentials in layers list
        layers = [module for sq in layers for module in sq.children()]

        # add modules to sequential container
        self.seq = torch.nn.Sequential(*layers)

    def _linear_block(self, in_f : int, out_f : int, activation : torch.nn.Module,
                      dropout_p : float) -> torch.nn.Sequential:
        """
        Returns a linear block consisting of a linear layer, an activation function
        (SELU), and dropout (optional) stack.

        Args:
        ----
            in_f (int)                   : Size of each input sample.
            out_f (int)                  : Size of each output sample.
            activation (torch.nn.Module) : Activation function.
            dropout_p (float)            : Probability of dropping a weight.

        Returns:
        -------
            torch.nn.Sequential : The linear block.
        """
        # bias must be used in most MLPs in our models to learn from empty graphs
        linear = torch.nn.Linear(in_f, out_f, bias=True)
        torch.nn.init.xavier_uniform_(linear.weight)
        return torch.nn.Sequential(linear, activation(), torch.nn.AlphaDropout(dropout_p))

    def forward(self, layers_input : torch.nn.Sequential) -> torch.nn.Sequential:
        """
        Defines forward pass.
        """
        return self.seq(layers_input)


class GlobalReadout(torch.nn.Module):
    """
    Global readout function class. Used to predict the action probability distributions
    (APDs) for molecular graphs.

    The first tier of two `MLP`s take as input, for each graph in the batch, the
    final transformed node feature vectors. These feed-forward networks correspond
    to the preliminary "f_add" and "f_conn" distributions.

    The second tier of three `MLP`s takes as input the output of the first tier
    of `MLP`s (the "preliminary" APDs) as well as the graph embeddings for all
    graphs in the batch. Output are the final APD components, which are then flattened
    and concatenated. No activation function is applied after the final layer, so
    that this can be done outside (e.g. in the loss function, and before sampling).
    """
    def __init__(self, f_add_elems : int, f_conn_elems : int, f_term_elems : int,
                 mlp1_depth : int, mlp1_dropout_p : float, mlp1_hidden_dim : int,
                 mlp2_depth : int, mlp2_dropout_p : float, mlp2_hidden_dim : int,
                 graph_emb_size : int, max_n_nodes : int, node_emb_size : int,
                 device : str) -> None:
        super().__init__()

        self.device = device

        # preliminary f_add
        self.fAddNet1 = MLP(
            in_features=node_emb_size,
            hidden_layer_sizes=[mlp1_hidden_dim] * mlp1_depth,
            out_features=f_add_elems,
            dropout_p=mlp1_dropout_p
        )

        # preliminary f_conn
        self.fConnNet1 = MLP(
            in_features=node_emb_size,
            hidden_layer_sizes=[mlp1_hidden_dim] * mlp1_depth,
            out_features=f_conn_elems,
            dropout_p=mlp1_dropout_p
        )

        # final f_add
        self.fAddNet2 = MLP(
            in_features=(max_n_nodes * f_add_elems + graph_emb_size),
            hidden_layer_sizes=[mlp2_hidden_dim] * mlp2_depth,
            out_features=f_add_elems * max_n_nodes,
            dropout_p=mlp2_dropout_p
        )

        # final f_conn
        self.fConnNet2 = MLP(
            in_features=(max_n_nodes * f_conn_elems + graph_emb_size),
            hidden_layer_sizes=[mlp2_hidden_dim] * mlp2_depth,
            out_features=f_conn_elems * max_n_nodes,
            dropout_p=mlp2_dropout_p
        )

        # final f_term (only takes as input graph embeddings)
        self.fTermNet2 = MLP(
            in_features=graph_emb_size,
            hidden_layer_sizes=[mlp2_hidden_dim] * mlp2_depth,
            out_features=f_term_elems,
            dropout_p=mlp2_dropout_p
        )

    def forward(self, node_level_output : torch.Tensor,
                graph_embedding_batch : torch.Tensor) -> torch.Tensor:
        """
        Defines forward pass.
        """
        if self.device == "cuda":
            self.fAddNet1  = self.fAddNet1.to("cuda", non_blocking=True)
            self.fConnNet1 = self.fConnNet1.to("cuda", non_blocking=True)
            self.fAddNet2  = self.fAddNet2.to("cuda", non_blocking=True)
            self.fConnNet2 = self.fConnNet2.to("cuda", non_blocking=True)
            self.fTermNet2 = self.fTermNet2.to("cuda", non_blocking=True)

        # get preliminary f_add and f_conn
        f_add_1  = self.fAddNet1(node_level_output)
        f_conn_1 = self.fConnNet1(node_level_output)

        if self.device == "cuda":
            f_add_1  = f_add_1.to("cuda", non_blocking=True)
            f_conn_1 = f_conn_1.to("cuda", non_blocking=True)

        # reshape preliminary APDs into flattenened vectors (e.g. one vector per
        # graph in batch)
        f_add_1_size  = f_add_1.size()
        f_conn_1_size = f_conn_1.size()
        f_add_1  = f_add_1.view((f_add_1_size[0], f_add_1_size[1] * f_add_1_size[2]))
        f_conn_1 = f_conn_1.view((f_conn_1_size[0], f_conn_1_size[1] * f_conn_1_size[2]))

        # get final f_add, f_conn, and f_term
        f_add_2 = self.fAddNet2(
            torch.cat((f_add_1, graph_embedding_batch), dim=1).unsqueeze(dim=1)
        )
        f_conn_2 = self.fConnNet2(
            torch.cat((f_conn_1, graph_embedding_batch), dim=1).unsqueeze(dim=1)
        )
        f_term_2 = self.fTermNet2(graph_embedding_batch)

        if self.device == "cuda":
            f_add_2  = f_add_2.to("cuda", non_blocking=True)
            f_conn_2 = f_conn_2.to("cuda", non_blocking=True)
            f_term_2 = f_term_2.to("cuda", non_blocking=True)

        # flatten and concatenate
        cat = torch.cat((f_add_2.squeeze(dim=1), f_conn_2.squeeze(dim=1), f_term_2), dim=1)

        return cat  # note: no activation function before returning
