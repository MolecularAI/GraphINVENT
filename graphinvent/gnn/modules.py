# load general packages and functions
import torch

# load program-specific functions
# (none)

# defines MPNN modules and readout functions, and APD readout functions

# define "BIG" constants (for masks)
BIG_NEGATIVE = -1e6
BIG_POSITIVE = 1e6


class GraphGather(torch.nn.Module):
    """ GGNN readout function.
    """
    def __init__(self, node_features, hidden_node_features, out_features,
                 att_depth, att_hidden_dim, att_dropout_p, emb_depth,
                 emb_hidden_dim, emb_dropout_p, init):

        super(GraphGather, self).__init__()

        self.att_nn = MLP(
            in_features=node_features + hidden_node_features,
            hidden_layer_sizes=[att_hidden_dim] * att_depth,
            out_features=out_features,
            init=init,
            dropout_p=att_dropout_p
        )

        self.emb_nn = MLP(
            in_features=hidden_node_features,
            hidden_layer_sizes=[emb_hidden_dim] * emb_depth,
            out_features=out_features,
            init=init,
            dropout_p=emb_dropout_p
        )

    def forward(self, hidden_nodes, input_nodes, node_mask):
        """ Defines forward pass.
        """
        Softmax = torch.nn.Softmax(dim=1)

        cat = torch.cat((hidden_nodes, input_nodes), dim=2)
        energy_mask = (node_mask == 0).float() * BIG_POSITIVE
        energies = self.att_nn(cat) - energy_mask.unsqueeze(-1)
        attention = Softmax(energies)
        embedding = self.emb_nn(hidden_nodes)

        return torch.sum(attention * embedding, dim=1)


class Set2Vec(torch.nn.Module):
    """ S2V readout function.
    """
    def __init__(self, node_features, hidden_node_features, lstm_computations, memory_size):

        super(Set2Vec, self).__init__()

        self.lstm_computations = lstm_computations

        self.memory_size = memory_size

        self.embedding_matrix = torch.nn.Linear(
            in_features=node_features + hidden_node_features,
            out_features=self.memory_size,
            bias=True
        )

        self.lstm = torch.nn.LSTMCell(input_size=self.memory_size,
                                      hidden_size=self.memory_size,
                                      bias=True)

    def forward(self, hidden_output_nodes, input_nodes, node_mask):
        """ Defines forward pass.
        """
        Softmax = torch.nn.Softmax(dim=1)

        batch_size = input_nodes.shape[0]
        energy_mask = torch.bitwise_not(node_mask).float() * BIG_NEGATIVE

        lstm_input = torch.zeros(batch_size, self.memory_size, device="cuda")

        cat = torch.cat((hidden_output_nodes, input_nodes), dim=2)
        memory = self.embedding_matrix(cat)

        hidden_state = torch.zeros(batch_size, self.memory_size, device="cuda")
        cell_state = torch.zeros(batch_size, self.memory_size, device="cuda")

        for _ in range(self.lstm_computations):
            query, cell_state = self.lstm(lstm_input, (hidden_state, cell_state))

            # dot product query x memory
            energies = (query.view(batch_size, 1, self.memory_size) * memory).sum(dim=-1)
            attention = Softmax(energies + energy_mask)
            read = (attention.unsqueeze(-1) * memory).sum(dim=1)

            hidden_state = query
            lstm_input = read

        cat = torch.cat((query, read), dim=1)
        return cat


class MLP(torch.nn.Module):
    """ Multi-layer perceptron. Applies SELU after every linear layer.

    Args:
      in_features (int) : Size of each input sample.
      hidden_layer_sizes (list) : Hidden layer sizes.
      out_features (int) : Size of each output sample.
      init (str) : Weight initialization ('none', 'normal', or 'uniform').
      dropout_p (float) : Probability of dropping a weight.
    """

    def __init__(self, in_features, hidden_layer_sizes, out_features, init, dropout_p):

        super(MLP, self).__init__()

        activation_function = torch.nn.SELU

        # create list of all layer feature sizes
        fs = [in_features, *hidden_layer_sizes, out_features]

        # create list of linear_blocks
        layers = [self._linear_block(in_f, out_f,
                                     activation_function, init,
                                     dropout_p)
                  for in_f, out_f in zip(fs, fs[1:])]

        # concatenate modules in all sequentials in layers list
        layers = [module for sq in layers for module in sq.children()]

        # add modules to sequential container
        self.seq = torch.nn.Sequential(*layers)

    def _linear_block(self, in_f, out_f, activation, init, dropout_p):
        """ Returns a linear block consisting of a linear layer, an activation
        function (SELU), and dropout (optional) stack.

        Args:
          in_f (int) : Size of each input sample.
          out_f (int) : Size of each output sample.
          activation (torch.nn.Module) : Activation function.
          init (str) : Weight initialization ('none', 'normal', or 'uniform').
          dropout_p (float) : Probability of dropping a weight.
        """
        # bias must be used in most MLPs in our models to learn from empty graphs
        linear = torch.nn.Linear(in_f, out_f, bias=True)
        linear.weight = self.define_weight_initialization(init, linear)

        return torch.nn.Sequential(linear, activation(), torch.nn.AlphaDropout(dropout_p))

    @staticmethod
    def define_weight_initialization(initialization, linear_layer):
        """ Defines the weight initialization scheme to use in `linear_layer`.
        """
        if initialization == "none":
            pass
        elif initialization == "uniform":
            torch.nn.init.xavier_uniform_(linear_layer.weight)
        elif initialization == "normal":
            torch.nn.init.xavier_normal_(linear_layer.weight)
        else:
            raise NotImplementedError

        return linear_layer.weight

    def forward(self, layers_input):
        """ Defines forward pass.
        """
        return self.seq(layers_input)


class GlobalReadout(torch.nn.Module):
    """ Global readout function class. Used to predict the action probability
    distributions (APDs) for molecular graphs.

    The first tier of two `MLP`s take as input, for each graph in the batch,
    the final transformed node feature vectors. These feed-forward networks
    correspond to the preliminary "f_add" and "f_conn" distributions.

    The second tier of three `MLP`s takes as input the output of the first tier
    of `MLP`s (the "preliminary" APDs) as well as the graph embeddings for all
    graphs in the batch. Output are the final APD components, which are then
    flattened and concatenated. No activation function is applied after the
    final layer, so that this can be done outside (e.g. in the loss function,
    and before sampling).
    """
    def __init__(self, f_add_elems, f_conn_elems, f_term_elems, mlp1_depth,
                 mlp1_dropout_p, mlp1_hidden_dim, mlp2_depth, mlp2_dropout_p,
                 mlp2_hidden_dim, graph_emb_size, init, max_n_nodes, node_emb_size):

        super(GlobalReadout, self).__init__()

        # preliminary f_add
        self.fAddNet1 = MLP(
            in_features=node_emb_size,
            hidden_layer_sizes=[mlp1_hidden_dim] * mlp1_depth,
            out_features=f_add_elems,
            init=init,
            dropout_p=mlp1_dropout_p
        )

        # preliminary f_conn
        self.fConnNet1 = MLP(
            in_features=node_emb_size,
            hidden_layer_sizes=[mlp1_hidden_dim] * mlp1_depth,
            out_features=f_conn_elems,
            init=init,
            dropout_p=mlp1_dropout_p
        )

        # final f_add
        self.fAddNet2 = MLP(
            in_features=(max_n_nodes * f_add_elems + graph_emb_size),
            hidden_layer_sizes=[mlp2_hidden_dim] * mlp2_depth,
            out_features=f_add_elems * max_n_nodes,
            init=init,
            dropout_p=mlp2_dropout_p
        )

        # final f_conn
        self.fConnNet2 = MLP(
            in_features=(max_n_nodes * f_conn_elems + graph_emb_size),
            hidden_layer_sizes=[mlp2_hidden_dim] * mlp2_depth,
            out_features=f_conn_elems * max_n_nodes,
            init=init,
            dropout_p=mlp2_dropout_p
        )

        # final f_term (only takes as input graph embeddings)
        self.fTermNet2 = MLP(
            in_features=graph_emb_size,
            hidden_layer_sizes=[mlp2_hidden_dim] * mlp2_depth,
            out_features=f_term_elems,
            init=init,
            dropout_p=mlp2_dropout_p
        )

    def forward(self, node_level_output, graph_embedding_batch):
        """ Defines forward pass.
        """
        self.fAddNet1 = self.fAddNet1.to("cuda", non_blocking=True)
        self.fConnNet1 = self.fConnNet1.to("cuda", non_blocking=True)
        self.fAddNet2 = self.fAddNet2.to("cuda", non_blocking=True)
        self.fConnNet2 = self.fConnNet2.to("cuda", non_blocking=True)
        self.fTermNet2 = self.fTermNet2.to("cuda", non_blocking=True)

        # get preliminary f_add and f_conn
        f_add_1 = self.fAddNet1(node_level_output)
        f_conn_1 = self.fConnNet1(node_level_output)

        f_add_1 = f_add_1.to("cuda", non_blocking=True)
        f_conn_1 = f_conn_1.to("cuda", non_blocking=True)

        # reshape preliminary APDs into flattenened vectors (e.g. one vector per
        # graph in batch)
        f_add_1_size = f_add_1.size()
        f_conn_1_size = f_conn_1.size()
        f_add_1 = f_add_1.view((f_add_1_size[0], f_add_1_size[1] * f_add_1_size[2]))
        f_conn_1 = f_conn_1.view((f_conn_1_size[0], f_conn_1_size[1] * f_conn_1_size[2]))

        # get final f_add, f_conn, and f_term
        f_add_2 = self.fAddNet2(torch.cat((f_add_1, graph_embedding_batch), dim=1).unsqueeze(dim=1))
        f_conn_2 = self.fConnNet2(torch.cat((f_conn_1, graph_embedding_batch), dim=1).unsqueeze(dim=1))
        f_term_2 = self.fTermNet2(graph_embedding_batch)

        f_add_2 = f_add_2.to("cuda", non_blocking=True)
        f_conn_2 = f_conn_2.to("cuda", non_blocking=True)
        f_term_2 = f_term_2.to("cuda", non_blocking=True)

        # flatten and concatenate
        cat = torch.cat((f_add_2.squeeze(dim=1), f_conn_2.squeeze(dim=1), f_term_2), dim=1)

        return cat  # note: no activation function before returning
