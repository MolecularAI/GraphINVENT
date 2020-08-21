# load general packages and functions
# (none)

# load program-specific functions
from parameters.constants import constants as C
import gnn.mpnn
import util

# defines the models with parameters from `constants.py`


def initialize_model():
    """
    Initializes the model to be trained. Possible models: "MNN", "S2V",
    "AttS2V", "GGNN", "AttGGNN", or "EMN".

    Returns:
      model (modules.SummationMPNN or modules.AggregationMPNN or
        modules.EdgeMPNN) : Neural net model.
    """
    try: 
        hidden_node_features = C.hidden_node_features
    except AttributeError:  # raised for EMN model only
        hidden_node_features = None
        edge_emb_hidden_dim = C.edge_emb_hidden_dim

    if C.model == "MNN":
        net = gnn.mpnn.MNN(
            f_add_elems=C.dim_f_add_p1,
            edge_features=C.dim_edges[2],
            mlp1_depth=C.mlp1_depth,
            mlp1_dropout_p=C.mlp1_dropout_p,
            mlp1_hidden_dim=C.mlp1_hidden_dim,
            mlp2_depth=C.mlp2_depth,
            mlp2_dropout_p=C.mlp2_dropout_p,
            mlp2_hidden_dim=C.mlp2_hidden_dim,
            hidden_node_features=hidden_node_features,
            initialization=C.weights_initialization,
            message_passes=C.message_passes,
            message_size=C.message_size,
            n_nodes_largest_graph=C.max_n_nodes,
            node_features=C.dim_nodes[1],
        )
    elif C.model == "S2V":
        net = gnn.mpnn.S2V(
            f_add_elems=C.dim_f_add_p1,
            edge_features=C.dim_edges[2],
            enn_depth=C.enn_depth,
            enn_dropout_p=C.enn_dropout_p,
            enn_hidden_dim=C.enn_hidden_dim,
            mlp1_depth=C.mlp1_depth,
            mlp1_dropout_p=C.mlp1_dropout_p,
            mlp1_hidden_dim=C.mlp1_hidden_dim,
            mlp2_dropout_p=C.mlp2_dropout_p,
            mlp2_depth=C.mlp2_depth,
            mlp2_hidden_dim=C.mlp2_hidden_dim,
            hidden_node_features=hidden_node_features,
            initialization=C.weights_initialization,
            message_passes=C.message_passes,
            message_size=C.message_size,
            n_nodes_largest_graph=C.max_n_nodes,
            node_features=C.dim_nodes[1],
            s2v_lstm_computations=C.s2v_lstm_computations,
            s2v_memory_size=C.s2v_memory_size,
        )
    elif C.model == "AttS2V":
        net = gnn.mpnn.AttentionS2V(
            f_add_elems=C.dim_f_add_p1,
            att_depth=C.att_depth,
            att_dropout_p=C.att_dropout_p,
            att_hidden_dim=C.att_hidden_dim,
            edge_features=C.dim_edges[2],
            enn_depth=C.enn_depth,
            enn_dropout_p=C.enn_dropout_p,
            enn_hidden_dim=C.enn_hidden_dim,
            mlp1_depth=C.mlp1_depth,
            mlp1_dropout_p=C.mlp1_dropout_p,
            mlp1_hidden_dim=C.mlp1_hidden_dim,
            mlp2_depth=C.mlp2_depth,
            mlp2_dropout_p=C.mlp2_dropout_p,
            mlp2_hidden_dim=C.mlp2_hidden_dim,
            hidden_node_features=hidden_node_features,
            initialization=C.weights_initialization,
            message_passes=C.message_passes,
            message_size=C.message_size,
            n_nodes_largest_graph=C.max_n_nodes,
            node_features=C.dim_nodes[1],
            s2v_lstm_computations=C.s2v_lstm_computations,
            s2v_memory_size=C.s2v_memory_size,
        )
    elif C.model == "GGNN":
        net = gnn.mpnn.GGNN(
            f_add_elems=C.dim_f_add_p1,
            edge_features=C.dim_edges[2],
            enn_depth=C.enn_depth,
            enn_dropout_p=C.enn_dropout_p,
            enn_hidden_dim=C.enn_hidden_dim,
            mlp1_depth=C.mlp1_depth,
            mlp1_dropout_p=C.mlp1_dropout_p,
            mlp1_hidden_dim=C.mlp1_hidden_dim,
            mlp2_depth=C.mlp2_depth,
            mlp2_dropout_p=C.mlp2_dropout_p,
            mlp2_hidden_dim=C.mlp2_hidden_dim,
            gather_att_depth=C.gather_att_depth,
            gather_att_dropout_p=C.gather_att_dropout_p,
            gather_att_hidden_dim=C.gather_att_hidden_dim,
            gather_width=C.gather_width,
            gather_emb_depth=C.gather_emb_depth,
            gather_emb_dropout_p=C.gather_emb_dropout_p,
            gather_emb_hidden_dim=C.gather_emb_hidden_dim,
            hidden_node_features=hidden_node_features,
            initialization=C.weights_initialization,
            message_passes=C.message_passes,
            message_size=C.message_size,
            n_nodes_largest_graph=C.max_n_nodes,
            node_features=C.dim_nodes[1],
        )
    elif C.model == "AttGGNN":
        net = gnn.mpnn.AttentionGGNN(
            f_add_elems=C.dim_f_add_p1,
            att_depth=C.att_depth,
            att_dropout_p=C.att_dropout_p,
            att_hidden_dim=C.att_hidden_dim,
            edge_features=C.dim_edges[2],
            mlp1_depth=C.mlp1_depth,
            mlp1_dropout_p=C.mlp1_dropout_p,
            mlp1_hidden_dim=C.mlp1_hidden_dim,
            mlp2_depth=C.mlp2_depth,
            mlp2_dropout_p=C.mlp2_dropout_p,
            mlp2_hidden_dim=C.mlp2_hidden_dim,
            gather_att_depth=C.gather_att_depth,
            gather_att_dropout_p=C.gather_att_dropout_p,
            gather_att_hidden_dim=C.gather_att_hidden_dim,
            gather_emb_depth=C.gather_emb_depth,
            gather_emb_dropout_p=C.gather_emb_dropout_p,
            gather_emb_hidden_dim=C.gather_emb_hidden_dim,
            gather_width=C.gather_width,
            hidden_node_features=hidden_node_features,
            initialization=C.weights_initialization,
            message_passes=C.message_passes,
            message_size=C.message_size,
            msg_depth=C.msg_depth,
            msg_dropout_p=C.msg_dropout_p,
            msg_hidden_dim=C.msg_hidden_dim,
            n_nodes_largest_graph=C.max_n_nodes,
            node_features=C.dim_nodes[1],
        )
    elif C.model == "EMN":
        net = gnn.mpnn.EMN(
            f_add_elems=C.dim_f_add_p1,
            att_depth=C.att_depth,
            att_dropout_p=C.att_dropout_p,
            att_hidden_dim=C.att_hidden_dim,
            edge_emb_depth=C.edge_emb_depth,
            edge_emb_dropout_p=C.edge_emb_dropout_p,
            edge_emb_hidden_dim=edge_emb_hidden_dim,
            edge_emb_size=C.edge_emb_size,
            edge_features=C.dim_edges[2],
            mlp1_depth=C.mlp1_depth,
            mlp1_dropout_p=C.mlp1_dropout_p,
            mlp1_hidden_dim=C.mlp1_hidden_dim,
            mlp2_depth=C.mlp2_depth,
            mlp2_dropout_p=C.mlp2_dropout_p,
            mlp2_hidden_dim=C.mlp2_hidden_dim,
            gather_att_depth=C.gather_att_depth,
            gather_att_dropout_p=C.gather_att_dropout_p,
            gather_att_hidden_dim=C.gather_att_hidden_dim,
            gather_emb_depth=C.gather_emb_depth,
            gather_emb_dropout_p=C.gather_emb_dropout_p,
            gather_emb_hidden_dim=C.gather_emb_hidden_dim,
            gather_width=C.gather_width,
            initialization=C.weights_initialization,
            message_passes=C.message_passes,
            msg_depth=C.msg_depth,
            msg_dropout_p=C.msg_dropout_p,
            msg_hidden_dim=C.msg_hidden_dim,
            n_nodes_largest_graph=C.max_n_nodes,
            node_features=C.dim_nodes[1],
        )
    elif C.model == "RNN":
        raise NotImplementedError
    else:
        raise NotImplementedError("Model is not defined.")

    net = net.to("cuda", non_blocking=True)

    return net
