"""
Defines default model parameters, hyperparameters, and settings.
Recommended not to modify the default settings here, but rather create an input
file with the modified parameters in a new job directory (see README). **Used
as an alternative to using argparser, as there are many variables.**
"""
# load general packages and functions
import sys

# load GraphINVENT-specific functions
sys.path.insert(1, "./parameters/")  # search "parameters/" directory
import parameters.args as args
import parameters.load as load


# default parameters defined below
"""
General settings for the generative model:
    atom_types (list)         : Contains atom types (str) to encode in node features.
    formal_charge (list)      : Contains formal charges (int) to encode in node
                                features.
    imp_H (list)              : Contains number of implicit hydrogens (int) to encode
                                in node features.
    chirality (list)          : Contains chiral states (str) to encode in node features.
    device (str)              : Specifies type of architecture to run on. Options:
                                "cuda" or "cpu").
    generation_epoch (int)    : Epoch to sample during a 'generation' job.
    n_samples (int)           : Number of molecules to generate during each sampling
                                epoch. Note: if `n_samples` > 100000 molecules, these
                                will be generated in batches of 100000.
    n_workers (int)           : Number of subprocesses to use during data loading.
    restart (bool)            : If specified, will restart training from previous saved
                                state. Can only be used for preprocessing or training
                                jobs.
    max_n_nodes (int)         : Maximum number of allowed nodes in graph. Must be
                                greater than or equal to the number of nodes in
                                largest graph in training set.
    job_type (str)            : Type of job to run; options: 'preprocess', 'train',
                                'generate', 'test', or 'fine-tune'.
    sample_every (int)        : Specifies when to sample the model (i.e. epochs
                                between sampling).
    dataset_dir (str)         : Full path to directory containing testing ("test.smi"),
                                training ("train.smi"), and validation ("valid.smi")
                                sets.
    use_aromatic_bonds (bool) : If specified, aromatic bond types will be used.
    use_canon (bool)          : If specified, uses canonical RDKit ordering in graph
                                representations.
    use_chirality (bool)      : If specified, includes chirality in the atomic
                                representations.
    use_explicit_H (bool)     : If specified, uses explicit Hs in molecular
                                representations (not recommended for most applications).
    ignore_H (bool)           : If specified, ignores H's completely in graph
                                representations (treats them neither as explicit or
                                implicit). When generating graphs, H's are added to
                                graphs after generation is terminated.
    use_tensorboard (bool)    : If specified, enables the use of tensorboard during
                                training.
    tensorboard_dir (str)     : Path to directory in which to write tensorboard
                                things.
    batch_size (int)          : Number of graphs in a mini-batch. When preprocessing
                                graphs, this is the size of the preprocessing groups
                                (e.g. how many subgraphs preprocessed at once).
    epochs (int)              : Number of training epochs.
    init_lr (float)           : Initial learning rate.
    max_rel_lr (float)        : Maximum allowed learning rate relative to the initial
                                (used for learning rate ramp-up).
    model (str)               : MPNN model to use ('MNN', 'S2V', 'AttS2V', 'GGNN',
                                'AttGGNN', or 'EMN').
    decoding_route (str)      : Breadth-first search ("bfs") or depth-first search
                                ("dfs").
    score_components (list)   : A list of all the components to use in the RL scoring
                                function. Can include "target_size={int}", "QED",
                                "{name}_activity".
    score_thresholds (list)   : Acceptable thresholds for the above score components.
    score_type (str)          : If there are multiple components used in the scoring
                                function, determines if the final score should be
                                "continuous" (in which case, the above thresholds
                                are ignored), or "binary" (in which case a generated
                                molecule will receive a score of 1 iff all its score
                                components are greater than the specified thresholds).
    qsar_models (dict)        : A dictionary containing the path to each activity
                                model specified in `score_components`. Note that
                                the key in this dict must correspond to the name
                                of the score component.
    sigma (float)             : Can take any value. Tunes the contribution of the
                                score in the augmented log-likelihood. See Atance
                                et al (2021) https://doi.org/10.33774/chemrxiv-2021-9w3tc
                                for suitable values.
    alpha (float)             : Can take values between [0.0, 1.0]. Tunes the contribution
                                from the best agent so far (BASF) in the loss.
"""
# general job parameters
parameters = {
    "atom_types"          : ["C", "N", "O", "S", "Cl"],
    "formal_charge"       : [-1, 0, 1],
    "imp_H"               : [0, 1, 2, 3],
    "chirality"           : ["None", "R", "S"],
    "device"              : "cuda",
    "generation_epoch"    : 30,
    "n_samples"           : 2000,
    "n_workers"           : 2,
    "restart"             : False,
    "max_n_nodes"         : 13,
    "job_type"            : "train",
    "sample_every"        : 10,
    "dataset_dir"         : "data/gdb13_1K/",
    "use_aromatic_bonds"  : False,
    "use_canon"           : True,
    "use_chirality"       : False,
    "use_explicit_H"      : False,
    "ignore_H"            : True,
    "tensorboard_dir"     : "tensorboard/",
    "batch_size"          : 1000,
    "block_size"          : 100000,
    "epochs"              : 100,
    "init_lr"             : 1e-4,
    "max_rel_lr"          : 1,
    "min_rel_lr"          : 0.0001,
    "decoding_route"      : "bfs",
    "activity_model_dir"  : "data/fine-tuning/",
    "score_components"    : ["QED", "drd2_activity", "target_size=13"],
    "score_thresholds"    : [0.5, 0.5, 0.0],  # 0.0 essentially means no threshold
    "score_type"          : "binary",
    "qsar_models"         : {"drd2_activity": "data/fine-tuning/qsar_model.pickle"},
    "pretrained_model_dir": "output/",
    "sigma"               : 20,
    "alpha"               : 0.5,
}

# make sure job dir ends in "/"
if args.job_dir[-1] != "/":
    print("* Adding '/' to end of `job_dir`.")
    args.job_dir += "/"

# get the model before loading model-specific hyperparameters
try:
    input_csv_path  = args.job_dir + "input.csv"
    model           = load.which_model(input_csv_path=input_csv_path)
except:
    model           = "GGNN"  # default model
parameters["model"] = model


# model-specific hyperparameters (implementation-specific)
if parameters["model"] == "MNN":
    """
    MNN hyperparameters:
      mlp1_depth (int)       : Num layers in first-tier MLP in `APDReadout`.
      mlp1_dropout_p (float) : Dropout probability in first-tier MLP in `APDReadout`.
      mlp1_hidden_dim (int)  : Number of weights (layer width) in first-tier MLP
                               in `APDReadout`.
      mlp2_depth (int)       : Num layers in second-tier MLP in `APDReadout`.
      mlp2_dropout_p (float) : Dropout probability in second-tier MLP in `APDReadout`.
      mlp2_hidden_dim (int)  : Number of weights (layer width) in second-tier MLP
                               in `APDReadout`.
      message_passes (int)   : Number of message passing steps.
      message_size (int)     : Size of message passed ('enn' MLP output size).
    """
    hyperparameters = {
        "mlp1_depth"          : 4,
        "mlp1_dropout_p"      : 0.0,
        "mlp1_hidden_dim"     : 500,
        "mlp2_depth"          : 4,
        "mlp2_dropout_p"      : 0.0,
        "mlp2_hidden_dim"     : 500,
        "hidden_node_features": 100,
        "message_passes"      : 3,
        "message_size"        : 100,
    }
elif parameters["model"] == "S2V":
    """
    S2V hyperparameters:
      enn_depth (int)             : Num layers in 'enn' MLP.
      enn_dropout_p (float)       : Dropout probability in 'enn' MLP.
      enn_hidden_dim (int)        : Number of weights (layer width) in 'enn' MLP.
      mlp1_depth (int)            : Num layers in first-tier MLP in `APDReadout`.
      mlp1_dropout_p (float)      : Dropout probability in first-tier MLP in `APDReadout`.
      mlp1_hidden_dim (int)       : Number of weights (layer width) in first-tier
                                    MLP in `APDReadout`.
      mlp2_depth (int)            : Num layers in second-tier MLP in `APDReadout`.
      mlp2_dropout_p (float)      : Dropout probability in second-tier MLP in `APDReadout`.
      mlp2_hidden_dim (int)       : Number of weights (layer width) in second-tier
                                    MLP in `APDReadout`.
      message_passes (int)        : Number of message passing steps.
      message_size (int)          : Size of message passed (input size to `GRU`).
      s2v_lstm_computations (int) : Number of LSTM computations (loop) in S2V readout.
      s2v_memory_size (int)       : Number of input features and hidden state size
                                    in LSTM cell in S2V readout.
    """
    hyperparameters = {
        "enn_depth"            : 4,
        "enn_dropout_p"        : 0.0,
        "enn_hidden_dim"       : 250,
        "mlp1_depth"           : 4,
        "mlp1_dropout_p"       : 0.0,
        "mlp1_hidden_dim"      : 500,
        "mlp2_depth"           : 4,
        "mlp2_dropout_p"       : 0.0,
        "mlp2_hidden_dim"      : 500,
        "hidden_node_features" : 100,
        "message_passes"       : 3,
        "message_size"         : 100,
        "s2v_lstm_computations": 3,
        "s2v_memory_size"      : 100,
    }
elif parameters["model"] == "AttS2V":
    """
    AttS2V hyperparameters:
      att_depth (int)             : Num layers in 'att_enn' MLP.
      att_dropout_p (float)       : Dropout probability in 'att_enn' MLP.
      att_hidden_dim (int)        : Number of weights (layer width) in 'att_enn'
                                    MLP.
      enn_depth (int)             : Num layers in 'enn' MLP.
      enn_dropout_p (float)       : Dropout probability in 'enn' MLP.
      enn_hidden_dim (int)        : Number of weights (layer width) in 'enn' MLP.
      mlp1_depth (int)            : Num layers in first-tier MLP in `APDReadout`.
      mlp1_dropout_p (float)      : Dropout probability in first-tier MLP in `APDReadout`.
      mlp1_hidden_dim (int)       : Number of weights (layer width) in first-tier
                                    MLP in `APDReadout`.
      mlp2_depth (int)            : Num layers in second-tier MLP in `APDReadout`.
      mlp2_dropout_p (float)      : Dropout probability in second-tier MLP in `APDReadout`.
      mlp2_hidden_dim (int)       : Number of weights (layer width) in second-tier
                                    MLP in `APDReadout`.
      message_passes (int)        : Number of message passing steps.
      message_size (int)          : Size of message passed (output size of 'att_enn'
                                    MLP, input size to `GRU`).
      s2v_lstm_computations (int) : Number of LSTM computations (loop) in S2V readout.
      s2v_memory_size (int)       : Number of input features and hidden state size
                                    in LSTM cell in S2V readout.
    """
    hyperparameters = {
        "att_depth"            : 4,
        "att_dropout_p"        : 0.0,
        "att_hidden_dim"       : 250,
        "enn_depth"            : 4,
        "enn_dropout_p"        : 0.0,
        "enn_hidden_dim"       : 250,
        "mlp1_depth"           : 4,
        "mlp1_dropout_p"       : 0.0,
        "mlp1_hidden_dim"      : 500,
        "mlp2_depth"           : 4,
        "mlp2_dropout_p"       : 0.0,
        "mlp2_hidden_dim"      : 500,
        "hidden_node_features" : 100,
        "message_passes"       : 3,
        "message_size"         : 100,
        "s2v_lstm_computations": 3,
        "s2v_memory_size"      : 100,
    }
elif parameters["model"] == "GGNN":
    """
    GGNN hyperparameters:
      enn_depth (int)              : Num layers in 'enn' MLP.
      enn_dropout_p (float)        : Dropout probability in 'enn' MLP.
      enn_hidden_dim (int)         : Number of weights (layer width) in 'enn' MLP.
      mlp1_depth (int)             : Num layers in first-tier MLP in `APDReadout`.
      mlp1_dropout_p (float)       : Dropout probability in first-tier MLP in `APDReadout`.
      mlp1_hidden_dim (int)        : Number of weights (layer width) in first-tier
                                     MLP in `APDReadout`.
      mlp2_depth (int)             : Num layers in second-tier MLP in `APDReadout`.
      mlp2_dropout_p (float)       : Dropout probability in second-tier MLP in `APDReadout`.
      mlp2_hidden_dim (int)        : Number of weights (layer width) in second-tier
                                     MLP in `APDReadout`.
      gather_att_depth (int)       : Num layers in 'gather_att' MLP in `GraphGather`.
      gather_att_dropout_p (float) : Dropout probability in 'gather_att' MLP in
                                     `GraphGather`.
      gather_att_hidden_dim (int)  : Number of weights (layer width) in 'gather_att'
                                     MLP in `GraphGather`.
      gather_emb_depth (int)       : Num layers in 'gather_emb' MLP in `GraphGather`.
      gather_emb_dropout_p (float) : Dropout probability in 'gather_emb' MLP in
                                     `GraphGather`.
      gather_emb_hidden_dim (int)  : Number of weights (layer width) in 'gather_emb'
                                     MLP in `GraphGather`.
      gather_width (int)           : Output size of `GraphGather` block.
      message_passes (int)         : Number of message passing steps.
      message_size (int)           : Size of message passed (output size of all
                                     MLPs in message aggregation step, input size
                                     to `GRU`).
    """
    hyperparameters = {
        "enn_depth"            : 4,
        "enn_dropout_p"        : 0.0,
        "enn_hidden_dim"       : 250,
        "mlp1_depth"           : 4,
        "mlp1_dropout_p"       : 0.0,
        "mlp1_hidden_dim"      : 500,
        "mlp2_depth"           : 4,
        "mlp2_dropout_p"       : 0.0,
        "mlp2_hidden_dim"      : 500,
        "gather_att_depth"     : 4,
        "gather_att_dropout_p" : 0.0,
        "gather_att_hidden_dim": 250,
        "gather_emb_depth"     : 4,
        "gather_emb_dropout_p" : 0.0,
        "gather_emb_hidden_dim": 250,
        "gather_width"         : 100,
        "hidden_node_features" : 100,
        "message_passes"       : 3,
        "message_size"         : 100,
    }
elif parameters["model"] == "AttGGNN":
    """
    AttGGNN hyperparameters:
      att_depth (int)              : Num layers in 'att_nns' MLP (message aggregation
                                     step).
      att_dropout_p (float)        : Dropout probability in 'att_nns' MLP (message
                                     aggregation step).
      att_hidden_dim (int)         : Number of weights (layer width) in 'att_nns'
                                     MLP (message aggregation step).
      mlp1_depth (int)             : Num layers in first-tier MLP in `APDReadout`.
      mlp1_dropout_p (float)       : Dropout probability in first-tier MLP in `APDReadout`.
      mlp1_hidden_dim (int)        : Number of weights (layer width) in first-tier
                                     MLP in `APDReadout`.
      mlp2_depth (int)             : Num layers in second-tier MLP in `APDReadout`.
      mlp2_dropout_p (float)       : Dropout probability in second-tier MLP in `APDReadout`.
      mlp2_hidden_dim (int)        : Number of weights (layer width) in second-tier
                                     MLP in `APDReadout`.
      gather_att_depth (int)       : Num layers in 'gather_att' MLP in `GraphGather`.
      gather_att_dropout_p (float) : Dropout probability in 'gather_att' MLP in
                                     `GraphGather`.
      gather_att_hidden_dim (int)  : Number of weights (layer width) in 'gather_att'
                                     MLP in `GraphGather`.
      gather_emb_depth (int)       : Num layers in 'gather_emb' MLP in `GraphGather`.
      gather_emb_dropout_p (float) : Dropout probability in 'gather_emb' MLP in
                                     `GraphGather`.
      gather_emb_hidden_dim (int)  : Number of weights (layer width) in 'gather_emb'
                                     MLP in `GraphGather`.
      gather_width (int)           : Output size of `GraphGather` block.
      message_passes (int)         : Number of message passing steps.
      message_size (int)           : Size of message passed (output size of all
                                     MLPs in message aggregation step, input size
                                     to `GRU`).
      msg_depth (int)              : Num layers in 'msg_nns' MLP (message aggregation
                                     step).
      msg_dropout_p (float)        : Dropout probability in 'msg_nns' MLP (message
                                     aggregation step).
      msg_hidden_dim (int)         : Number of weights (layer width) in 'msg_nns'
                                     MLP (message aggregation step).
    """
    hyperparameters = {
        "att_depth"            : 4,
        "att_dropout_p"        : 0.0,
        "att_hidden_dim"       : 250,
        "mlp1_depth"           : 4,
        "mlp1_dropout_p"       : 0.0,
        "mlp1_hidden_dim"      : 500,
        "mlp2_depth"           : 4,
        "mlp2_dropout_p"       : 0.0,
        "mlp2_hidden_dim"      : 500,
        "gather_att_depth"     : 4,
        "gather_att_dropout_p" : 0.0,
        "gather_att_hidden_dim": 250,
        "gather_emb_depth"     : 4,
        "gather_emb_dropout_p" : 0.0,
        "gather_emb_hidden_dim": 250,
        "gather_width"         : 100,
        "hidden_node_features" : 100,
        "message_passes"       : 3,
        "message_size"         : 100,
        "msg_depth"            : 4,
        "msg_dropout_p"        : 0.0,
        "msg_hidden_dim"       : 250,
    }
elif parameters["model"] == "EMN":
    """
    EMN hyperparameters:
      att_depth (int)              : Num layers in 'att_msg_nn' MLP (edge propagation
                                     step).
      att_dropout_p (float)        : Dropout probability in 'att_msg_nn' MLP (edge
                                     propagation step).
      att_hidden_dim (int)         : Number of weights (layer width) in 'att_msg_nn'
                                     MLP (edge propagation step).
      edge_emb_depth (int)         : Num layers in 'embedding_nn' MLP (edge processing
                                     step).
      edge_emb_dropout_p (float)   : Dropout probability in 'embedding_nn' MLP (edge
                                     processing step).
      edge_emb_hidden_dim (int)    : Number of weights (layer width) in 'embedding_nn'
                                     MLP (edge processing step).
      edge_emb_size (int)          : Output size of all MLPs in edge propagation
                                     and processing steps (input size to `GraphGather`).
      mlp1_depth (int)             : Num layers in first-tier MLP in `APDReadout`.
      mlp1_dropout_p (float)       : Dropout probability in first-tier MLP in `APDReadout`.
      mlp1_hidden_dim (int)        : Number of weights (layer width) in first-tier
                                     MLP in `APDReadout`.
      mlp2_depth (int)             : Num layers in second-tier MLP in `APDReadout`.
      mlp2_dropout_p (float)       : Dropout probability in second-tier MLP in `APDReadout`.
      mlp2_hidden_dim (int)        : Number of weights (layer width) in second-tier
                                     MLP in `APDReadout`.
      gather_att_depth (int)       : Num layers in 'gather_att' MLP in `GraphGather`.
      gather_att_dropout_p (float) : Dropout probability in 'gather_att' MLP in
                                     `GraphGather`.
      gather_att_hidden_dim (int)  : Number of weights (layer width) in 'gather_att'
                                     MLP in `GraphGather`.
      gather_emb_depth (int)       : Num layers in 'gather_emb' MLP in `GraphGather`.
      gather_emb_dropout_p (float) : Dropout probability in 'gather_emb' MLP in
                                     `GraphGather`.
      gather_emb_hidden_dim (int)  : Number of weights (layer width) in 'gather_emb'
                                     MLP in `GraphGather`.
      gather_width (int)           : Output size of `GraphGather` block.
      message_passes (int)         : Number of message passing steps.
      msg_depth (int)              : Num layers in 'emb_msg_nn' MLP (edge propagation
                                     step).
      msg_dropout_p (float)        : Dropout probability in 'emb_msg_n' MLP (edge
                                     propagation step).
      msg_hidden_dim (int)         : Number of weights (layer width) in 'emb_msg_nn'
                                     MLP (edge propagation step).
    """
    hyperparameters = {
        "att_depth"            : 4,
        "att_dropout_p"        : 0.0,
        "att_hidden_dim"       : 250,
        "edge_emb_depth"       : 4,
        "edge_emb_dropout_p"   : 0.0,
        "edge_emb_hidden_dim"  : 250,
        "edge_emb_size"        : 100,
        "mlp1_depth"           : 4,
        "mlp1_dropout_p"       : 0.0,
        "mlp1_hidden_dim"      : 500,
        "mlp2_depth"           : 4,
        "mlp2_dropout_p"       : 0.0,
        "mlp2_hidden_dim"      : 500,
        "gather_att_depth"     : 4,
        "gather_att_dropout_p" : 0.0,
        "gather_att_hidden_dim": 250,
        "gather_emb_depth"     : 4,
        "gather_emb_dropout_p" : 0.0,
        "gather_emb_hidden_dim": 250,
        "gather_width"         : 100,
        "message_passes"       : 3,
        "msg_depth"            : 4,
        "msg_dropout_p"        : 0.0,
        "msg_hidden_dim"       : 250,
    }

# make sure dataset dir ends in "/"
if parameters["dataset_dir"][-1] != "/":
    print("* Adding '/' to end of `dataset_dir`.")
    parameters["dataset_dir"] += "/"

# join dictionaries
parameters.update(hyperparameters)
