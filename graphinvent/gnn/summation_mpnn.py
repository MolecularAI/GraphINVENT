"""
Defines the `SummationMPNN` class.
"""
# load general packages and functions
from collections import namedtuple
import torch


class SummationMPNN(torch.nn.Module):
    """
    Abstract `SummationMPNN` class. Specific models using this class are
    defined in `mpnn.py`; these are MNN, S2V, and GGNN.
    """
    def __init__(self, constants : namedtuple):

        super().__init__()

        self.hidden_node_features = constants.hidden_node_features
        self.edge_features        = constants.n_edge_features
        self.message_size         = constants.message_size
        self.message_passes       = constants.message_passes
        self.constants            = constants

    def message_terms(self, nodes : torch.Tensor, node_neighbours : torch.Tensor,
                      edges : torch.Tensor) -> None:
        """
        Message passing function, to be implemented in all `SummationMPNN` subclasses.

        Args:
        ----
            nodes (torch.Tensor)           : Batch of node feature vectors.
            node_neighbours (torch.Tensor) : Batch of node feature vectors for neighbors.
            edges (torch.Tensor)           : Batch of edge feature vectors.

        Shapes:
        ------
            nodes           : (total N nodes in batch, N node features)
            node_neighbours : (total N nodes in batch, max node degree, N node features)
            edges           : (total N nodes in batch, max node degree, N edge features)
        """
        raise NotImplementedError

    def update(self, nodes : torch.Tensor, messages : torch.Tensor) -> None:
        """
        Message update function, to be implemented in all `SummationMPNN` subclasses.

        Args:
        ----
            nodes (torch.Tensor)    : Batch of node feature vectors.
            messages (torch.Tensor) : Batch of incoming messages.

        Shapes:
        ------
            nodes    : (total N nodes in batch, N node features)
            messages : (total N nodes in batch, N node features)
        """
        raise NotImplementedError

    def readout(self, hidden_nodes : torch.Tensor, input_nodes : torch.Tensor,
                node_mask : torch.Tensor) -> None:
        """
        Local readout function, to be implemented in all `SummationMPNN` subclasses.

        Args:
        ----
            hidden_nodes (torch.Tensor) : Batch of node feature vectors.
            input_nodes (torch.Tensor) : Batch of node feature vectors.
            node_mask (torch.Tensor) : Mask for non-existing neighbors, where elements
                                       are 1 if corresponding element exists and 0
                                       otherwise.

        Shapes:
        ------
            hidden_nodes : (total N nodes in batch, N node features)
            input_nodes : (total N nodes in batch, N node features)
            node_mask : (total N nodes in batch, N features)
        """
        raise NotImplementedError

    def forward(self, nodes : torch.Tensor, edges : torch.Tensor) -> None:
        """
        Defines forward pass.

        Args:
        ----
            nodes (torch.Tensor) : Batch of node feature matrices.
            edges (torch.Tensor) : Batch of edge feature tensors.

        Shapes:
        ------
            nodes : (batch size, N nodes, N node features)
            edges : (batch size, N nodes, N nodes, N edge features)

        Returns:
        -------
            output (torch.Tensor) : This would normally be the learned graph representation,
                                    but in all MPNN readout functions in this work,
                                    the last layer is used to predict the action
                                    probability distribution for a batch of graphs
                                    from the learned graph representation.
        """
        adjacency = torch.sum(edges, dim=3)

        # **note: "idc" == "indices", "nghb{s}" == "neighbour(s)"
        (edge_batch_batch_idc,
         edge_batch_node_idc,
         edge_batch_nghb_idc) = adjacency.nonzero(as_tuple=True)

        (node_batch_batch_idc, node_batch_node_idc) = adjacency.sum(-1).nonzero(as_tuple=True)

        same_batch = node_batch_batch_idc.view(-1, 1) == edge_batch_batch_idc
        same_node  = node_batch_node_idc.view(-1, 1) == edge_batch_node_idc

        # element ij of `message_summation_matrix` is 1 if `edge_batch_edges[j]`
        # is connected with `node_batch_nodes[i]`, else 0
        message_summation_matrix = (same_batch * same_node).float()

        edge_batch_edges = edges[edge_batch_batch_idc, edge_batch_node_idc, edge_batch_nghb_idc, :]

        # pad up the hidden nodes
        hidden_nodes = torch.zeros(nodes.shape[0],
                                   nodes.shape[1],
                                   self.hidden_node_features,
                                   device=self.constants.device)
        hidden_nodes[:nodes.shape[0], :nodes.shape[1], :nodes.shape[2]] = nodes.clone()
        node_batch_nodes = hidden_nodes[node_batch_batch_idc, node_batch_node_idc, :]

        for _ in range(self.message_passes):
            edge_batch_nodes = hidden_nodes[edge_batch_batch_idc, edge_batch_node_idc, :]

            edge_batch_nghbs = hidden_nodes[edge_batch_batch_idc, edge_batch_nghb_idc, :]

            message_terms    = self.message_terms(edge_batch_nodes,
                                                  edge_batch_nghbs,
                                                  edge_batch_edges)

            if len(message_terms.size()) == 1:  # if a single graph in batch
                message_terms = message_terms.unsqueeze(0)

            # the summation in eq. 1 of the NMPQC paper happens here
            messages = torch.matmul(message_summation_matrix, message_terms)

            node_batch_nodes = self.update(node_batch_nodes, messages)
            hidden_nodes[node_batch_batch_idc, node_batch_node_idc, :] = node_batch_nodes.clone()

        node_mask = adjacency.sum(-1) != 0
        output    = self.readout(hidden_nodes, nodes, node_mask)

        return output
