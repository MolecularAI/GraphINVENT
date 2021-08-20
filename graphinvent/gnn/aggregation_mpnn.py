"""
Defines the `AggregationMPNN` class.
"""
# load general packages and functions
from collections import  namedtuple
import torch


class AggregationMPNN(torch.nn.Module):
    """
    Abstract `AggregationMPNN` class. Specific models using this class are
    defined in `mpnn.py`; these are the attention networks AttS2V and AttGGNN.
    """
    def __init__(self, constants : namedtuple) -> None:
        super().__init__()

        self.hidden_node_features = constants.hidden_node_features
        self.edge_features        = constants.n_edge_features
        self.message_size         = constants.message_size
        self.message_passes       = constants.message_passes
        self.constants            = constants

    def aggregate_message(self, nodes : torch.Tensor, node_neighbours : torch.Tensor,
                          edges : torch.Tensor, mask : torch.Tensor) -> None:
        """
        Message aggregation function, to be implemented in all `AggregationMPNN` subclasses.

        Args:
        ----
            nodes (torch.Tensor)           : Batch of node feature vectors.
            node_neighbours (torch.Tensor) : Batch of node feature vectors for neighbors.
            edges (torch.Tensor)           : Batch of edge feature vectors.
            mask (torch.Tensor)            : Mask for non-existing neighbors, where
                                             elements are 1 if corresponding element
                                             exists and 0 otherwise.

        Shapes:
        ------
            nodes           : (total N nodes in batch, N node features)
            node_neighbours : (total N nodes in batch, max node degree, N node features)
            edges           : (total N nodes in batch, max node degree, N edge features)
            mask            : (total N nodes in batch, max node degree)
        """
        raise NotImplementedError

    def update(self, nodes : torch.Tensor, messages : torch.Tensor) -> None:
        """
        Message update function, to be implemented in all `AggregationMPNN` subclasses.

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
        Local readout function, to be implemented in all `AggregationMPNN` subclasses.

        Args:
        ----
            hidden_nodes (torch.Tensor) : Batch of node feature vectors.
            input_nodes (torch.Tensor)  : Batch of node feature vectors.
            node_mask (torch.Tensor)    : Mask for non-existing neighbors, where
                                          elements are 1 if corresponding element
                                          exists and 0 otherwise.

        Shapes:
        ------
            hidden_nodes : (total N nodes in batch, N node features)
            input_nodes  : (total N nodes in batch, N node features)
            node_mask    : (total N nodes in batch, N features)
        """
        raise NotImplementedError

    def forward(self, nodes : torch.Tensor, edges : torch.Tensor) -> torch.Tensor:
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
            output (torch.Tensor) : This would normally be the learned graph
                                    representation, but in all MPNN readout functions
                                    in this work, the last layer is used to predict
                                    the action probability distribution for a batch
                                    of graphs from the learned graph representation.
        """
        adjacency = torch.sum(edges, dim=3)

        # **note: "idc" == "indices", "nghb{s}" == "neighbour(s)"
        edge_batch_batch_idc, edge_batch_node_idc, edge_batch_nghb_idc = \
            adjacency.nonzero(as_tuple=True)

        node_batch_batch_idc, node_batch_node_idc = adjacency.sum(-1).nonzero(as_tuple=True)
        node_batch_adj  = adjacency[node_batch_batch_idc, node_batch_node_idc, :]
        node_batch_size = node_batch_batch_idc.shape[0]
        node_degrees    = node_batch_adj.sum(-1).long()
        max_node_degree = node_degrees.max()

        node_batch_node_nghbs = torch.zeros(node_batch_size,
                                            max_node_degree,
                                            self.hidden_node_features,
                                            device=self.constants.device)
        node_batch_edges      = torch.zeros(node_batch_size,
                                            max_node_degree,
                                            self.edge_features,
                                            device=self.constants.device)

        node_batch_nghb_nghb_idc = torch.cat(
            [torch.arange(i) for i in node_degrees]
        ).long()

        edge_batch_node_batch_idc = torch.cat(
            [i * torch.ones(degree) for i, degree in enumerate(node_degrees)]
        ).long()

        node_batch_node_nghb_mask = torch.zeros(node_batch_size,
                                                max_node_degree,
                                                device=self.constants.device)

        node_batch_node_nghb_mask[edge_batch_node_batch_idc, node_batch_nghb_nghb_idc] = 1

        node_batch_edges[edge_batch_node_batch_idc, node_batch_nghb_nghb_idc, :] = \
            edges[edge_batch_batch_idc, edge_batch_node_idc, edge_batch_nghb_idc, :]

        # pad up the hidden nodes
        hidden_nodes = torch.zeros(nodes.shape[0],
                                   nodes.shape[1],
                                   self.hidden_node_features,
                                   device=self.constants.device)
        hidden_nodes[:nodes.shape[0], :nodes.shape[1], :nodes.shape[2]] = nodes.clone()

        for _ in range(self.message_passes):

            node_batch_nodes = hidden_nodes[node_batch_batch_idc, node_batch_node_idc, :]
            node_batch_node_nghbs[edge_batch_node_batch_idc, node_batch_nghb_nghb_idc, :] = \
                hidden_nodes[edge_batch_batch_idc, edge_batch_nghb_idc, :]

            messages = self.aggregate_message(nodes=node_batch_nodes,
                                              node_neighbours=node_batch_node_nghbs.clone(),
                                              edges=node_batch_edges,
                                              mask=node_batch_node_nghb_mask)

            hidden_nodes[node_batch_batch_idc, node_batch_node_idc, :] = \
                self.update(node_batch_nodes.clone(), messages)

        node_mask = (adjacency.sum(-1) != 0)

        output = self.readout(hidden_nodes, nodes, node_mask)

        return output
