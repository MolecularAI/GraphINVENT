"""
Defines the `EdgeMPNN` class.
"""# load general packages and functions
from collections import namedtuple
import torch


class EdgeMPNN(torch.nn.Module):
    """
    Abstract `EdgeMPNN` class. A specific model using this class is defined
    in `mpnn.py`; this is the EMN.
    """
    def __init__(self, constants : namedtuple) -> None:
        super().__init__()

        self.edge_features         = constants.edge_features
        self.edge_embedding_size   = constants.edge_embedding_size
        self.message_passes        = constants.message_passes
        self.n_nodes_largest_graph = constants.max_n_nodes
        self.constants             = constants

    def preprocess_edges(self, nodes : torch.Tensor, node_neighbours : torch.Tensor,
                         edges : torch.Tensor) -> None:
        """
        Edge preprocessing step, to be implemented in all `EdgeMPNN` subclasses.

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

    def propagate_edges(self, edges : torch.Tensor, ingoing_edge_memories : torch.Tensor,
                        ingoing_edges_mask : torch.Tensor) -> None:
        """
        Edge propagation rule, to be implemented in all `EdgeMPNN` subclasses.

        Args:
        ----
            edges (torch.Tensor)                 : Batch of edge feature tensors.
            ingoing_edge_memories (torch.Tensor) : Batch of memories for all
                                                   ingoing edges.
            ingoing_edges_mask (torch.Tensor)    : Mask for ingoing edges.

        Shapes:
        ------
            edges                 : (batch size, N nodes, N nodes, total N edge features)
            ingoing_edge_memories : (total N edges in batch, total N edge features)
            ingoing_edges_mask    : (total N edges in batch, max node degree, total N edge features)
        """
        raise NotImplementedError

    def readout(self, hidden_nodes : torch.Tensor, input_nodes : torch.Tensor,
                node_mask : torch.Tensor) -> None:
        """
        Local readout function, to be implemented in all `EdgeMPNN` subclasses.

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
            output (torch.Tensor) : This would normally be the learned graph representation,
                                    but in all MPNN readout functions in this work,
                                    the last layer is used to predict the action
                                    probability distribution for a batch of graphs from
                                    the learned graph representation.
        """
        adjacency = torch.sum(edges, dim=3)

        # indices for finding edges in batch; `edges_b_idx` is batch index,
        # `edges_n_idx` is the node index, and `edges_nghb_idx` is the index
        # that each node in `edges_n_idx` is bound to
        edges_b_idx, edges_n_idx, edges_nghb_idx = adjacency.nonzero(as_tuple=True)

        n_edges               = edges_n_idx.shape[0]
        adj_of_edge_batch_idc = adjacency.clone().long()

        # +1 to distinguish idx 0 from empty elements, subtracted few lines down
        r = torch.arange(1, n_edges + 1, device=self.constants.device)

        adj_of_edge_batch_idc[edges_b_idx, edges_n_idx, edges_nghb_idx] = r

        ingoing_edges_eb_idx = (
            torch.cat([row[row.nonzero()] for row in
            adj_of_edge_batch_idc[edges_b_idx, edges_nghb_idx, :]]) - 1
        ).squeeze()

        edge_degrees = adjacency[edges_b_idx, edges_nghb_idx, :].sum(-1).long()
        ingoing_edges_igeb_idx = torch.cat(
            [i * torch.ones(d) for i, d in enumerate(edge_degrees)]
        ).long()
        ingoing_edges_ige_idx = torch.cat([torch.arange(i) for i in edge_degrees]).long()


        batch_size      = adjacency.shape[0]
        n_nodes         = adjacency.shape[1]
        max_node_degree = adjacency.sum(-1).max().int()
        edge_memories   = torch.zeros(n_edges,
                                      self.edge_embedding_size,
                                      device=self.constants.device)

        ingoing_edge_memories = torch.zeros(n_edges, max_node_degree,
                                            self.edge_embedding_size,
                                            device=self.constants.device)
        ingoing_edges_mask    = torch.zeros(n_edges,
                                            max_node_degree,
                                            device=self.constants.device)

        edge_batch_nodes = nodes[edges_b_idx, edges_n_idx, :]
        # **note: "nghb{s}" == "neighbour(s)"
        edge_batch_nghbs = nodes[edges_b_idx, edges_nghb_idx, :]
        edge_batch_edges = edges[edges_b_idx, edges_n_idx, edges_nghb_idx, :]
        edge_batch_edges = self.preprocess_edges(nodes=edge_batch_nodes,
                                                 node_neighbours=edge_batch_nghbs,
                                                 edges=edge_batch_edges)

        # remove h_ji:s influence on h_ij
        ingoing_edges_nghb_idx = edges_nghb_idx[ingoing_edges_eb_idx]
        ingoing_edges_receiving_edge_n_idx = edges_n_idx[ingoing_edges_igeb_idx]
        diff_idx = (ingoing_edges_receiving_edge_n_idx != ingoing_edges_nghb_idx).nonzero()

        try:
            ingoing_edges_eb_idx   = ingoing_edges_eb_idx[diff_idx].squeeze()
            ingoing_edges_ige_idx  = ingoing_edges_ige_idx[diff_idx].squeeze()
            ingoing_edges_igeb_idx = ingoing_edges_igeb_idx[diff_idx].squeeze()
        except:
            pass

        ingoing_edges_mask[ingoing_edges_igeb_idx, ingoing_edges_ige_idx] = 1

        for _ in range(self.message_passes):
            ingoing_edge_memories[ingoing_edges_igeb_idx, ingoing_edges_ige_idx, :] = \
                edge_memories[ingoing_edges_eb_idx, :]
            edge_memories = self.propagate_edges(
                edges=edge_batch_edges,
                ingoing_edge_memories=ingoing_edge_memories.clone(),
                ingoing_edges_mask=ingoing_edges_mask
            )

        node_mask = (adjacency.sum(-1) != 0)

        node_sets = torch.zeros(batch_size,
                                n_nodes,
                                max_node_degree,
                                self.edge_embedding_size,
                                device=self.constants.device)

        edge_batch_edge_memory_idc = torch.cat(
            [torch.arange(row.sum()) for row in adjacency.view(-1, n_nodes)]
        ).long()

        node_sets[edges_b_idx, edges_n_idx, edge_batch_edge_memory_idc, :] = edge_memories
        graph_sets = node_sets.sum(2)

        output = self.readout(graph_sets, graph_sets, node_mask)
        return output
