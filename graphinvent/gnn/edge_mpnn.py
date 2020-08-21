# load general packages and functions
import torch


class EdgeMPNN(torch.nn.Module):
    """ Abstract `EdgeMPNN` class. A specific model using this class is defined
    in `mpnn.py`; this is the EMN.
    """
    def __init__(self, edge_features, edge_embedding_size, message_passes, n_nodes_largest_graph):
        super(EdgeMPNN, self).__init__()
        self.edge_features = edge_features
        self.edge_embedding_size = edge_embedding_size
        self.message_passes = message_passes
        self.n_nodes_largest_graph = n_nodes_largest_graph

    def preprocess_edges(self, nodes, node_neighbours, edges):
        """ Edge preprocessing step, to be implemented in all `EdgeMPNN`
        subclasses.

        Args:
          nodes (torch.Tensor) : Batch of size {total number of nodes in batch,
            number of node features}.
          node_neighbours (torch.Tensor) : Batch of size {total number of nodes
            in batch, max node degree, number of node features}.
          edges (torch.Tensor) : Batch of size {total number of nodes in batch,
            max node degree, number of edge features}.
        """
        raise NotImplementedError

    def propagate_edges(self, edges, ingoing_edge_memories, ingoing_edges_mask):
        """ Edge propagation rule, to be implemented in all `EdgeMPNN` subclasses.

        Args:
          edges (torch.Tensor) : Batch of size {N, number of nodes, number of
            nodes, total number of edge features}, where N is the total number
            of subgraphs in the batch.
          ingoing_edge_memories (torch.Tensor) : Batch of size {total number of
            edges in batch, total number of edge features}.
          ingoing_edges_mask (torch.Tensor) : Batch of size {total number of
            edges in batch, max node degree, total number of edge features}.
        """
        raise NotImplementedError

    def readout(self, hidden_nodes, input_nodes, node_mask):
        """ Local readout function, to be implemented in all `EdgeMPNN` subclasses.

        Args:
          hidden_nodes (torch.Tensor) : Batch of size {total number of nodes in
            batch, number of node features}.
          input_nodes (torch.Tensor) : Batch of size {total number of nodes in
            batch, number of node features}.
          node_mask (torch.Tensor) : Batch of size {total number of nodes in
            batch, number of node features}, where elements are 1 if
            corresponding element exists and 0 otherwise.
        """
        raise NotImplementedError

    def forward(self, nodes, edges):
        """ Defines forward pass.

        Args:
          nodes (torch.Tensor) : Batch of size {N, number of node features,
            number of nodes}, where N is the number of subgraphs in each batch.
          edges (torch.Tensor) : Batch of size {N, number of nodes, number of
            nodes, number of edge features}, where N is the number of subgraphs
            in each batch.
        """
        adjacency = torch.sum(edges, dim=3)

        # indices for finding edges in batch; `edges_b_idx` is batch index,
        # `edges_n_idx` is the node index, and `edges_nghb_idx` is the index
        # that each node in `edges_n_idx` is bound to
        edges_b_idx, edges_n_idx, edges_nghb_idx = adjacency.nonzero(as_tuple=True)

        n_edges = edges_n_idx.shape[0]
        adj_of_edge_batch_idc = adjacency.clone().long()

        # +1 to distinguish idx 0 from empty elements, subtracted few lines down
        r = torch.arange(1, n_edges + 1, device="cuda")

        adj_of_edge_batch_idc[edges_b_idx, edges_n_idx, edges_nghb_idx] = r

        ingoing_edges_eb_idx = (
            torch.cat([row[row.nonzero()] for row in adj_of_edge_batch_idc[edges_b_idx, edges_nghb_idx, :]]) - 1
        ).squeeze()

        edge_degrees = adjacency[edges_b_idx, edges_nghb_idx, :].sum(-1).long()
        ingoing_edges_igeb_idx = torch.cat([i * torch.ones(d) for i, d in enumerate(edge_degrees)]).long()
        ingoing_edges_ige_idx = torch.cat([torch.arange(i) for i in edge_degrees]).long()


        batch_size = adjacency.shape[0]
        n_nodes = adjacency.shape[1]
        max_node_degree = adjacency.sum(-1).max().int()
        edge_memories = torch.zeros(n_edges, self.edge_embedding_size, device="cuda")

        ingoing_edge_memories = torch.zeros(n_edges, max_node_degree, self.edge_embedding_size, device="cuda")
        ingoing_edges_mask = torch.zeros(n_edges, max_node_degree, device="cuda")

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
            ingoing_edges_eb_idx = ingoing_edges_eb_idx[diff_idx].squeeze()
            ingoing_edges_ige_idx = ingoing_edges_ige_idx[diff_idx].squeeze()
            ingoing_edges_igeb_idx = ingoing_edges_igeb_idx[diff_idx].squeeze()
        except:
            pass

        ingoing_edges_mask[ingoing_edges_igeb_idx, ingoing_edges_ige_idx] = 1

        for _ in range(self.message_passes):
            ingoing_edge_memories[ingoing_edges_igeb_idx, ingoing_edges_ige_idx, :] = edge_memories[ingoing_edges_eb_idx, :]
            edge_memories = self.propagate_edges(edges=edge_batch_edges,
                                                 ingoing_edge_memories=ingoing_edge_memories.clone(),#.detach(),
                                                 ingoing_edges_mask=ingoing_edges_mask)

        node_mask = (adjacency.sum(-1) != 0)

        node_sets = torch.zeros(batch_size, n_nodes, max_node_degree, self.edge_embedding_size, device="cuda")

        edge_batch_edge_memory_idc = torch.cat(
            [torch.arange(row.sum()) for row in adjacency.view(-1, n_nodes)]
        ).long()

        node_sets[edges_b_idx, edges_n_idx, edge_batch_edge_memory_idc, :] = edge_memories
        graph_sets = node_sets.sum(2)

        output = self.readout(graph_sets, graph_sets, node_mask)
        return output
