# load general packages and functions
import torch


class AggregationMPNN(torch.nn.Module):
    """ Abstract `AggregationMPNN` class. Specific models using this class are
    defined in `mpnn.py`; these are the attention networks AttS2V and AttGGNN.
    """
    def __init__(self, node_features, hidden_node_features, edge_features,
                 message_size, message_passes):
        super(AggregationMPNN, self).__init__()

        self.hidden_node_features = hidden_node_features
        self.edge_features = edge_features
        self.message_size = message_size
        self.message_passes = message_passes

    def aggregate_message(self, nodes, node_neighbours, edges, mask):
        """ Message aggregation function, to be implemented in all
        `AggregationMPNN` subclasses.

        Args:
          nodes (torch.Tensor) : Batch of size {total number of nodes in batch,
            number of node features}.
          node_neighbours (torch.Tensor) : Batch of size {total number of nodes
            in batch, max node degree, number of node features}.
          edges (torch.Tensor) : Batch of size {total number of nodes in batch,
            max node degree, number of edge features}.
          mask (torch.Tensor) : Batch of size {total number of nodes in batch,
            max node degree}, where elements are 1 if corresponding neighbour
            exists and 0 otherwise.
        """
        raise NotImplementedError

    def update(self, nodes, messages):
        """ Message update function, to be implemented in all `AggregationMPNN`
        subclasses.

        Args:
          nodes (torch.Tensor) : Batch of size {total number of nodes in batch,
            number of node features}.
          messages (torch.Tensor) : Batch of size {total number of nodes in
            batch, number of node features}.
        """
        raise NotImplementedError

    def readout(self, hidden_nodes, input_nodes, node_mask):
        """ Local readout function, to be implemented in all `AggregationMPNN`
        subclasses.

        Args:
          hidden_nodes (torch.Tensor) : Batch of size {total number of nodes in
            batch, number of node features}.
          input_nodes (torch.Tensor) : Batch of size {total number of nodes in
            batch, number of node features}.
          node_mask (torch.Tensor) : Batch of size {total number of nodes in
            batch, number of features}, where elements are 1 if corresponding
            element exists and 0 otherwise.
        """
        raise NotImplementedError

    def forward(self, nodes, edges):
        """ Defines forward pass.

        Args:
          nodes (torch.Tensor) : Batch of size (i{N, number of node features,
            number of nodes}, where N is the number of subgraphs in each batch.
          edges (torch.Tensor) : Batch of size {N, number of node features,
            number of edge features}, where N is the number of subgraphs in
            each batch.
        """
        adjacency = torch.sum(edges, dim=3)

        # **note: "idc" == "indices", "nghb{s}" == "neighbour(s)"
        edge_batch_batch_idc, edge_batch_node_idc, edge_batch_nghb_idc = adjacency.nonzero(as_tuple=True)

        node_batch_batch_idc, node_batch_node_idc = adjacency.sum(-1).nonzero(as_tuple=True)
        node_batch_adj = adjacency[node_batch_batch_idc, node_batch_node_idc, :]

        node_batch_size = node_batch_batch_idc.shape[0]
        node_degrees = node_batch_adj.sum(-1).long()
        max_node_degree = node_degrees.max()

        node_batch_node_nghbs = torch.zeros(node_batch_size, max_node_degree, self.hidden_node_features, device="cuda")
        node_batch_edges = torch.zeros(node_batch_size, max_node_degree, self.edge_features, device="cuda")

        node_batch_nghb_nghb_idc = torch.cat(
            [torch.arange(i) for i in node_degrees]
        ).long()

        edge_batch_node_batch_idc = torch.cat(
            [i * torch.ones(degree) for i, degree in enumerate(node_degrees)]
        ).long()

        node_batch_node_nghb_mask = torch.zeros(
            node_batch_size, max_node_degree, device="cuda"
        )

        node_batch_node_nghb_mask[edge_batch_node_batch_idc, node_batch_nghb_nghb_idc] = 1

        node_batch_edges[edge_batch_node_batch_idc, node_batch_nghb_nghb_idc, :] = \
            edges[edge_batch_batch_idc, edge_batch_node_idc, edge_batch_nghb_idc, :]

        # pad up the hidden nodes
        hidden_nodes = torch.zeros(nodes.shape[0], nodes.shape[1], self.hidden_node_features, device="cuda")
        hidden_nodes[:nodes.shape[0], :nodes.shape[1], :nodes.shape[2]] = nodes.clone()

        for _ in range(self.message_passes):

            node_batch_nodes = hidden_nodes[node_batch_batch_idc, node_batch_node_idc, :]
            node_batch_node_nghbs[edge_batch_node_batch_idc, node_batch_nghb_nghb_idc, :] = \
                hidden_nodes[edge_batch_batch_idc, edge_batch_nghb_idc, :]

            messages = self.aggregate_message(nodes=node_batch_nodes,
                                              node_neighbours=node_batch_node_nghbs.clone(),
                                              edges=node_batch_edges,
                                              mask=node_batch_node_nghb_mask)

            hidden_nodes[node_batch_batch_idc, node_batch_node_idc, :] = self.update(node_batch_nodes.clone(), messages)

        node_mask = (adjacency.sum(-1) != 0)

        output = self.readout(hidden_nodes, nodes, node_mask)

        return output
