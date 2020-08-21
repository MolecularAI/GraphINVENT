# load general packages and functions
import torch


class SummationMPNN(torch.nn.Module):
    """ Abstract `SummationMPNN` class. Specific models using this class are
    defined in `mpnn.py`; these are MNN, S2V, and GGNN.
    """
    def __init__(self, node_features, hidden_node_features, edge_features, message_size, message_passes):

        super(SummationMPNN, self).__init__()

        self.hidden_node_features = hidden_node_features
        self.edge_features = edge_features
        self.message_size = message_size
        self.message_passes = message_passes

    def message_terms(self, nodes, node_neighbours, edges):
        """ Message passing function, to be implemented in all `SummationMPNN`
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

    def update(self, nodes, messages):
        """ Message update function, to be implemented in all `SummationMPNN`
        subclasses.

        Args:
          nodes (torch.Tensor) : Batch of size {total number of nodes in
            batch, number of node features}.
          messages (torch.Tensor) : Batch of size {total number of nodes in
            batch, number of node features}.
        """
        raise NotImplementedError

    def readout(self, hidden_nodes, input_nodes, node_mask):
        """ Local readout function, to be implemented in all `SummationMPNN`
        subclasses.

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

        # **note: "idc" == "indices", "nghb{s}" == "neighbour(s)"
        (
            edge_batch_batch_idc,
            edge_batch_node_idc,
            edge_batch_nghb_idc,
        ) = adjacency.nonzero(as_tuple=True)

        (node_batch_batch_idc, node_batch_node_idc) = adjacency.sum(-1).nonzero(as_tuple=True)

        same_batch = node_batch_batch_idc.view(-1, 1) == edge_batch_batch_idc
        same_node = node_batch_node_idc.view(-1, 1) == edge_batch_node_idc

        # element ij of `message_summation_matrix` is 1 if `edge_batch_edges[j]`
        # is connected with `node_batch_nodes[i]`, else 0
        message_summation_matrix = (same_batch * same_node).float()

        edge_batch_edges = edges[edge_batch_batch_idc, edge_batch_node_idc, edge_batch_nghb_idc, :]

        # pad up the hidden nodes
        hidden_nodes = torch.zeros(nodes.shape[0], nodes.shape[1], self.hidden_node_features, device="cuda")
        hidden_nodes[:nodes.shape[0], :nodes.shape[1], :nodes.shape[2]] = nodes.clone()
        node_batch_nodes = hidden_nodes[node_batch_batch_idc, node_batch_node_idc, :]

        for _ in range(self.message_passes):
            edge_batch_nodes = hidden_nodes[edge_batch_batch_idc, edge_batch_node_idc, :]

            edge_batch_nghbs = hidden_nodes[edge_batch_batch_idc, edge_batch_nghb_idc, :]

            message_terms = self.message_terms(edge_batch_nodes,
                                               edge_batch_nghbs,
                                               edge_batch_edges)

            if len(message_terms.size()) == 1:  # if a single graph in batch
                message_terms = message_terms.unsqueeze(0)

            # the summation in eq. 1 of the NMPQC paper happens here
            messages = torch.matmul(message_summation_matrix, message_terms)

            node_batch_nodes = self.update(node_batch_nodes, messages)
            hidden_nodes[node_batch_batch_idc, node_batch_node_idc, :] = node_batch_nodes.clone()

        node_mask = adjacency.sum(-1) != 0

        output = self.readout(hidden_nodes, nodes, node_mask)

        return output
