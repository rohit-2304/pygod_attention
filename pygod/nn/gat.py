import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class MultiLayerGATv2(nn.Module):
    """
    Multi-layer GATv2 backbone for encoder/decoder.
    
    This wrapper stacks multiple GATv2Conv layers to create a deep
    graph attention network, similar to how PyG's GCN model works.
    
    Parameters
    ----------
    in_channels : int
        Input feature dimension.
    hidden_channels : int
        Hidden layer dimension.
    out_channels : int
        Output feature dimension.
    num_layers : int
        Number of GATv2 layers to stack.
    heads : int, optional
        Number of attention heads. Default: ``1``.
    concat : bool, optional
        Whether to concatenate attention heads in the output layer.
        Intermediate layers always concatenate. Default: ``True``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None. Default: ``torch.nn.functional.relu``.
    add_self_loops : bool, optional
        Whether to add self-loops to the graph. Default: ``True``.
    """
    
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 num_layers,
                 heads=1, 
                 concat=True, 
                 dropout=0., 
                 act=torch.nn.functional.relu,
                 add_self_loops=True):
        super(MultiLayerGATv2, self).__init__()
        
        self.num_layers = num_layers
        self.act = act
        self.dropout = dropout
        
        # ModuleList to hold all GATv2Conv layers
        self.convs = nn.ModuleList()
        
        if num_layers == 1:
            # Single layer: directly map input to output
            self.convs.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=heads,
                    concat=concat,
                    dropout=dropout,
                    add_self_loops=add_self_loops
                )
            )
        else:
            # Multiple layers
            
            # First layer: in_channels -> hidden_channels
            self.convs.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    heads=heads,
                    concat=True,  # Always concatenate heads in intermediate layers
                    dropout=dropout,
                    add_self_loops=add_self_loops
                )
            )
            
            # Middle layers: (hidden_channels * heads) -> hidden_channels
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATv2Conv(
                        in_channels=hidden_channels * heads,
                        out_channels=hidden_channels,
                        heads=heads,
                        concat=True,
                        dropout=dropout,
                        add_self_loops=add_self_loops
                    )
                )
            
            # Last layer: (hidden_channels * heads) -> out_channels
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_channels * heads,
                    out_channels=out_channels,
                    heads=heads,
                    concat=concat,  # Use the specified concat for output
                    dropout=dropout,
                    add_self_loops=add_self_loops
                )
            )
    
    def forward(self, x, edge_index):
        """
        Forward pass through all GATv2 layers.
        
        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, in_channels].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].
        
        Returns
        -------
        torch.Tensor
            Output node embeddings of shape [num_nodes, out_channels * heads]
            if concat=True, else [num_nodes, out_channels].
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            # Apply activation and dropout to all layers except the last
            if i < self.num_layers - 1:
                if self.act is not None:
                    x = self.act(x)
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        
        return x