import math
import torch
import torch.nn as nn
from torch_geometric.nn import GCN, GATv2Conv
from torch_geometric.utils import to_dense_adj
from gat import MultiLayerGATv2
from .decoder import DotProductDecoder
from .functional import double_recon_loss


class DOMINANTBase(nn.Module):
    """
    Deep Anomaly Detection on Attributed Networks (DOMINANT)
    
    DOMINANT is an anomaly detector consisting of a shared graph
    convolutional encoder, a structure reconstruction decoder, and an
    attribute reconstruction decoder. The reconstruction mean squared
    error of the decoders are defined as structure anomaly score and
    attribute anomaly score, respectively.
    
    This implementation supports both GCN and GATv2 backbones.
    
    See :cite:`ding2019deep` for details.
    
    Parameters
    ----------
    in_dim : int
        Input dimension of model (number of node features).
    hid_dim : int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. A half (floor) of the layers
        are for the encoder, the other half (ceil) of the layers are
        for decoders. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    sigmoid_s : bool, optional
        Whether to apply sigmoid to the structure reconstruction.
        Default: ``False``.
    backbone : torch.nn.Module, optional
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    use_gat : bool, optional
        Whether to use GATv2 attention mechanism instead of GCN.
        If True, the backbone parameter is ignored for encoder/decoder.
        Default: ``False``.
    heads : int, optional
        Number of attention heads for GATv2 (only used if use_gat=True).
        Default: ``4``.
    concat : bool, optional
        Whether to concatenate attention heads in output layers
        (only used if use_gat=True). Default: ``False``.
    add_self_loops : bool, optional
        Whether to add self-loops in GATv2 layers
        (only used if use_gat=True). Default: ``True``.
    **kwargs : optional
        Additional arguments for the GCN backbone.
    """
    
    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 sigmoid_s=False,
                 backbone=GCN,
                 use_gat=False,
                 heads=4,
                 concat=False,
                 add_self_loops=True,
                 **kwargs):
        super(DOMINANTBase, self).__init__()
        
        # Split the number of layers for the encoder and decoders
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)
        
        self.use_gat = use_gat
        self.heads = heads
        
        # Build encoder and decoder based on backbone choice
        if use_gat:
            print(f"Using GATv2 with {heads} attention heads")
            
            # Shared Encoder: in_dim -> hid_dim
            self.shared_encoder = MultiLayerGATv2(
                in_channels=in_dim,
                hidden_channels=hid_dim,
                out_channels=hid_dim,
                num_layers=encoder_layers,
                heads=heads,
                concat=concat,  # Use specified concat mode
                dropout=dropout,
                act=act,
                add_self_loops=add_self_loops
            )
            
            # Calculate actual encoder output dimension
            encoder_out_dim = hid_dim * heads if concat else hid_dim
            
            # Attribute Decoder: encoder_out_dim -> in_dim
            self.attr_decoder = MultiLayerGATv2(
                in_channels=encoder_out_dim,
                hidden_channels=hid_dim,
                out_channels=in_dim,
                num_layers=decoder_layers,
                heads=heads,
                concat=False,  # Always False for final reconstruction
                dropout=dropout,
                act=act,
                add_self_loops=add_self_loops
            )
            
            # Structure decoder input dimension
            struct_decoder_in_dim = encoder_out_dim
            struct_decoder_backbone = GCN  # Use GCN for structure decoder
            
        else:
            print("Using GCN backbone")
            
            # Shared Encoder using GCN
            self.shared_encoder = backbone(
                in_channels=in_dim,
                hidden_channels=hid_dim,
                num_layers=encoder_layers,
                out_channels=hid_dim,
                dropout=dropout,
                act=act,
                **kwargs
            )
            
            # Attribute Decoder using GCN
            self.attr_decoder = backbone(
                in_channels=hid_dim,
                hidden_channels=hid_dim,
                num_layers=decoder_layers,
                out_channels=in_dim,
                dropout=dropout,
                act=act,
                **kwargs
            )
            
            struct_decoder_in_dim = hid_dim
            struct_decoder_backbone = backbone
        
        # Structure Decoder (always uses simpler backbone for dot product)
        self.struct_decoder = DotProductDecoder(
            in_dim=struct_decoder_in_dim,
            hid_dim=hid_dim,
            num_layers=decoder_layers - 1,
            dropout=dropout,
            act=act,
            sigmoid_s=sigmoid_s,
            backbone=struct_decoder_backbone,
            **kwargs
        )
        
        self.loss_func = double_recon_loss
        self.emb = None
    
    def forward(self, x, edge_index):
        """
        Forward computation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings of shape [num_nodes, in_dim].
        edge_index : torch.Tensor
            Edge index of shape [2, num_edges].
        
        Returns
        -------
        x_ : torch.Tensor
            Reconstructed attribute embeddings of shape [num_nodes, in_dim].
        s_ : torch.Tensor
            Reconstructed adjacency matrix of shape [num_nodes, num_nodes].
        """
        # Encode: learn node embeddings
        self.emb = self.shared_encoder(x, edge_index)
        
        # Decode: reconstruct node features
        x_ = self.attr_decoder(self.emb, edge_index)
        
        # Decode: reconstruct graph structure
        s_ = self.struct_decoder(self.emb, edge_index)
        
        return x_, s_
    
    @staticmethod
    def process_graph(data):
        """
        Obtain the dense adjacency matrix of the graph.
        
        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        """
        data.s = to_dense_adj(data.edge_index)[0]