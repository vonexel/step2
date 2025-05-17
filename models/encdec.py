import torch.nn as nn
from models.resnet import Resnet1D


# The encoder progressively reduces the temporal dimension of the input data while the decoder reverses this process to reconstruct the original input.


class PrintModule(nn.Module):
    """
    Utility module for debugging that prints the shape of tensors as they pass through the network.
    """
    def __init__(self, me=''):
        super().__init__()
        self.me = me

    def forward(self, x):
        print(self.me, x.shape)
        return x
    
class Encoder(nn.Module):
    """
    Module that transforms input data into a latent representation by:
    ResNet1D processes motion features after downsampling:

    Encoder ->
    -> Conv1d (processes the input) ->
    -> Downsampling (applying a series of downsampling blocks (specified by down_t parameter)) ->
    -> ResNet1D (each downsampling block uses strided convolutions and residual 1D networks) ->
    -> Output embeddings (finishing with a final convolution to reach the desired output embedding width).
    """
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()
        
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    """
    Module that reconstructs data from the latent representation by:
    ResNet1D with reverse dilation is used before upsampling:

    Decoder ->
    -> Conv1d (starting with a 1D convolution on the latent representation) ->
    -> ResNet1D (reverse_dilation) ->
    -> Upsampling (applying a series of upsampling blocks (matching the down_t parameter) ->
    -> Output motion (finishing with additional convolutions to produce the final output).
    """
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)