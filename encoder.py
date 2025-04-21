import torch
from torch import nn
from torch.nn import functional as F
from sd.decoder import VAE_AttentionBlock, VAE_ResidualBlock
class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # (batch_size, channel , height , width)  --> (batch_size, 128, height, width) (same as input h, w)
            nn.Conv2d(in_channels=3, out_channels=128 , kernel_size=3, padding=1),
            # (bs, channel, height, w) --> (bs, channel, h, w)
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            # (bs,  128, h, w ) --> (bs, 128, h/2 , w/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0 ),
            # (bs, 128, h/2, w/2) --> (bs, 256, h/2, w/2)
            VAE_ResidualBlock(128, 256),
            # (bs, 256, h/2, w/2) --> (bs, 256, h/2, w/2)
            VAE_ResidualBlock(256, 256),
            # (bs, 256, h/2, w/2) --> (bs, 256, h/4, w/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0 ),
            # (bs, 256, h/4, w/4) --> (bs, 512, h/4, w/4)
            VAE_ResidualBlock(256, 512),
            # (bs, 512, h/4, w/4) --> (bs, 512, h/4, w/4)
            VAE_ResidualBlock(512, 512),
            # (bs, 512, h/4, w/4) --> (bs, 512, h/8, w/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0 ),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (bs, 512, h/8, w/8) -->  (bs, 512, h/8, w/8)
            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            # (bs, 512, h/8, w/8) -->  (bs, 512, h/8, w/8)
            VAE_ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),  # no change in shape
            nn.SiLU(),
            # (bs, 512, h/8, w/8) --> (bs, 8, h/8, w/8)
            nn.Conv2d(512, 8, kernal_size=3, padding=1), 
            nn.Conv2d(8,8, kernal_size=1, padding=0)
        
        )
    def forward(self, x:torch.Tensor , noise: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channel, height, width)
        # noise: (batch_size, out_channels, heigth/8, width/8 )
        for module in self: # traverse all modules
            if getattr(module, 'stride' , None) == (2,2):
                #(padding left, padding_right, Paddng_top, paddng_bottom)
                x = F.pad(x, (0,1, 0, 1))
            x = module(x)

        # (bs, 8, h/8, w/8) --> two tensors of shpae (bs, 4 , h/8, w/8)
        mean, log_variance = torch.chunk(x, 2, dim = 1)
        variance = log_variance.exp()
        stdev = variance.sqrt()

        # N(0, 1) --> N(mean, variance) 
        x = mean + stdev*noise

        #scale the output by a constant, unkown reason
        x *= 0.18215

        
    