"""CSTVT: CNN + Super Token Vision Transformer.

A five-stage image classifier. Stages 1 and 2 are convolutional; stages 3 and 4
replace ordinary spatial attention with super-token attention. Stage width,
depth of the transformer stages, the super-token sampling grid and the Class
Token insertion point are all constructor parameters.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ConvBNGELU(nn.Sequential):
    def __init__(self, incoming, outgoing, kernel, stride):
        super().__init__(nn.Conv2d(incoming, outgoing, kernel, stride,
                                   padding=kernel//2, bias=True),
                         nn.BatchNorm2d(outgoing), nn.GELU())


class SuperTokenSampler(nn.Module):
    """Dense, differentiable manuscript association and iterative aggregation."""
    def __init__(self, grid_side=10, iterations=2):
        super().__init__()
        self.grid_side = grid_side
        self.iterations = iterations

    def forward(self, features):
        tokens = features.flatten(2).transpose(1,2)
        super_tokens = F.adaptive_avg_pool2d(features, (self.grid_side,self.grid_side)).flatten(2).transpose(1,2)
        for _ in range(self.iterations):
            association = (tokens @ super_tokens.transpose(-1,-2) / math.sqrt(tokens.shape[-1])).float().softmax(-1)
            mass = association.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
            super_tokens = (association.transpose(1,2).to(tokens.dtype) @ tokens) / mass.to(tokens.dtype)
        return super_tokens, association


class SuperTokenMHSA(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width,3*width,bias=True)
        self.projection = nn.Linear(width,width,bias=True)

    def forward(self, tokens):
        batch,count,width = tokens.shape
        q,k,v = self.qkv(tokens).reshape(batch,count,3,self.heads,width//self.heads).permute(2,0,3,1,4).unbind(0)
        attention = F.scaled_dot_product_attention(q,k,v,dropout_p=0.0)
        return self.projection(attention.transpose(1,2).reshape(batch,count,width))


class ManuscriptSTBlock(nn.Module):
    def __init__(self, width=128, heads=4, grid_side=10, iterations=2, expansion=2):
        super().__init__()
        self.cpe = nn.Conv2d(width,width,3,padding=1,groups=width,bias=True)
        self.norm = nn.LayerNorm(width,eps=1e-6)
        self.sts = SuperTokenSampler(grid_side,iterations)
        self.mhsa = SuperTokenMHSA(width,heads)
        self.ffn = nn.Sequential(
            nn.BatchNorm2d(width), nn.Conv2d(width,width*expansion,1,bias=True),
            nn.GELU(), nn.Conv2d(width*expansion,width*expansion,3,padding=1,
                                groups=width*expansion,bias=True),
            nn.Conv2d(width*expansion,width,1,bias=True))

    def forward(self, features, class_token):
        position_enhanced = features + self.cpe(features)
        normalized = self.norm(position_enhanced.permute(0,2,3,1)).permute(0,3,1,2)
        super_tokens,association = self.sts(normalized)
        # Class propagation is an explicit implementation choice where the
        # manuscript specifies a Class Token but omits its insertion location.
        enhanced = self.mhsa(torch.cat([class_token,super_tokens],dim=1))
        class_token = class_token + enhanced[:,:1]
        remapped = association.to(enhanced.dtype) @ enhanced[:,1:]
        remapped = remapped.transpose(1,2).reshape_as(position_enhanced)
        visual = position_enhanced + remapped
        return visual + self.ffn(visual), class_token


class ManuscriptCSTVT(nn.Module):
    def __init__(self,image_size=160,channels=(16,32,64,128),depth=2,heads=4,
                 grid_side=10,iterations=2,expansion=2,num_classes=5):
        super().__init__()
        assert image_size%4==0 and channels[-1]%heads==0
        self.image_size = image_size
        self.settings = dict(image_size=image_size,channels=list(channels),depth=depth,
                             heads=heads,grid_side=grid_side,iterations=iterations,
                             expansion=expansion,num_classes=num_classes)
        self.embedding = nn.Sequential(
            ConvBNGELU(3,channels[0],3,1),
            ConvBNGELU(channels[0],channels[1],3,2),
            ConvBNGELU(channels[1],channels[2],3,2),
            ConvBNGELU(channels[2],channels[3],1,1))
        width = channels[-1]
        self.absolute_position = nn.Parameter(torch.zeros(1,(image_size//4)**2,width))
        self.class_token = nn.Parameter(torch.zeros(1,1,width))
        self.blocks = nn.ModuleList([ManuscriptSTBlock(width,heads,grid_side,iterations,expansion)
                                     for _ in range(depth)])
        self.class_norm = nn.LayerNorm(width,eps=1e-6)
        self.head = nn.Linear(width,num_classes,bias=True)
        nn.init.trunc_normal_(self.absolute_position,std=0.02)
        nn.init.trunc_normal_(self.class_token,std=0.02)

    def forward(self,images):
        if images.shape[1:] != (3,self.image_size,self.image_size):
            raise ValueError('Input shape differs from fixed learned-position grid')
        features = self.embedding(images)
        tokens = features.flatten(2).transpose(1,2) + self.absolute_position
        features = tokens.transpose(1,2).reshape_as(features)
        cls = self.class_token.expand(images.shape[0],-1,-1)
        for block in self.blocks:
            features,cls = block(features,cls)
        return self.head(self.class_norm(cls[:,0]))
