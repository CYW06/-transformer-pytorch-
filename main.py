import numpy as np
import torch
from torch import nn

class SublayerConnection(nn.Module):

    def __init__(self, size,dropout=0.1) -> None:
        super(SublayerConnection,self).__init__()
        self.layer_norm = LayNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x,sublayer):
        return self.dropout(self.layer_norm(x+sublayer(x)))
    
class LayNorm(nn.Module):
    
    def __init__(self,feature,eps=1e-6) -> None:

        super(LayNorm,self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self,x):

        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)

        return ((x-mean)/(std+self.eps))*self.a_2+self.b_2

if __name__ == '__main__':
    
    A = SublayerConnection(2)
