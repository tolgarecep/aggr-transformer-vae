import torch
import torch.nn as nn
import math

def reparam(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

class EncoderBase(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lut = nn.Embedding(args.vocab, args.d_model)
        self.d_model = args.d_model

        self.h2distr = nn.Linear(args.d_model, 2*args.d_z)
    
class TFREncoder(EncoderBase):
    def __init__(self, args):
        super().__init__(args)
        self.stack = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=args.d_model, nhead=args.nhead, dim_feedforward=args.d_ff, 
            dropout=args.dropout, batch_first=True), args.nlayers)
        
    def forward(self, src, src_mask=None):
        x = self.lut(src) * math.sqrt(self.d_model)
        h = self.stack(x, src_mask)
        mu, logvar = self.h2distr(h[:, 0, :]).chunk(2, -1)
        kl_loss = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
        return reparam(mu, logvar).unsqueeze(1), kl_loss.mean(dim=-1), mu