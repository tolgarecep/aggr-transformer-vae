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
        self.d_model = args.d_model
        self.lut = nn.Embedding(args.vocab, args.d_model)
        self.h2distr = nn.Linear(args.d_model, 2*args.d_z)

    def calc_mi(self, src):
        """ returns mutual information between x and z
        I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z)) """
        z, _, mu, logvar = self.forward(src, None)
        bsz, d_z = mu.size()
        neg_entropy = (-0.5 * d_z * math.log(
            2 * math.pi) - 0.5 * (1 + logvar).sum(-1)).mean()
        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()
        dev = z - mu
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (d_z * math.log(2 * math.pi) + logvar.sum(-1))
        log_qz = log_density.exp().sum(1, False).log() - math.log(bsz)
        return (neg_entropy - log_qz.mean(-1)).item()

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
        return reparam(mu, logvar).unsqueeze(1), \
               kl_loss.mean(dim=-1), \
               mu, logvar