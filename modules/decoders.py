import torch
import torch.nn as nn
import math

class DecoderBase(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lut = nn.Embedding(args.vocab, args.d_model)
        self.d_model = args.d_model
        self.z2mem = nn.Linear(args.d_z, args.d_model)
        self.proj = nn.Linear(args.d_model, args.vocab)
        self.d_z = args.d_z
        self.crit = nn.CrossEntropyLoss()

    def eval_likeli(self, src, z):
        tgt = src[:, :-1]
        y = src[:, 1:]
        bsz, nsample, _ = z.size()
        logits = self.forward(tgt, z)
        loss = self.crit(logits.reshape(-1, logits.size(2)), y.reshape(-1))
        return loss.view(bsz, nsample, -1).sum(-1)

class TFRDecoder(DecoderBase):
    def __init__(self, args):
        super().__init__(args)
        self.stack = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=args.d_model, nhead=args.nhead, dim_feedforward=args.d_ff, 
            dropout=args.dropout, batch_first=True), args.nlayers)
        
    def forward(self, tgt, z, tgt_mask=None, mem_mask=None):
        bsz, L = tgt.size()
        _, nsample, d_z = z.size()

        x = self.lut(tgt) * math.sqrt(self.d_model)
        if nsample != 1:
            x = x.unsqueeze(1).repeat(1, nsample, 1, 1).view(
                bsz * nsample, L, self.d_model)
        z = z.view(bsz * nsample, d_z)
        m = self.z2mem(z).unsqueeze(1).repeat(1, L + 1, 1)
        out = self.stack(x, m, tgt_mask, mem_mask)
        return self.proj(out)