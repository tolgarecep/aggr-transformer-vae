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

    def eval_likeli(self, plot_data, grid_z):
        raise NotImplementedError

class TFRDecoder(DecoderBase):
    def __init__(self, args):
        super().__init__(args)
        self.stack = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=args.d_model, nhead=args.nhead, dim_feedforward=args.d_ff, 
            dropout=args.dropout, batch_first=True), args.nlayers)
        
    def forward(self, tgt, z, tgt_mask=None, mem_mask=None):
        x = self.lut(tgt) * math.sqrt(self.d_model)
        m = self.z2mem(z).unsqueeze(1).repeat(1, tgt.size(1) + 1, 1)
        out = self.stack(x, m, tgt_mask, mem_mask)
        return self.proj(out)