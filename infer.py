import torch
import argparse

from modules.encoders import TFREncoder
from modules.decoders import TFRDecoder
from modules.vae import VAE
from utils import logging, subsequent_mask

parser = argparse.ArgumentParser()
parser.add_argument('--vocab', default=100)
parser.add_argument('--d_model', default=32)
parser.add_argument('--d_z', default=16)
parser.add_argument('--dropout', default=0.1)
parser.add_argument('--nhead', default=1)
parser.add_argument('--d_ff', default=128)
parser.add_argument('--nlayers', default=4)

def main(args):
    log_file = 'log.txt'
    logging(str(args), log_file)

    model = VAE(args, TFREncoder(args), TFRDecoder(args))
    model.load_state_dict(torch.load('./model/model.pt'))
    model.eval()

    # src = torch.LongTensor([[0,1,2,3,4,5,6,7,8,9]]) # 1, 10
    src = torch.LongTensor([[0, 91, 67, 72, 87]])
    src_mask = torch.ones(1, src.size(1), src.size(1)) # 1, 10, 10

    z = model.encode(src, src_mask)[0]
    ys = torch.zeros(1, 1).type_as(src)

    for _ in range(4):
        logits = model.decode(ys, z)
        prob = logits[:, -1]
        _, next = torch.max(prob, dim=1)
        next = next.data[0]
        ys = torch.cat([ys, torch.empty(1, 1).type_as(src.data).fill_(next)], dim=1)
        print(ys)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)