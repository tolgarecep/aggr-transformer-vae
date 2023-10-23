import torch
import torch.nn as nn
from torch import optim
import os
import argparse
import pickle

from modules.encoders import TFREncoder
from modules.decoders import TFRDecoder
from modules.vae import VAE
from utils import logging, subsequent_mask, make_std_mask

parser = argparse.ArgumentParser()
parser.add_argument('--train')
parser.add_argument('--val')
parser.add_argument('--test')
parser.add_argument('--save', default='./model')
parser.add_argument('--bsz', default=100)

parser.add_argument('--epoch', default=1)
parser.add_argument('--log_niter', default=250)
parser.add_argument('--plot_niter', default=250)
parser.add_argument('--optim', default='adam')

parser.add_argument('--aggr', default=False)
parser.add_argument('--zmin', default=-20.0)
parser.add_argument('--zmax', default=20.0)
parser.add_argument('--gran', default=0.1)
parser.add_argument('--nplot', default=5000)

parser.add_argument('--vocab', default=100)
parser.add_argument('--d_model', default=32)
parser.add_argument('--d_z', default=16)
parser.add_argument('--dropout', default=0.1)
parser.add_argument('--nhead', default=1)
parser.add_argument('--d_ff', default=128)
parser.add_argument('--nlayers', default=4)

def plot_multiple(model, plot_data, grid_z, iter_):
    plot_data_tuple = torch.chunk(plot_data, round(args.nplot / args.bsz))
    qp_means = []
    for plot_data in plot_data_tuple:
        mu_p = model.calc_posterior_mean(plot_data, grid_z)
        mu_q = model.calc_inference_mean(plot_data)
        qp_means.append(torch.cat([mu_p, mu_q], 1))
    qp_means = torch.cat(qp_means, 0)
    save_path = './model/space' + str(iter_) + '.pickle'
    save_data = {'posterior': qp_means[:, 0].cpu().numpy(),
                 'inference': qp_means[:, 1].cpu().numpy()}
    pickle.dump(save_data, open(save_path, 'wb'))

def generate_grid(zmin, zmax, gran):
    return torch.arange(zmin, zmax, gran).unsqueeze(1)

"""batch_first=True"""

"""Training and validation"""
L = 5
pad = 0
size_tr = 100000
size_val = 10000 # %10

data_tr = torch.randint(1, 100, size=(size_tr, L))
data_val = torch.randint(1, 100, size=(size_val, L))
data_tr[:, 0] = 1
data_val[:, 0] = 1

src_batch_tr = data_tr.requires_grad_(False).clone().detach()
tgt_data_tr = data_tr.requires_grad_(False).clone().detach()
src_batch_val = data_val.requires_grad_(False).clone().detach()
tgt_data_val = data_val.requires_grad_(False).clone().detach()

tgt_batch_tr = tgt_data_tr[:, :-1]
tgt_batch_val = tgt_data_val[:, :-1]
y_batch_tr = tgt_data_tr[:, 1:]
y_batch_val = tgt_data_val[:, 1:]

tgt_mask_tr = ~make_std_mask(tgt_batch_tr, pad)
tgt_mask_val = ~make_std_mask(tgt_batch_val, pad)
# src_mask = (src_batch != pad).unsqueeze(-2)
# src_mask = torch.ones(bsz, 1, L)

def main(args):
    args.d_z = 1
    # umm...

    """Grid and data to plot posterior mean space"""
    grid_z = generate_grid(args.zmin, args.zmax, args.gran)
    plot_data_idx = torch.randint(data_tr.size(0), (args.nplot,))
    plot_data = src_batch_tr[plot_data_idx]

    # log_file = os.path.join(args.save, 'log.txt')
    log_file = './model/log.txt'
    logging(str(args), log_file)

    # encoder = TFREncoder(args)
    # decoder = TFRDecoder(args)
    vae = VAE(args, TFREncoder(args), TFRDecoder(args))

    if args.optim == 'sgd':
        enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=1.0)
        dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=1.0)
    elif args.optim == 'adam':
        enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=0.001, betas=(0.9, 0.999))
        dec_optimizer = optim.Adam(vae.decoder.parameters(), lr=0.001, betas=(0.9, 0.999))
    crit = nn.CrossEntropyLoss()

    for epoch in range(args.epoch):
        for i in range(size_tr // args.bsz):
            vae.train()
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            src = src_batch_tr[i*100:(i+1)*100]
            tgt = tgt_batch_tr[i*100:(i+1)*100]
            # src_mask_mini = src_mask[i*100:(i+1)*100]
            tgt_mask_mini = tgt_mask_tr[i*100:(i+1)*100]
            y = y_batch_tr[i*100:(i+1)*100]

            logits, kl_loss = vae(src, tgt, None, tgt_mask_mini)
            rec_loss = crit(logits.reshape(-1, logits.size(2)), y.reshape(-1))
            loss = rec_loss + kl_loss
            loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            if i % args.log_niter == 0:
                vae.eval()
                logits_val, kl_loss_val = vae(src_batch_val, tgt_batch_val, None, tgt_mask_val)
                rec_loss_val = crit(logits_val.reshape(-1, logits.size(2)), y_batch_val.reshape(-1))
                loss_val = kl_loss_val + rec_loss_val
                logging(f'Training loss at epoch {epoch}, iteration {i}: {loss.item()}', log_file)
                logging(f'Validation loss at epoch {epoch}: {loss_val.item()}', log_file)
            if i % args.plot_niter == 0:
                vae.eval()
                with torch.no_grad():
                    plot_multiple(vae, plot_data, grid_z, i)
    # save_path = os.path.join(args.save, 'model.pt')
    torch.save(vae.state_dict(), './model/model.pt')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)