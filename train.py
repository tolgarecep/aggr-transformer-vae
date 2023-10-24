import torch
import torch.nn as nn
from torch import optim
import os
import argparse

from modules.encoders import TFREncoder
from modules.decoders import TFRDecoder
from modules.vae import VAE
from utils import logging, make_std_mask, plot_multiple, generate_grid, subsequent_mask

parser = argparse.ArgumentParser()
parser.add_argument('--train')
parser.add_argument('--val')
parser.add_argument('--test')
parser.add_argument('--save', default='./model/')
parser.add_argument('--bsz', default=60)

parser.add_argument('--epoch', default=1)
parser.add_argument('--log_niter', default=25)
parser.add_argument('--plot_niter', default=25)
parser.add_argument('--optim', default='adam')

parser.add_argument('--aggr', default=False)
parser.add_argument('--zmin', default=-20.0)
parser.add_argument('--zmax', default=20.0)
parser.add_argument('--gran', default=0.1)
parser.add_argument('--nplot', default=5000)

parser.add_argument('--vocab', default=50)
parser.add_argument('--d_model', default=32)
parser.add_argument('--d_z', default=16)
parser.add_argument('--dropout', default=0.1)
parser.add_argument('--nhead', default=1)
parser.add_argument('--d_ff', default=128)
parser.add_argument('--nlayers', default=4)

# def calc_mi(model, src_batch_val):
#     mi = 0
#     nsample = 0
#     for src in src_batch_val: Chunk this in batches
#         bsz = src.size(0)
#         nsample += bsz
#         mutual_info = model.encoder.calc_mi(src)
#         mi += mutual_info * bsz
#     return mi / nsample

"""batch_first=True"""
"""Training and validation"""
L = 5
pad = 0
size_tr = 10000
size_val = 1000

data_tr = torch.randint(1, 50, size=(size_tr, L))
data_val = torch.randint(1, 50, size=(size_val, L))
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

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    log_file = os.path.join(args.save, 'log.txt')
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

    aggr_flag = True if args.aggr else False
    pre_mi = -1
    for epoch in range(args.epoch):
        # calc_mi(vae, src_batch_val)
        vae(src_batch_val, tgt_batch_val, None, tgt_mask_val)
        for i in range(size_tr // args.bsz):
            src = src_batch_tr[i*10:(i+1)*10]
            tgt = tgt_batch_tr[i*10:(i+1)*10]
            # src_mask_mini = src_mask[i*100:(i+1)*100]
            tgt_mask_mini = tgt_mask_tr[i*10:(i+1)*10]
            y = y_batch_tr[i*10:(i+1)*10]

            i_aggr = 0
            pre = 1e4
            cur = 0
            nword = 0
            while aggr_flag and i_aggr < 100:
                vae.train()
                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()

                logits, kl_loss = vae(src, tgt, None, tgt_mask_mini)
                rec_loss = crit(logits.reshape(-1, logits.size(2)), y.reshape(-1))
                loss = rec_loss + kl_loss
                cur = loss.item()
                loss.backward()
                enc_optimizer.step()

                # should I break the inner loop in this epoch?
                nword = src.size(0) * (src.size(1) - 1)
                if i_aggr % 15 == 0:
                    cur = cur / nword
                    if pre - cur < 0:
                        break
                    pre = cur
                    cur = nword = 0
                i_aggr += 1

            vae.train()
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            logits, kl_loss = vae(src, tgt, None, tgt_mask_mini)
            rec_loss = crit(logits.reshape(-1, logits.size(2)), y.reshape(-1))
            loss = rec_loss + kl_loss
            loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            if i % args.log_niter == 0:
                vae.eval()
                logits_val, kl_loss_val = vae(src_batch_val, tgt_batch_val, None, tgt_mask_val)
                rec_loss_val = crit(logits_val.reshape(-1, logits.size(2)), 
                                    y_batch_val.reshape(-1))
                loss_val = kl_loss_val + rec_loss_val
                logging('Epoch %d, iteration %d: %.4f / %.4f' % (
                    epoch, i, loss.item(), loss_val.item()), log_file)
            if i % args.plot_niter == 0:
                vae.eval()
                with torch.no_grad():
                    plot_multiple(args, vae, plot_data, grid_z, i)
                logging(f'Statistics at this point are saved at ./model/space{i}.pickle', log_file)
        # if aggr_flag:
        #     vae.eval()
        #     cur_mi = calc_mi(vae, src_batch_val)
        #     vae.train()
        #     if cur_mi - pre_mi < 0:
        #         aggr_flag = False
        #         logging(f'No more aggressive training (lasted {epoch} epochs)', log_file)
        #     pre_mi = cur_mi
    torch.save(vae.state_dict(), os.path.join(args.save, 'model.pt'))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)