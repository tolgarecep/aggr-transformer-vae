import torch
import pickle

def logging(s, path, print_=True):
    if print_:
        print(s)
    if path:
        with open(path, 'a+') as f:
            f.write(s + '\n')

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8)
    return subsequent_mask == 0

def make_std_mask(tgt, pad):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask

def plot_multiple(args, model, plot_data, grid_z, iter_):
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