import torch
import torch.nn as nn
    
class VAE(nn.Module):
    def __init__(self, args, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.encoder.lut.weight = self.decoder.lut.weight
        # self.gen.proj.weight = self.embed.lut.weight

        loc = torch.zeros(args.d_z)
        scale = torch.ones(args.d_z)
        self.prior = torch.distributions.normal.Normal(loc, scale)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        z, kl_loss, _ = self.encode(src, src_mask)
        return self.decode(tgt, z, tgt_mask, src_mask), kl_loss
    def encode(self, src, src_mask=None):
        return self.encoder(src, src_mask)
    def decode(self, tgt, z, tgt_mask=None, mem_mask=None):
        return self.decoder(tgt, z, tgt_mask, mem_mask)

    def calc_inference_mean(self, plot_data):
        return self.encoder(plot_data, None)[2]

    def calc_posterior_mean(self, plot_data, grid_z):
        """returns E_{z ~ p(z|x)}[z] = Σ_{z_i ∈ C}[z_i * p(z_i|x)]"""
        log_post = self.calc_log_posterior(plot_data, grid_z)
        post = log_post.exp()
        return torch.mul(post.unsqueeze(2), grid_z.unsqueeze(0)).sum(1)

    def calc_log_posterior(self, plot_data, grid_z):
        bsz = plot_data.size(0)
        grid_z = grid_z.unsqueeze(0).repeat(bsz, 1, 1)
        log_joint = self.calc_joint(plot_data, grid_z)
        log_norm = log_joint.exp().sum(dim=1, keepdim=True).log()
        # normalizing value for each sample in batch
        log_post = log_joint - log_norm
        return log_post

    def calc_joint(self, plot_data, grid_z):
        log_prior = self.prior.log_prob(grid_z).sum(dim=-1)
        log_gen = self.decoder.eval_likeli(plot_data, grid_z)
        return log_prior + log_gen