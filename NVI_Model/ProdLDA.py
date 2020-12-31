"""
Adapt from https://github.com/hyqneuron/pytorch-avitm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProdLDA(nn.Module):

    def __init__(self, net_arch, num_input):
        super(ProdLDA, self).__init__()
        ac = net_arch
        self.net_arch = net_arch
        self.num_input = num_input
        # encoder
        self.en1_fc = nn.Linear(self.num_input, ac.en1_units)             # 1995 -> 100
        self.en2_fc = nn.Linear(ac.en1_units, ac.en2_units)             # 100  -> 100
        self.en2_drop = nn.Dropout(0.2)
        self.mean_fc = nn.Linear(ac.en2_units, ac.num_topic)             # 100  -> 50
        self.mean_bn = nn.BatchNorm1d(ac.num_topic)                      # bn for mean
        self.logvar_fc = nn.Linear(ac.en2_units, ac.num_topic)             # 100  -> 50
        self.logvar_bn = nn.BatchNorm1d(ac.num_topic)                      # bn for logvar
        # z
        self.p_drop = nn.Dropout(0.2)
        # decoder
        self.decoder = nn.Linear(ac.num_topic, self.num_input)             # 50   -> 1995
        self.decoder_bn = nn.BatchNorm1d(self.num_input)                      # bn for decoder
        # prior mean and variance as constant buffers
        prior_mean = torch.Tensor(1, ac.num_topic).fill_(0)
        prior_var = torch.Tensor(1, ac.num_topic).fill_(ac.variance)
        prior_logvar = prior_var.log()
        self.register_buffer('prior_mean',    prior_mean)
        self.register_buffer('prior_var',     prior_var)
        self.register_buffer('prior_logvar',  prior_logvar)
        # initialize decoder weight
        if ac.init_mult != 0:
            #std = 1. / math.sqrt( ac.init_mult * (ac.num_topic + ac.num_input))
            self.decoder.weight.data.uniform_(0, ac.init_mult)
        # remove BN's scale parameters
        # self.logvar_bn .register_parameter('weight')
        # self.mean_bn   .register_parameter('weight')
        # self.decoder_bn.register_parameter('weight')
        # self.decoder_bn.register_parameter('weight')

    def forward(self, input, compute_loss=True, avg_loss=False):
        # compute posterior
        en1 = F.softplus(self.en1_fc(input))                            # en1_fc   output
        en2 = F.softplus(self.en2_fc(en1))                              # encoder2 output
        en2 = self.en2_drop(en2)
        posterior_mean = self.mean_bn(self.mean_fc(en2))          # posterior mean
        posterior_logvar = self.logvar_bn(self.logvar_fc(en2))          # posterior log variance
        posterior_var = posterior_logvar.exp()
        # take sample
        eps = input.data.new().resize_as_(posterior_mean.data).normal_() # noise
        z = posterior_mean + posterior_var.sqrt() * eps                 # reparameterization
        p = F.softmax(z, dim=-1)                                                # mixture probability
        p = self.p_drop(p)
        # do reconstruction
        recon = F.softmax(self.decoder_bn(self.decoder(p)), dim=-1)             # reconstructed distribution over vocabulary

        if compute_loss:
            return p, self.loss(input, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss)
        else:
            return p

    def loss(self, input, recon, posterior_mean, posterior_logvar, posterior_var, avg=True):
        # NL
        NL = -(input * (recon+1e-10).log()).sum(1)
        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017,
        # https://arxiv.org/pdf/1703.01488.pdf
        prior_mean = self.prior_mean.expand_as(posterior_mean)
        prior_var = self.prior_var.expand_as(posterior_mean)
        prior_logvar = self.prior_logvar.expand_as(posterior_mean)
        var_division = posterior_var / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.net_arch.num_topic)
        # loss
        loss = (NL + KLD)
        # in training mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean()
        else:
            return loss


# parser = argparse.ArgumentParser()
# parser.add_argument('-f', '--en1-units',        type=int,   default=100)
# parser.add_argument('-s', '--en2-units',        type=int,   default=100)
# parser.add_argument('-t', '--num-topic',        type=int,   default=50)
# parser.add_argument('-b', '--batch-size',       type=int,   default=200)
# parser.add_argument('-o', '--optimizer',        type=str,   default='Adam')
# parser.add_argument('-r', '--learning-rate',    type=float, default=0.002)
# parser.add_argument('-m', '--momentum',         type=float, default=0.99)
# parser.add_argument('-e', '--num-epoch',        type=int,   default=80)
# parser.add_argument('-q', '--init-mult',        type=float, default=1.0)    # multiplier in initialization of decoder weight
# parser.add_argument('-v', '--variance',         type=float, default=0.995)  # default variance in prior normal
# parser.add_argument('--start',                  action='store_true')        # start training at invocation
# parser.add_argument('--nogpu',                  action='store_true')
#
# args = parser.parse_args()
#
# args.num_input = 10000
#
# prod_lda = ProdLDA(args)
