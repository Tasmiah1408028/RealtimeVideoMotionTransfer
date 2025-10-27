# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Normal, kl_divergence


# # =============== Config Dict ===============
# class ConfigDict(dict):
#     def __getattr__(self, name):
#         try:
#             return self[name]
#         except KeyError:
#             raise AttributeError(name)

#     def __setattr__(self, name, value):
#         self[name] = value

#     def get(self, key, default=None):
#         return super().get(key, default)


# # =============== Config ===============
# def get_config():
#     cfg = ConfigDict()

#     # Training
#     cfg.batch_size = 32
#     cfg.steps_per_epoch = 50
#     cfg.num_epochs = 15
#     cfg.learning_rate = 0.001
#     cfg.clipnorm = 10

#     # Sequence
#     cfg.observed_steps = 6
#     cfg.predicted_steps = 6
#     cfg.num_keypoints = 10

#     # Dynamics
#     cfg.num_rnn_units = 256
#     cfg.prior_net_dim = 128
#     cfg.posterior_net_dim = 128
#     cfg.latent_code_size = 20
#     cfg.kl_loss_scale = 0.0001
#     cfg.kl_annealing_steps = 1000
#     cfg.use_deterministic_belief = False
#     cfg.num_samples_for_bom = 10

#     # Scheduled Sampling
#     cfg.scheduled_sampling_ramp_steps = cfg.steps_per_epoch * int(cfg.num_epochs * 0.8)
#     cfg.scheduled_sampling_p_true_start_obs = 1.0
#     cfg.scheduled_sampling_p_true_end_obs = 0.1
#     cfg.scheduled_sampling_p_true_start_pred = 1.0
#     cfg.scheduled_sampling_p_true_end_pred = 0.5

#     return cfg


# # =============== Prior Net ===============
# class PriorNet(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.fc1 = nn.Linear(cfg.num_rnn_units, cfg.prior_net_dim)
#         self.means = nn.Linear(cfg.prior_net_dim, cfg.latent_code_size)
#         self.stds = nn.Linear(cfg.prior_net_dim, cfg.latent_code_size)

#     def forward(self, rnn_state):
#         hidden = F.relu(self.fc1(rnn_state))
#         mean = self.means(hidden)
#         std = F.softplus(self.stds(hidden)) + 1e-4
#         return mean, std


# # =============== Posterior Net ===============
# class PosteriorNet(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.fc1 = nn.Linear(cfg.num_rnn_units + cfg.num_keypoints * 6, cfg.posterior_net_dim)
#         self.means = nn.Linear(cfg.posterior_net_dim, cfg.latent_code_size)
#         self.stds = nn.Linear(cfg.posterior_net_dim, cfg.latent_code_size)

#     def forward(self, rnn_state, keypoints_flat):
#         hidden = F.relu(self.fc1(torch.cat([rnn_state, keypoints_flat], dim=-1)))
#         mean = self.means(hidden)
#         std = F.softplus(self.stds(hidden)) + 1e-4
#         return mean, std


# # =============== Decoder ===============
# class Decoder(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.fc1 = nn.Linear(cfg.num_rnn_units + cfg.latent_code_size, 128)
#         self.fc2 = nn.Linear(128, cfg.num_keypoints * 6)

#     def forward(self, rnn_state, latent_code):
#         hidden = F.relu(self.fc1(torch.cat([rnn_state, latent_code], dim=-1)))
#         keypoints = torch.tanh(self.fc2(hidden))
#         return keypoints


# # =============== Scheduled Sampling ===============
# class ScheduledSampling(nn.Module):
#     def __init__(self, p_true_start=1.0, p_true_end=0.2, ramp_steps=10000):
#         super().__init__()
#         self.p_true_start = p_true_start
#         self.p_true_end = p_true_end
#         self.ramp_steps = ramp_steps
#         self.train_step = 0

#     def forward(self, true, pred):
#         ramp = min(self.train_step / self.ramp_steps, 1.0)
#         p_true = self.p_true_start - (self.p_true_start - self.p_true_end) * ramp
#         self.train_step += 1
#         if self.training and torch.rand(1).item() < p_true:
#             return true
#         return pred


# # =============== Best-of-Many Sampling ===============
# class SampleBestBelief(nn.Module):
#     """
#     Implements Best-of-Many sampling objective.
#     """

#     def __init__(self, num_samples, decoder, use_mean_instead_of_sample=False):
#         super().__init__()
#         self.num_samples = num_samples
#         self.decoder = decoder
#         self.use_mean_instead_of_sample = use_mean_instead_of_sample

#     def forward(self, latent_mean, latent_std, rnn_state, observed_keypoints_flat):
#         B = latent_mean.size(0)

#         # draw latent samples
#         if self.use_mean_instead_of_sample:
#             sampled_latent = latent_mean.unsqueeze(0).repeat(self.num_samples, 1, 1)
#         else:
#             dist = Normal(latent_mean, latent_std)
#             sampled_latent = dist.rsample((self.num_samples,))  # [S,B,L]

#         # decode samples
#         sampled_keypoints = []
#         for i in range(self.num_samples):
#             kp_flat = self.decoder(rnn_state, sampled_latent[i])  # [B,N*6]
#             sampled_keypoints.append(kp_flat)
#         sampled_keypoints = torch.stack(sampled_keypoints, dim=0)

#         # if only one sample
#         if self.num_samples == 1:
#             return sampled_latent[0], sampled_keypoints[0]

#         # compute sample losses [S,B]
#         sample_losses = ((sampled_keypoints - observed_keypoints_flat.unsqueeze(0)) ** 2).mean(-1)

#         return choose_sample(sampled_latent, sampled_keypoints, sample_losses, self.training)


# def choose_sample(sampled_latent, sampled_keypoints, sample_losses, training=True):
#     if training:
#         best_idx = torch.argmin(sample_losses, dim=0)  # [B]
#         batch_idx = torch.arange(sampled_latent.size(1), device=sampled_latent.device)
#         best_latent = sampled_latent[best_idx, batch_idx]
#         best_keypoints = sampled_keypoints[best_idx, batch_idx]
#     else:
#         best_latent = sampled_latent[0]
#         best_keypoints = sampled_keypoints[0]

#     return best_latent, best_keypoints


# # =============== One Iteration of VRNN ===============
# def vrnn_iteration(cfg, input_kp, rnn_state, rnn_cell,
#                    prior_net, decoder, sample_layer=None,
#                    posterior_net=None, scheduled_sampler=None):
#     B, N, D = input_kp.shape
#     observed_kp_flat = input_kp.view(B, -1)

#     mean_prior, std_prior = prior_net(rnn_state)
#     if posterior_net:
#         mean, std = posterior_net(rnn_state, observed_kp_flat)
#         kl = kl_divergence(Normal(mean, std), Normal(mean_prior, std_prior)).sum(-1)
#     else:
#         mean, std = mean_prior.detach(), std_prior.detach()
#         kl = None

#     # latent + decode
#     if sample_layer is not None:
#         z, output_flat = sample_layer(mean, std, rnn_state, observed_kp_flat)
#     else:
#         z = Normal(mean, std).rsample()
#         output_flat = decoder(rnn_state, z)

#     output_kp = output_flat.view(B, N, D)

#     # scheduled sampling for RNN input
#     rnn_input_kp = observed_kp_flat
#     if scheduled_sampler is not None:
#         rnn_input_kp = scheduled_sampler(observed_kp_flat, output_flat)

#     rnn_input = torch.cat([rnn_input_kp, z], dim=-1)
#     rnn_state = rnn_cell(rnn_input, rnn_state)

#     return output_kp, rnn_state, kl


# # =============== VRNN Model ===============
# class VRNN(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self.rnn_cell = nn.GRUCell(cfg.num_keypoints * 6 + cfg.latent_code_size, cfg.num_rnn_units)
#         self.prior_net = PriorNet(cfg)
#         self.posterior_net = PosteriorNet(cfg)
#         self.decoder = Decoder(cfg)

#         self.sample_layer = SampleBestBelief(
#             cfg.num_samples_for_bom,
#             self.decoder,
#             use_mean_instead_of_sample=cfg.use_deterministic_belief
#         )

#         self.sched_sampler_obs = ScheduledSampling(
#             cfg.scheduled_sampling_p_true_start_obs,
#             cfg.scheduled_sampling_p_true_end_obs,
#             cfg.scheduled_sampling_ramp_steps
#         )
#         self.sched_sampler_pred = ScheduledSampling(
#             cfg.scheduled_sampling_p_true_start_pred,
#             cfg.scheduled_sampling_p_true_end_pred,
#             cfg.scheduled_sampling_ramp_steps
#         )

#     def forward(self, x):
#         """
#         x: [B, T, num_keypoints, 6]
#         Returns: output keypoints [B, T, num_keypoints, 6], KL stack [B, T_obs]
#         """
#         B, T, N, D = x.shape
#         rnn_state = torch.zeros(B, self.cfg.num_rnn_units, device=x.device)
#         outputs, kls = [], []

#         # observed steps
#         for t in range(self.cfg.observed_steps):
#             out, rnn_state, kl = vrnn_iteration(
#                 self.cfg, x[:, t], rnn_state, self.rnn_cell,
#                 self.prior_net, self.decoder,
#                 sample_layer=self.sample_layer,
#                 posterior_net=self.posterior_net,
#                 scheduled_sampler=self.sched_sampler_obs
#             )
#             outputs.append(out)
#             kls.append(kl)

#         # predicted steps
#         for t in range(self.cfg.observed_steps, T):
#             out, rnn_state, _ = vrnn_iteration(
#                 self.cfg, x[:, t], rnn_state, self.rnn_cell,
#                 self.prior_net, self.decoder,
#                 sample_layer=self.sample_layer,
#                 posterior_net=None,
#                 scheduled_sampler=self.sched_sampler_pred
#             )
#             outputs.append(out)

#         output_stack = torch.stack(outputs, dim=1)
#         kl_stack = torch.stack(kls, dim=1) if len(kls) > 0 else None
#         return output_stack, kl_stack

# def l2_loss_tf_style(pred, target):
#     diff = pred - target
#     return 0.5 * torch.sum(diff ** 2) / (target.shape[0] * target.shape[1])

# # =============== Weight Init ===============
# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


# ========= Config =========
class ConfigDict(dict):
    """Dictionary with attribute-style access."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value
    def get(self, key, default=None):
        return self[key] if key in self else default


def get_config():
    cfg = ConfigDict()
    cfg.batch_size = 32
    cfg.steps_per_epoch = 50
    cfg.num_epochs = 15
    cfg.learning_rate = 0.001
    cfg.clipnorm = 10

    cfg.observed_steps = 6
    cfg.predicted_steps = 6
    cfg.num_keypoints = 10
    cfg.num_rnn_units = 256
    cfg.prior_net_dim = 128
    cfg.posterior_net_dim = 128
    cfg.latent_code_size = 20
    cfg.kl_loss_scale = 0.0001
    cfg.kl_annealing_steps = 1000

    cfg.num_samples_for_bom = 10   # Best-of-Many samples
    cfg.num_samples = 100          # Samples at inference
    cfg.use_deterministic_belief = False
    return cfg


# ========= Networks =========
class PriorNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.num_rnn_units, cfg.prior_net_dim)
        self.means = nn.Linear(cfg.prior_net_dim, cfg.latent_code_size)
        self.stds = nn.Linear(cfg.prior_net_dim, cfg.latent_code_size)

    def forward(self, rnn_state):
        h = F.relu(self.fc1(rnn_state))
        mean = self.means(h)
        std = F.softplus(self.stds(h)) + 1e-4
        return mean, std


class PosteriorNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.num_rnn_units + cfg.num_keypoints * 6,
                             cfg.posterior_net_dim)
        self.means = nn.Linear(cfg.posterior_net_dim, cfg.latent_code_size)
        self.stds = nn.Linear(cfg.posterior_net_dim, cfg.latent_code_size)

    def forward(self, rnn_state, keypoints_flat):
        h = F.relu(self.fc1(torch.cat([rnn_state, keypoints_flat], dim=-1)))
        mean = self.means(h)
        std = F.softplus(self.stds(h)) + 1e-4
        return mean, std


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.num_rnn_units + cfg.latent_code_size, 128)
        self.fc2 = nn.Linear(128, cfg.num_keypoints * 6)

    def forward(self, rnn_state, latent_code):
        h = F.relu(self.fc1(torch.cat([rnn_state, latent_code], dim=-1)))
        return torch.tanh(self.fc2(h))


# ========= Samplers =========
class SampleBestBelief(nn.Module):
    """Best-of-Many sampling for training."""
    def __init__(self, num_samples, decoder, use_mean_instead_of_sample=True):
        super().__init__()
        self.num_samples = num_samples
        self.decoder = decoder
        self.use_mean_instead_of_sample = use_mean_instead_of_sample

    def forward(self, mean, std, rnn_state, observed_kp_flat):
        B = mean.size(0)
        if self.use_mean_instead_of_sample:
            sampled_latent = mean.unsqueeze(0).repeat(self.num_samples, 1, 1)
        else:
            dist = Normal(mean, std)
            sampled_latent = dist.rsample((self.num_samples,))  # [S,B,L]

        # Decode samples
        sampled_keypoints = []
        for i in range(self.num_samples):
            kp_flat = self.decoder(rnn_state, sampled_latent[i])  # [B,N*6]
            sampled_keypoints.append(kp_flat)
        sampled_keypoints = torch.stack(sampled_keypoints, dim=0)  # [S,B,N*6]

        # Loss for each sample
        obs = observed_kp_flat.unsqueeze(0)  # [1,B,N*6]
        losses = ((sampled_keypoints - obs) ** 2).mean(dim=-1)  # [S,B]

        # Best sample per batch
        best_idx = torch.argmin(losses, dim=0)  # [B]
        best_latent = sampled_latent[best_idx, torch.arange(B)]
        best_kp = sampled_keypoints[best_idx, torch.arange(B)]
        return best_latent, best_kp


class SampleAllBeliefs(nn.Module):
    """Sample all beliefs during inference."""
    def __init__(self, num_samples, decoder, use_mean_instead_of_sample=False):
        super().__init__()
        self.num_samples = num_samples
        self.decoder = decoder
        self.use_mean_instead_of_sample = use_mean_instead_of_sample

    def forward(self, mean, std, rnn_state, observed_kp_flat):
        B = mean.size(0)
        if self.use_mean_instead_of_sample:
            sampled_latent = mean.unsqueeze(0).repeat(self.num_samples, 1, 1)
        else:
            dist = Normal(mean, std)
            sampled_latent = dist.rsample((self.num_samples,))  # [S,B,L]

        # Decode all
        sampled_keypoints = []
        for i in range(self.num_samples):
            kp_flat = self.decoder(rnn_state, sampled_latent[i])
            sampled_keypoints.append(kp_flat)
        sampled_keypoints = torch.stack(sampled_keypoints, dim=0)  # [S,B,N*6]

        return sampled_latent, sampled_keypoints, sampled_latent[0], sampled_keypoints[0]


class KLDivergence(nn.Module):
    def __init__(self, kl_annealing_steps=0):
        super().__init__()
        self.kl_annealing_steps = kl_annealing_steps
        self.register_buffer('train_step', torch.tensor(0.0))

    def forward(self, mean_prior, std_prior, mean, std):
        # Compute KL divergence between two Normal distributions
        kl = kl_divergence(Normal(mean, std), Normal(mean_prior, std_prior)).sum(-1)

        if self.kl_annealing_steps > 0:
            weight = min(self.train_step / self.kl_annealing_steps, 1.0)
            kl = kl * weight

        return kl

    def step(self):
        self.train_step += 1


# ========= VRNN Iteration =========
def vrnn_iteration(cfg, input_kp, rnn_state, rnn_cell,
                   prior_net, decoder,
                   posterior_net=None,
                   sample_best=None, sample_all=None,kl_module=None,
                   training=True):
    B, N, D = input_kp.shape
    observed_kp_flat = input_kp.view(B, -1)

    # Prior & Posterior
    mean_prior, std_prior = prior_net(rnn_state)
    if posterior_net is not None:
        mean, std = posterior_net(rnn_state, observed_kp_flat)
        kl = kl_module(mean_prior, std_prior, mean, std)
    else:
        mean, std = mean_prior.detach(), std_prior.detach()
        kl = None

    # Sampling
    if training:
        z, output_flat = sample_best(mean, std, rnn_state, observed_kp_flat)  # [B,L],[B,N*6]
        output_kp = output_flat.view(B, N, D)
    else:
        z_all, kp_all, z, output_flat = sample_all(mean, std, rnn_state, observed_kp_flat)
        output_kp = kp_all.view(cfg.num_samples, B, N, D)  # [S,B,N,D]

    # Update RNN
    rnn_input = torch.cat([observed_kp_flat, z], dim=-1)
    rnn_state = rnn_cell(rnn_input, rnn_state)

    return output_kp, rnn_state, kl


# ========= VRNN Model =========
class VRNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.rnn_cell = nn.GRUCell(cfg.num_keypoints * 6 + cfg.latent_code_size,
                                   cfg.num_rnn_units)
        self.prior_net = PriorNet(cfg)
        self.posterior_net = PosteriorNet(cfg)
        self.decoder = Decoder(cfg)

        self.sample_best = SampleBestBelief(cfg.num_samples_for_bom,
                                            self.decoder,
                                            use_mean_instead_of_sample=cfg.use_deterministic_belief)
        self.sample_all = SampleAllBeliefs(cfg.num_samples,
                                           self.decoder,
                                           use_mean_instead_of_sample=cfg.use_deterministic_belief)
        self.kl_module = KLDivergence(cfg.kl_annealing_steps)

    def forward(self, x):
        """
        x: [B,T,N,D]
        Returns:
            Training → [B,T,N,D], [B,T_obs] KL
            Inference → [S,B,T,N,D], None
        """
        B, T, N, D = x.shape
        rnn_state = torch.zeros(B, self.cfg.num_rnn_units, device=x.device)
        outputs, kls = [], []

        # Observed steps
        for t in range(self.cfg.observed_steps):
            out, rnn_state, kl = vrnn_iteration(
                self.cfg, x[:, t], rnn_state,
                self.rnn_cell, self.prior_net, self.decoder,
                posterior_net=self.posterior_net,
                sample_best=self.sample_best,
                sample_all=self.sample_all,
                kl_module=self.kl_module,
                training=self.training
            )
            outputs.append(out)
            kls.append(kl)

        # Predicted steps
        for t in range(self.cfg.observed_steps, T):
            out, rnn_state, _ = vrnn_iteration(
                self.cfg, x[:, t], rnn_state,
                self.rnn_cell, self.prior_net, self.decoder,
                posterior_net=None,
                sample_best=self.sample_best,
                sample_all=self.sample_all,
                training=self.training
            )
            outputs.append(out)

        if self.training:
            output_stack = torch.stack(outputs, dim=1)   # [B,T,N,D]
            kl_stack = torch.stack(kls, dim=1)
        else:
            output_stack = torch.stack(outputs, dim=2)   # [S,B,T,N,D]
            output_stack = output_stack.permute(1,0,2,3,4)
            kl_stack = None

        return output_stack, kl_stack

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
