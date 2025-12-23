# vrnn_overlap_all_samples.py
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
    cfg.learning_rate = 3e-4 #0.0001 #0.001
    cfg.clipnorm = 10

    # Short non-overlap block settings (kept for compatibility)
    cfg.observed_steps = 5
    cfg.predicted_steps = 20

    cfg.num_keypoints = 10
    cfg.num_rnn_units = 128 #64 #256
    cfg.prior_net_dim = 512 #128
    cfg.posterior_net_dim = 512 #128
    cfg.decoder_dim = 512 #128
    cfg.latent_code_size = 15
    cfg.kl_loss_scale = 3e-5 #0.0001
    cfg.kl_annealing_steps = 1000

    cfg.num_samples_for_bom = 50 #10   # Best-of-Many samples (training)
    cfg.num_samples = 100          # Inference samples per step
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
        self.fc1 = nn.Linear(cfg.num_rnn_units + cfg.num_keypoints*6,
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
        self.fc1 = nn.Linear(cfg.num_rnn_units + cfg.latent_code_size, cfg.decoder_dim)
        self.fc2 = nn.Linear(cfg.decoder_dim, cfg.num_keypoints*6)

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
            kp_flat = self.decoder(rnn_state, sampled_latent[i])  # [B,N*3]
            sampled_keypoints.append(kp_flat)
        sampled_keypoints = torch.stack(sampled_keypoints, dim=0)  # [S,B,N*3]

        # Loss for each sample (MSE vs observed)
        obs = observed_kp_flat.unsqueeze(0)  # [1,B,N*3]
        losses = ((sampled_keypoints - obs) ** 2).mean(dim=-1)  # [S,B]

        # Best sample per batch element
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

    def forward(self, mean, std, rnn_state):
        B = mean.size(0)
        #if self.use_mean_instead_of_sample:
        sampled_latent_mean = mean #.unsqueeze(0) #.repeat(self.num_samples, 1, 1)
        #else:
        dist = Normal(mean, std)
        sampled_latent = dist.rsample((self.num_samples,))  # [S,B,L]
        sampled_keypoint_mean = self.decoder(rnn_state, sampled_latent_mean)
        # Decode all
        sampled_keypoints = []
        for i in range(self.num_samples):
            kp_flat = self.decoder(rnn_state, sampled_latent[i])  # [B,N*3]
            sampled_keypoints.append(kp_flat)
        sampled_keypoints = torch.stack(sampled_keypoints, dim=0)  # [S,B,N*3]

        # Return also the first sample (latent and kp) for RNN update consistency
        return sampled_latent, sampled_keypoints, sampled_latent_mean, sampled_keypoint_mean


class KLDivergence(nn.Module):
    def __init__(self, kl_annealing_steps=0):
        super().__init__()
        self.kl_annealing_steps = kl_annealing_steps
        self.register_buffer('train_step', torch.tensor(0.0))

    def forward(self, mean_prior, std_prior, mean, std):
        kl = kl_divergence(Normal(mean, std), Normal(mean_prior, std_prior)).sum(-1)
        if self.kl_annealing_steps > 0:
            weight = min(self.train_step / self.kl_annealing_steps, 1.0)
            kl = kl * weight
        return kl

    def step(self):
        self.train_step += 1


# ========= One VRNN step =========
def vrnn_iteration(cfg, input_kp, rnn_state, rnn_cell,
                   prior_net, decoder,
                   posterior_net=None,
                   sample_best=None, sample_all=None, kl_module=None,
                   training=True):
    """
    input_kp: [B, N, D]
    rnn_state: [B, H]
    """
    B, N, D = input_kp.shape
    observed_kp_flat = input_kp.view(B, -1)  # [B, N*D]

    # Prior & Posterior
    mean_prior, std_prior = prior_net(rnn_state)
    if posterior_net is not None:
        mean, std = posterior_net(rnn_state, observed_kp_flat)
        kl = kl_module(mean_prior, std_prior, mean, std) if kl_module is not None else None
    else:
        mean, std = mean_prior.detach(), std_prior.detach()
        kl = None

    # Sampling / decoding
    if training:
        z, output_flat = sample_best(mean, std, rnn_state, observed_kp_flat)  # [B,L], [B,N*D]
        output_kp = output_flat.view(B, N, D)
    else:
        z_all, kp_all, z, _ = sample_all(mean, std, rnn_state) #, output_flat
        output_kp = kp_all.view(cfg.num_samples, B, N, D)  # [S,B,N,D]

    # Update RNN state (uses the single z chosen above)
    rnn_input = torch.cat([observed_kp_flat, z], dim=-1)
    rnn_state = rnn_cell(rnn_input, rnn_state)

    return output_kp, rnn_state, kl

def vrnn_iteration_pred(cfg, input_kp, rnn_state, rnn_cell,
                   prior_net, decoder,
                   posterior_net=None,
                   sample_best=None, sample_all=None, kl_module=None,
                   training=True):
    """
    input_kp: [B, N, D]
    rnn_state: [B, H]
    """
    B, N, D = input_kp.shape
    observed_kp_flat = input_kp.view(B, -1)  # [B, N*D]

    # Prior & Posterior
    mean_prior, std_prior = prior_net(rnn_state)
    if posterior_net is not None:
        mean, std = posterior_net(rnn_state, observed_kp_flat)
        kl = kl_module(mean_prior, std_prior, mean, std) if kl_module is not None else None
    else:
        mean, std = mean_prior.detach(), std_prior.detach()
        kl = None

    # Sampling / decoding
    if training:
        z, output_flat = sample_best(mean, std, rnn_state, observed_kp_flat)  # [B,L], [B,N*D]
        output_kp = output_flat.view(B, N, D)
    else:
        z_all, kp_all, z, kp = sample_all(mean, std, rnn_state) #, output_flat
        output_kp = kp_all.view(cfg.num_samples, B, N, D)  # [S,B,N,D]
        #output_flat = output_flat.view(B, -1)
        #output_flat = output_kp.mean(axis=0).view(B, -1)
        output_flat = kp.view(B, -1)

    # Update RNN state (uses the single z chosen above)
    if training:
        rnn_input = torch.cat([observed_kp_flat, z], dim=-1)
        rnn_state = rnn_cell(rnn_input, rnn_state)
    else:
        rnn_input = torch.cat([output_flat, z], dim=-1)
        rnn_state = rnn_cell(rnn_input, rnn_state)

    return output_kp, rnn_state, kl


# ========= VRNN Model =========
class VRNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.rnn_cell = nn.GRUCell(cfg.num_keypoints*6 + cfg.latent_code_size,
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
        Standard teacher-forced pass on a short block:
        x: [B,T,N,D]
        Returns:
            Training  -> ( [B,T,N,D], [B,T_obs] KL )
            Inference -> ( [S,B,T,N,D], None )
        """
        B, T, N, D = x.shape
        rnn_state = torch.zeros(B, self.cfg.num_rnn_units, device=x.device)
        outputs, kls = [], []

        # Observed steps (posterior)
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

        # Predicted steps (prior)
        for t in range(self.cfg.observed_steps, T):
            out, rnn_state, _ = vrnn_iteration_pred(
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
            kl_stack = torch.stack(kls, dim=1) if len(kls) > 0 else None
        else:
            # outputs are [S,B,N,D] per time -> stack -> [T,S,B,N,D] -> permute -> [S,B,T,N,D]
            output_stack = torch.stack(outputs, dim=2)   # [S,B,T,N,D]
            output_stack = output_stack.permute(1,0,2,3,4)
            kl_stack = None

        return output_stack, kl_stack

    # ====== NEW: Overlapping rollout that keeps ALL S samples ======
    # @torch.no_grad()
    # def forward(self, x, obs: int = 120, pred: int = 30, stride: int = 30, squeeze_batch_if_added: bool = False):
    #     """
    #     Overlapping rollout on a long sequence, keeping ALL S samples per forecast step.
    
    #     Accepts x in shapes:
    #       - [B, T_full, N, D]
    #       - [T_full, N, D]          (adds B=1)
    #       - [T_full, N]             (adds D=1 and B=1)
    
    #     Returns:
    #       pred_samples: [S, B, T_full, N, D] with predictions at forecast indices; NaN elsewhere.
    #                     If squeeze_batch_if_added=True and B was added here, returns [S, T_full, N, D].
    #     """
    #     self.eval()
    
    #     # ---- normalize to [B,T,N,D] ----
    #     added_batch = False
    #     if x.dim() == 2:            # [T,N]  -> [1,T,N,1]
    #         x = x.unsqueeze(0).unsqueeze(-1)
    #         added_batch = True
    #     elif x.dim() == 3:          # [T,N,D] -> [1,T,N,D]
    #         x = x.unsqueeze(0)
    #         added_batch = True
    #     elif x.dim() != 4:
    #         raise ValueError(f"Expected x with 2/3/4 dims, got {x.shape}")
    
    #     x = x.to(next(self.parameters()).device)
    #     B, T_full, N, D = x.shape
    
    #     H = self.cfg.num_rnn_units
    #     S = self.cfg.num_samples
    #     device = x.device
    
    #     # allocate output
    #     pred_samples = torch.full((S, B, T_full, N, D), float('nan'), device=device)
    
    #     t0 = 0
    #     while t0 + obs + pred <= T_full:
    #         rnn_state = torch.zeros(B, H, device=device)
    #         last_kp = None  # [B,N,D]
    
    #         for i in range(obs + pred):
    #             cur_t = t0 + i
    
    #             # context: ground truth + posterior; forecast: model output + prior
    #             if i < obs:
    #                 input_kp = x[:, cur_t]             # [B,N,D]
    #                 post = self.posterior_net
    #             else:
    #                 input_kp = x[:, t0 + obs - 1] if last_kp is None else last_kp
    #                 post = None
    
    #             # single step (inference mode): out is [S,B,N,D]
    #             out, rnn_state, _ = vrnn_iteration(
    #                 self.cfg,
    #                 input_kp,
    #                 rnn_state,
    #                 self.rnn_cell,
    #                 self.prior_net,
    #                 self.decoder,
    #                 posterior_net=post,
    #                 sample_best=self.sample_best,
    #                 sample_all=self.sample_all,
    #                 training=False
    #             )
    
    #             # advance autoregressively with sample-0
    #             last_kp = out[0]  # [B,N,D]
    
    #             # record only forecast portion (keep ALL S samples)
    #             if i >= obs:
    #                 pred_samples[:, :, cur_t] = out  # [S,B,N,D] -> time index cur_t
    
    #         t0 += stride
    
    #     # optionally squeeze the batch dim if we added it here
    #     if added_batch and squeeze_batch_if_added:
    #         pred_samples = pred_samples.squeeze(1)  # [S, T_full, N, D]
    
    #     return pred_samples



# ========= Utils =========
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ========= Convenience wrapper for your 120/30/30 schedule =========
def vrnn_overlap_predict(model: VRNN, x):
    """
    Convenience wrapper for 120-in / 30-out with stride 30.
    x: [B, T_full, N, D]
    Returns: [S, B, T_full, N, D] with NaNs outside forecast indices.
    """
    return model.rollout_overlapping(x, obs=120, pred=30, stride=30)