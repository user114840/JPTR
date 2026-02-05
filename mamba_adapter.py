# mamba_adapter.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba.mamba import Mamba, MambaConfig


class MambaTimePredictor(nn.Module):
    """
    Mamba-based time predictor that uses multi-frequency time encodings.
    """

    def __init__(self, args, multi_freq_encoder):
        super(MambaTimePredictor, self).__init__()

        # Multi-frequency time encoder
        self.multi_freq_encoder = multi_freq_encoder
        self.d_model = multi_freq_encoder.output_dim
        self.hidden_units = args.hidden_units
        self.poi_input_proj = nn.Linear(args.hidden_units, self.d_model)
        self.delta_proj = nn.Linear(self.d_model, self.d_model)
        self.delta_gate = nn.Linear(self.d_model, self.d_model)
        self.time_to_poi_proj = nn.Linear(self.d_model, args.hidden_units)
        use_cuda_scan = getattr(args, 'mamba_use_cuda', True) and (args.device != 'cpu')

        # Mamba configuration
        mamba_config = MambaConfig(
            d_model=self.d_model,
            n_layers=args.time_encoder_layers,
            dt_rank='auto',
            d_state=16,
            expand_factor=2,
            d_conv=4,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            rms_norm_eps=1e-5,
            bias=False,
            conv_bias=True,
            inner_layernorms=False,
            mup=False,
            pscan=True,
            use_cuda=use_cuda_scan,
        )

        # Dual-path Mamba blocks
        self.poi_mamba = Mamba(mamba_config)
        self.time_mamba = Mamba(mamba_config)

        # Von Mises mixture head
        self.num_components = getattr(args, 'vm_num_components', 3)
        self.pi_proj = nn.Linear(self.d_model, self.num_components)
        self.mu_proj = nn.Linear(self.d_model, self.num_components)
        self.kappa_proj = nn.Linear(self.d_model, self.num_components)
        self.min_kappa = getattr(args, 'vm_min_kappa', 0.1)
        self.max_kappa = getattr(args, 'vm_max_kappa', 20.0)

    def _project_to_von_mises(self, features):
        """Map Mamba outputs to Von Mises mixture parameters."""
        pi_logits = self.pi_proj(features)
        mu_raw = self.mu_proj(features)
        kappa_raw = self.kappa_proj(features)

        pi = torch.softmax(pi_logits, dim=-1)
        mu = torch.remainder(mu_raw, 2 * math.pi)
        kappa = F.softplus(kappa_raw) + 1e-4
        if self.max_kappa is not None:
            kappa = torch.clamp(kappa, min=self.min_kappa, max=self.max_kappa)
        else:
            kappa = torch.clamp_min(kappa, self.min_kappa)
        return {'pi': pi, 'mu': mu, 'kappa': kappa}

    def forward(self, poi_embeddings, time_seqs, return_backbone=False):
        # Encode time with multi-frequency embeddings
        time_h = self.multi_freq_encoder(time_seqs).contiguous()
        device = time_h.device
        poi_residual = poi_embeddings.to(device).contiguous()
        poi_h = self.poi_input_proj(poi_residual).contiguous()

        for layer_idx in range(len(self.time_mamba.layers)):
            poi_h = self.poi_mamba.layers[layer_idx](poi_h)
            delta = self.delta_proj(poi_h)
            gate = torch.sigmoid(self.delta_gate(poi_h))
            delta = delta * gate

            time_h = self.time_mamba.layers[layer_idx](time_h)
            time_h = time_h + delta
            delta_time = self.time_to_poi_proj(time_h)
            poi_residual = poi_residual + delta_time
            poi_h = self.poi_input_proj(poi_residual)

        vm_params = self._project_to_von_mises(time_h)
        vm_params['updated_poi'] = poi_residual
        if return_backbone:
            vm_params['backbone'] = time_h
        return vm_params
