# sasrec.py - fixed predict_time method
import math
import torch
import torch.nn as nn
import numpy as np
from Baselines.Baseline1.utils import get_mask
from encoder import POIEncoder, TimeEncoder
from Data_Module import Preprocessor


class SASRec(nn.Module):
    def __init__(self, n_item, args):
        super(SASRec, self).__init__()
        self.hidden = args.hidden_units
        self.dev = args.device
        self.args = args  # keep args for later use

        # Independent POI encoder (inputs: POI sequence + time sequence)
        self.poi_encoder = POIEncoder(args, n_item)

        # Independent time encoder (inputs: time sequence + POI sequence)
        self.time_encoder = TimeEncoder(args, n_item)

        # POI prediction output layer
        self.out = nn.Linear(args.hidden_units, n_item + 1)

    def forward(self, seq, pos_seqs, neg_seqs, time_seqs):
        # 1) POI encoding path: POI sequences + time sequences
        poi_encoded = self.poi_encoder(seq, time_seqs, seq)
        poi_context = poi_encoded[:, :, :self.hidden]

        # 2) Time encoding path: time sequences + POI sequences
        time_output = self.time_encoder(time_seqs, seq, seq, poi_context=poi_context)
        # POI prediction always uses the Transformer-encoded poi_context (not the Mamba-updated representation)
        output_poi = poi_context

        # 3) POI prediction
        pos_embs = self.poi_encoder.poi_embedding(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.poi_encoder.poi_embedding(torch.LongTensor(neg_seqs).to(self.dev))
        pos_logits = (output_poi * pos_embs).sum(dim=-1)
        neg_logits = (output_poi * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits, time_output

    def predict(self, user_ids, log_seqs, time_seqs, item_indices, min_year):
        poi_encoded = self.poi_encoder(log_seqs, time_seqs, log_seqs)
        poi_context = poi_encoded[:, :, :self.hidden]
        if getattr(self.args, 'time_encoder_type', 'transformer') == 'mamba':
            # Run time encoder for time prediction, but POI prediction sticks to Transformer output
            _ = self.time_encoder(time_seqs, log_seqs, log_seqs, poi_context=poi_context)
        logits = poi_context[:, -1:, :]
        item_embs = self.poi_encoder.poi_embedding(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(logits.unsqueeze(-1)).squeeze(-1)

        return logits

    def _von_mises_topk_times(self, params, topk=2):
        """Return the top-k (h, m, s) times by mixture weight."""
        pi = params['pi']
        mu = params['mu'] % (2 * math.pi)
        k = min(topk, pi.size(-1))
        if k <= 0:
            raise ValueError("topk must be positive")
        topk_pi, idx = torch.topk(pi, k=k, dim=-1)
        topk_mu = torch.gather(mu, -1, idx)
        minutes = topk_mu / (2 * math.pi) * 1440.0
        hours = torch.floor(minutes / 60.0)
        mins = torch.floor(minutes % 60.0)
        secs = (minutes - torch.floor(minutes)) * 60.0
        return torch.stack([hours, mins, secs], dim=-1)

    def predict_time(self, log_seqs, time_seqs, min_year, pos_time, return_details=False, vm_topk_mode='min'):
        log_arr = np.array(log_seqs)
        time_arr = np.array(time_seqs)
        if log_arr.ndim == 1:
            log_arr = log_arr.reshape(1, -1)
        if time_arr.ndim == 1:
            time_arr = time_arr.reshape(1, -1)

        poi_encoded = self.poi_encoder(log_arr, time_arr, log_arr)
        poi_context = poi_encoded[:, :, :self.hidden]
        time_output = self.time_encoder(time_arr, log_arr, log_arr, poi_context=poi_context)
        if isinstance(time_output, dict):
            vm_dict = {k: v.detach() for k, v in time_output.items() if k in ('pi', 'mu', 'kappa')}
            time_output = self._von_mises_topk_times(vm_dict, topk=2)
        vm_topk_mode = vm_topk_mode or 'min'

        from time_utils import timestamp_sequence_to_datetime_sequence_batch
        datetime_sequences = timestamp_sequence_to_datetime_sequence_batch(torch.LongTensor(pos_time).unsqueeze(0))
        input_data = []
        for datetime_sequence in datetime_sequences:
            sequence_data = []
            for dt0 in datetime_sequence:
                if dt0 is None:
                    sequence_data.append([0, 0, 0])
                else:
                    sequence_data.append([dt0.hour, dt0.minute, dt0.second])
            input_data.append(sequence_data)

        input_data_tensor = torch.tensor(input_data).to(self.dev).float()
        input_data_tensor = torch.round(input_data_tensor)
        time_output = torch.round(time_output)

        # Optionally take the first candidate from top-k
        is_topk_tensor = time_output.dim() == 4
        if is_topk_tensor and vm_topk_mode == 'first':
            time_output = time_output[:, :, 0, :]
            is_topk_tensor = False
        elif is_topk_tensor and vm_topk_mode not in ('min', 'first'):
            raise ValueError(f"Unsupported vm_topk_mode: {vm_topk_mode}")

        # ===== Circular time difference (supports top-k min error or first candidate) =====
        true_seconds = input_data_tensor[:, :, 0] * 3600 + input_data_tensor[:, :, 1] * 60 + input_data_tensor[:, :, 2]
        if not is_topk_tensor:
            # Transformer branch
            pred_seconds = time_output[:, :, 0] * 3600 + time_output[:, :, 1] * 60 + time_output[:, :, 2]
            direct_error = torch.abs(true_seconds - pred_seconds)
            circular_error = 86400 - torch.abs(true_seconds - pred_seconds)
            min_error = torch.min(direct_error, circular_error)
            time_diff = min_error[:, -1].mean().item()
            pred_print = time_output[0, -1, :]
        else:
            # Mamba top-k branch: [B, L, K, 3], pick candidate with smallest error
            pred_seconds = time_output[:, :, :, 0] * 3600 + time_output[:, :, :, 1] * 60 + time_output[:, :, :, 2]
            true_expanded = true_seconds.unsqueeze(-1)
            direct_error = torch.abs(true_expanded - pred_seconds)
            circular_error = 86400 - torch.abs(true_expanded - pred_seconds)
            circular_min = torch.min(direct_error, circular_error)
            min_error, _ = torch.min(circular_min, dim=-1)
            time_diff = min_error[:, -1].mean().item()
            pred_print = time_output[0, -1, 0, :]

        true_last = input_data_tensor[0, -1, :]
        true_h, true_m, true_s = int(true_last[0].item()), int(true_last[1].item()), int(true_last[2].item())
        pred_h, pred_m, pred_s = int(pred_print[0].item()), int(pred_print[1].item()), int(pred_print[2].item())
        print(f"Ground truth time: {true_h:02d}:{true_m:02d}:{true_s:02d}")
        print(f"Predicted time:    {pred_h:02d}:{pred_m:02d}:{pred_s:02d}")
        print(f"Circular time diff (min): {time_diff:.0f} seconds")
        print("-" * 50)

        detail = {
            'avg_error': time_diff,
            'per_pos_error': min_error.squeeze(0).detach().cpu().numpy(),
            'seq_length': int(np.count_nonzero(log_seqs))
        }
        if return_details:
            return detail
        return time_diff
