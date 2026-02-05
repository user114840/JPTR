import numpy as np
import torch
import torch.nn as nn

from Baselines.Baseline1.sasrec import SASRec
from Baselines.Baseline2.gru4rec import GRU4Rec
from Baselines.Baseline3.caser_timestamp import Caser
from Baselines.Baseline4.bert4rec import Bert4Rec
from Baselines.Baseline5.Tisasrec import TiSASRec
from Baselines.Baseline6.fmlp4rec import FMLP4Rec
from Baselines.Baseline7.geosan import GeoSANSeqRec
from encoder import TimeEncoder
from time_utils import timestamp_sequence_to_datetime_sequence_batch

import math
import numpy as np


def _decode_von_mises_topk(vm_params, topk=2):
    pi = vm_params['pi']
    mu = vm_params['mu'] % (2 * math.pi)
    k = min(topk, pi.size(-1))
    topk_pi, indices = torch.topk(pi, k=k, dim=-1)
    topk_mu = torch.gather(mu, -1, indices)
    minutes = topk_mu / (2 * math.pi) * 1440.0
    hours = torch.floor(minutes / 60.0)
    mins = torch.floor(minutes % 60.0)
    secs = (minutes - torch.floor(minutes)) * 60.0
    return torch.stack([hours, mins, secs], dim=-1)


def _compute_time_error(time_output, pos_time, device, is_dict=False, vm_topk_mode='min'):
    pos_np = np.array(pos_time)
    if pos_np.ndim == 1:
        pos_np = pos_np.reshape(1, -1)
    datetime_sequences = timestamp_sequence_to_datetime_sequence_batch(pos_np.tolist())
    input_data = []
    for datetime_sequence in datetime_sequences:
        sequence_data = []
        for dt0 in datetime_sequence:
            if dt0 is None:
                sequence_data.append([0, 0, 0])
            else:
                sequence_data.append([dt0.hour, dt0.minute, dt0.second])
        input_data.append(sequence_data)

    true_tensor = torch.tensor(input_data, device=device).float()
    true_seconds = true_tensor[:, :, 0] * 3600 + true_tensor[:, :, 1] * 60 + true_tensor[:, :, 2]

    if is_dict:
        pred_time = _decode_von_mises_topk(time_output, topk=2)
        pred_seconds = pred_time[:, :, :, 0] * 3600 + pred_time[:, :, :, 1] * 60 + pred_time[:, :, :, 2]
        true_expanded = true_seconds.unsqueeze(-1)
        direct_error = torch.abs(true_expanded - pred_seconds)
        circular_error = 86400 - torch.abs(true_expanded - pred_seconds)
        if vm_topk_mode == 'first':
            per_pos_error = torch.min(direct_error[..., 0], circular_error[..., 0])
        elif vm_topk_mode == 'min':
            per_pos_error, _ = torch.min(torch.min(direct_error, circular_error), dim=-1)
        else:
            raise ValueError(f"Unsupported vm_topk_mode: {vm_topk_mode}")
        pred_print = pred_time[0, -1, 0, :]
    else:
        pred_time = torch.round(time_output)
        pred_seconds = pred_time[:, :, 0] * 3600 + pred_time[:, :, 1] * 60 + pred_time[:, :, 2]
        direct_error = torch.abs(true_seconds - pred_seconds)
        circular_error = 86400 - torch.abs(true_seconds - pred_seconds)
        per_pos_error = torch.min(direct_error, circular_error)
        pred_print = pred_time[0, -1, :]

    time_diff = per_pos_error[:, -1].mean().item()
    detail = {
        'avg_error': time_diff,
        'per_pos_error': per_pos_error.squeeze(0).detach().cpu().numpy(),
        'seq_length': int(true_tensor.shape[1])
    }
    return time_diff, detail, pred_print


def _ensure_batch(arr):
    arr_np = np.array(arr)
    if arr_np.ndim == 1:
        arr_np = arr_np.reshape(1, -1)
    return arr_np

def _build_time_matrix_batch(time_batch, time_span, device):
    """Compute clipped absolute time difference matrices for a batch."""
    t = torch.tensor(time_batch, device=device, dtype=torch.long)
    diff = torch.abs(t.unsqueeze(2) - t.unsqueeze(1))
    return torch.clamp(diff, max=time_span)


def _scalar_time_detail(error, log_seqs):
    seq_len = int((torch.as_tensor(log_seqs) != 0).sum().item())
    return {
        "avg_error": float(error),
        "per_pos_error": np.array([float(error)], dtype=np.float32),
        "seq_length": seq_len,
    }


class BaseWrapper(nn.Module):
    def __init__(self, model, supports_time_prediction=True):
        super().__init__()
        self.model = model
        self.supports_time_prediction = supports_time_prediction

    def forward(self, users, seq, pos, neg, seq_time):
        raise NotImplementedError

    def predict(self, user_ids, log_seqs, time_seqs, item_indices, min_year):
        raise NotImplementedError

    def predict_time(self, log_seqs, time_seqs, min_year, pos_time, return_details=False, **kwargs):
        if not self.supports_time_prediction:
            raise NotImplementedError("Time prediction not supported by this baseline")
        raise NotImplementedError


class SASRecWrapper(BaseWrapper):
    def __init__(self, model):
        super().__init__(model, supports_time_prediction=True)

    def forward(self, users, seq, pos, neg, seq_time):
        return self.model(seq, pos, neg, seq_time)

    def predict(self, user_ids, log_seqs, time_seqs, item_indices, min_year):
        return self.model.predict(user_ids, log_seqs, time_seqs, item_indices, min_year)

    def predict_time(self, log_seqs, time_seqs, min_year, pos_time, return_details=False, vm_topk_mode='min'):
        return self.model.predict_time(log_seqs, time_seqs, min_year, pos_time, return_details=return_details, vm_topk_mode=vm_topk_mode)


class GRU4RecWrapper(BaseWrapper):
    def __init__(self, model, time_encoder):
        super().__init__(model, supports_time_prediction=True)
        self.time_encoder = time_encoder
        self.time_encoder_type = time_encoder.time_encoder_type

    def forward(self, users, seq, pos, neg, seq_time):
        pos_logits, neg_logits, _ = self.model(seq, pos, neg, seq_time)
        time_output = self.time_encoder(seq_time, seq, seq)
        return pos_logits, neg_logits, time_output

    def predict(self, user_ids, log_seqs, time_seqs, item_indices, min_year):
        return self.model.predict(user_ids, log_seqs, time_seqs, item_indices, min_year)

    def predict_time(self, log_seqs, time_seqs, min_year, pos_time, return_details=False, vm_topk_mode='min', **kwargs):
        log_seqs = _ensure_batch(log_seqs)
        time_seqs = _ensure_batch(time_seqs)
        pos_time = _ensure_batch(pos_time)
        time_output = self.time_encoder(time_seqs, log_seqs, log_seqs)
        time_diff, detail, _ = _compute_time_error(
            time_output,
            pos_time,
            self.time_encoder.dev,
            is_dict=isinstance(time_output, dict),
            vm_topk_mode=vm_topk_mode,
        )
        if return_details:
            return detail
        return time_diff


class CaserWrapper(BaseWrapper):
    def __init__(self, model, time_encoder):
        super().__init__(model, supports_time_prediction=True)
        self.time_encoder = time_encoder
        self.time_encoder_type = time_encoder.time_encoder_type

    def forward(self, users, seq, pos, neg, seq_time):
        pos_logits, neg_logits, _ = self.model(seq, users, pos, neg, seq_time, for_pred=False)
        time_output = self.time_encoder(seq_time, seq, seq)
        return pos_logits, neg_logits, time_output

    def predict(self, user_ids, log_seqs, time_seqs, item_indices, min_year):
        return self.model.predict(user_ids, log_seqs, time_seqs, item_indices, min_year)
    
    def predict_time(self, log_seqs, time_seqs, min_year, pos_time, return_details=False, vm_topk_mode='min', **kwargs):
        log_seqs = _ensure_batch(log_seqs)
        time_seqs = _ensure_batch(time_seqs)
        pos_time = _ensure_batch(pos_time)
        time_output = self.time_encoder(time_seqs, log_seqs, log_seqs)
        time_diff, detail, _ = _compute_time_error(
            time_output,
            pos_time,
            self.time_encoder.dev,
            is_dict=isinstance(time_output, dict),
            vm_topk_mode=vm_topk_mode,
        )
        if return_details:
            return detail
        return time_diff


class Bert4RecWrapper(BaseWrapper):
    def __init__(self, model, time_encoder):
        super().__init__(model, supports_time_prediction=True)
        self.time_encoder = time_encoder
        self.time_encoder_type = time_encoder.time_encoder_type

    def forward(self, users, seq, pos, neg, seq_time):
        pos_logits, neg_logits, _ = self.model(seq, pos, neg, seq_time)
        time_output = self.time_encoder(seq_time, seq, seq)
        return pos_logits, neg_logits, time_output

    def predict(self, user_ids, log_seqs, time_seqs, item_indices, min_year):
        return self.model.predict(user_ids, log_seqs, time_seqs, item_indices, min_year)

    def predict_time(self, log_seqs, time_seqs, min_year, pos_time, return_details=False, vm_topk_mode='min', **kwargs):
        log_seqs = _ensure_batch(log_seqs)
        time_seqs = _ensure_batch(time_seqs)
        pos_time = _ensure_batch(pos_time)
        time_output = self.time_encoder(time_seqs, log_seqs, log_seqs)
        time_diff, detail, _ = _compute_time_error(
            time_output,
            pos_time,
            self.time_encoder.dev,
            is_dict=isinstance(time_output, dict),
            vm_topk_mode=vm_topk_mode,
        )
        if return_details:
            return detail
        return time_diff


class TiSASRecWrapper(BaseWrapper):
    def __init__(self, model, time_span, device, time_encoder):
        super().__init__(model, supports_time_prediction=True)
        self.time_span = time_span
        self.device = device
        self.time_encoder = time_encoder
        self.time_encoder_type = time_encoder.time_encoder_type

    def _build_time_matrices(self, time_seqs):
        return _build_time_matrix_batch(time_seqs, self.time_span, self.device)

    def forward(self, users, seq, pos, neg, seq_time):
        time_matrices = self._build_time_matrices(seq_time)
        pos_logits, neg_logits = self.model(users, seq, time_matrices, pos, neg)
        time_output = self.time_encoder(seq_time, seq, seq)
        return pos_logits, neg_logits, time_output

    def predict(self, user_ids, log_seqs, time_seqs, item_indices, min_year):
        time_matrices = self._build_time_matrices(time_seqs)
        return self.model.predict(user_ids, log_seqs, time_matrices, item_indices)
    
    def predict_time(self, log_seqs, time_seqs, min_year, pos_time, return_details=False, vm_topk_mode='min', **kwargs):
        log_seqs = _ensure_batch(log_seqs)
        time_seqs = _ensure_batch(time_seqs)
        pos_time = _ensure_batch(pos_time)
        time_output = self.time_encoder(time_seqs, log_seqs, log_seqs)
        time_diff, detail, _ = _compute_time_error(
            time_output,
            pos_time,
            self.time_encoder.dev,
            is_dict=isinstance(time_output, dict),
            vm_topk_mode=vm_topk_mode,
        )
        if return_details:
            return detail
        return time_diff


class FMLP4RecWrapper(BaseWrapper):
    def __init__(self, model, time_encoder):
        super().__init__(model, supports_time_prediction=True)
        self.time_encoder = time_encoder
        self.time_encoder_type = time_encoder.time_encoder_type

    def forward(self, users, seq, pos, neg, seq_time):
        pos_logits, neg_logits, _ = self.model(seq, pos, neg, seq_time)
        time_output = self.time_encoder(seq_time, seq, seq)
        return pos_logits, neg_logits, time_output

    def predict(self, user_ids, log_seqs, time_seqs, item_indices, min_year):
        return self.model.predict(user_ids, log_seqs, time_seqs, item_indices, min_year)

    def predict_time(self, log_seqs, time_seqs, min_year, pos_time, return_details=False, vm_topk_mode='min', **kwargs):
        log_seqs = _ensure_batch(log_seqs)
        time_seqs = _ensure_batch(time_seqs)
        pos_time = _ensure_batch(pos_time)
        time_output = self.time_encoder(time_seqs, log_seqs, log_seqs)
        time_diff, detail, _ = _compute_time_error(
            time_output,
            pos_time,
            self.time_encoder.dev,
            is_dict=isinstance(time_output, dict),
            vm_topk_mode=vm_topk_mode,
        )
        if return_details:
            return detail
        return time_diff


class GeoSANWrapper(BaseWrapper):
    def __init__(self, model, time_encoder):
        super().__init__(model, supports_time_prediction=True)
        self.time_encoder = time_encoder
        self.time_encoder_type = time_encoder.time_encoder_type

    def forward(self, users, seq, pos, neg, seq_time):
        pos_logits, neg_logits, _ = self.model(seq, pos, neg, seq_time)
        time_output = self.time_encoder(seq_time, seq, seq)
        return pos_logits, neg_logits, time_output

    def predict(self, user_ids, log_seqs, time_seqs, item_indices, min_year):
        return self.model.predict(user_ids, log_seqs, time_seqs, item_indices, min_year)

    def predict_time(self, log_seqs, time_seqs, min_year, pos_time, return_details=False, vm_topk_mode='min', **kwargs):
        log_seqs = _ensure_batch(log_seqs)
        time_seqs = _ensure_batch(time_seqs)
        pos_time = _ensure_batch(pos_time)
        time_output = self.time_encoder(time_seqs, log_seqs, log_seqs)
        time_diff, detail, _ = _compute_time_error(
            time_output,
            pos_time,
            self.time_encoder.dev,
            is_dict=isinstance(time_output, dict),
            vm_topk_mode=vm_topk_mode,
        )
        if return_details:
            return detail
        return time_diff


def build_model(args, usernum, itemnum):
    name = getattr(args, "baseline", "baseline1").lower()
    if name in ("baseline1", "sasrec"):
        return SASRecWrapper(SASRec(itemnum, args).to(args.device))
    if name in ("baseline2", "gru4rec"):
        return GRU4RecWrapper(GRU4Rec(itemnum, args).to(args.device), TimeEncoder(args, itemnum).to(args.device))
    if name in ("baseline3", "caser"):
        return CaserWrapper(Caser(usernum, itemnum, args).to(args.device), TimeEncoder(args, itemnum).to(args.device))
    if name in ("baseline4", "bert4rec", "bert"):
        return Bert4RecWrapper(Bert4Rec(itemnum, args).to(args.device), TimeEncoder(args, itemnum).to(args.device))
    if name in ("baseline5", "tisasrec", "tisas"):
        return TiSASRecWrapper(TiSASRec(usernum, itemnum, args).to(args.device), args.time_span, args.device,
                               TimeEncoder(args, itemnum).to(args.device))
    if name in ("baseline6", "fmlp4rec", "fmlp"):
        return FMLP4RecWrapper(
            FMLP4Rec(
                itemnum,
                args,
                d_model=args.hidden_units,
                max_len=args.maxlen,
                depth=args.num_blocks,
            ).to(args.device),
            TimeEncoder(args, itemnum).to(args.device)
        )
    if name in ("baseline7", "geosan"):
        return GeoSANWrapper(
            GeoSANSeqRec(usernum, itemnum, args).to(args.device),
            TimeEncoder(args, itemnum).to(args.device)
        )
    raise ValueError(f"Unknown baseline: {args.baseline}")
