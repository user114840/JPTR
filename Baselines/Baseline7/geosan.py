# GeoSAN-based backbone adapted to this codebase
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _generate_square_mask(sz: int, device: torch.device):
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).T
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
    return mask


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, num_units: int, zeros_pad: bool = True, scale: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale
        self.lookup_table = nn.Parameter(torch.Tensor(vocab_size, num_units))
        nn.init.xavier_normal_(self.lookup_table.data)
        if self.zeros_pad:
            self.lookup_table.data[0, :].fill_(0)

    def forward(self, inputs):
        padding_idx = 0 if self.zeros_pad else -1
        outputs = F.embedding(inputs, self.lookup_table, padding_idx, None, 2, False, False)
        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)
        return outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class GeoSANBackbone(nn.Module):
    """
    Simplified GeoSAN (LocPredictor) for POI-only setting.
    """

    def __init__(self, nuser, nloc, emb_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.emb_user = Embedding(nuser, emb_dim, zeros_pad=True, scale=True)
        self.emb_loc = Embedding(nloc, emb_dim, zeros_pad=True, scale=True)
        self.emb_reg = Embedding(1, emb_dim, zeros_pad=True, scale=True)
        self.emb_time = Embedding(1, emb_dim, zeros_pad=True, scale=True)

        self.pos_encoder = PositionalEncoding(emb_dim, dropout)
        self.enc_layer = nn.TransformerEncoderLayer(emb_dim, num_heads, emb_dim * 2, dropout)
        self.encoder = nn.TransformerEncoder(self.enc_layer, num_layers)

        ident_mat = torch.eye(emb_dim)
        self.register_buffer("ident_mat", ident_mat)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(
        self,
        src_user: torch.Tensor,
        src_loc: torch.Tensor,
        src_reg: torch.Tensor,
        src_time: torch.Tensor,
        src_square_mask: torch.Tensor,
        src_binary_mask: torch.Tensor,
        trg_loc: torch.Tensor,
        mem_mask=None,
        ds: List[int] = None,
    ):
        loc_emb_src = self.emb_loc(src_loc)
        src = loc_emb_src * math.sqrt(loc_emb_src.size(-1))
        src = self.pos_encoder(src)
        src = self.encoder(src, mask=src_square_mask)

        loc_emb_trg = self.emb_loc(trg_loc)

        if self.training:
            output = src.repeat(loc_emb_trg.size(0) // src.size(0), 1, 1)
        else:
            ds_tensor = torch.as_tensor([max(1, int(d)) for d in ds], device=src.device, dtype=torch.long)
            src = src[ds_tensor - 1, torch.arange(len(ds_tensor), device=src.device), :]
            output = src.unsqueeze(0).repeat(loc_emb_trg.size(0), 1, 1)

        output = torch.sum(output * loc_emb_trg, dim=-1)
        return output


class GeoSANSeqRec(nn.Module):
    """
    GeoSAN backbone adapted to the unified training loop.
    """

    def __init__(self, n_user: int, n_item: int, args):
        super().__init__()
        self.dev = args.device
        self.supports_time_prediction = False
        self.hidden = args.hidden_units
        self.maxlen = args.maxlen

        self.backbone = GeoSANBackbone(
            nuser=n_user + 1,
            nloc=n_item + 1,
            emb_dim=args.hidden_units,
            num_heads=args.num_heads,
            num_layers=args.num_blocks,
            dropout=args.dropout_rate,
        ).to(self.dev)

    def _left_align(self, tensor: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        aligned = torch.zeros_like(tensor)
        for i, l in enumerate(lengths):
            if l > 0:
                aligned[i, :l] = tensor[i, -l:]
        return aligned

    def _prepare_sequences(
        self, seq, pos, neg, seq_time
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        seq_tensor = torch.as_tensor(seq, device=self.dev, dtype=torch.long)
        pos_tensor = torch.as_tensor(pos, device=self.dev, dtype=torch.long)
        neg_tensor = torch.as_tensor(neg, device=self.dev, dtype=torch.long)
        time_tensor = torch.as_tensor(seq_time, device=self.dev, dtype=torch.long)

        if seq_tensor.dim() == 1:
            seq_tensor = seq_tensor.unsqueeze(0)
            pos_tensor = pos_tensor.unsqueeze(0)
            neg_tensor = neg_tensor.unsqueeze(0)
            time_tensor = time_tensor.unsqueeze(0)

        lengths = (seq_tensor > 0).sum(dim=1).tolist()
        seq_aligned = self._left_align(seq_tensor, lengths)
        pos_aligned = self._left_align(pos_tensor, lengths)
        neg_aligned = self._left_align(neg_tensor, lengths)
        return seq_aligned, pos_aligned, neg_aligned, lengths

    def forward(self, seq, pos_seqs, neg_seqs, time_seqs):
        seq_aligned, pos_aligned, neg_aligned, lengths = self._prepare_sequences(seq, pos_seqs, neg_seqs, time_seqs)
        batch, seq_len = seq_aligned.shape

        src_loc = seq_aligned.transpose(0, 1)
        src_user = torch.zeros_like(src_loc)
        src_reg = torch.zeros_like(src_loc)
        src_time = torch.zeros_like(src_loc)

        trg_loc = torch.stack([pos_aligned, neg_aligned], dim=2)  # (B, L, 2)
        trg_loc = trg_loc.permute(2, 1, 0).contiguous().view(-1, batch)

        src_square_mask = _generate_square_mask(seq_len, src_loc.device)
        src_binary_mask = (src_loc.transpose(0, 1) == 0)

        output = self.backbone(src_user, src_loc, src_reg, src_time, src_square_mask, src_binary_mask, trg_loc)
        output = output.view(2, seq_len, batch).permute(2, 1, 0)  # (B, L, 2)
        pos_logits_left = output[:, :, 0]
        neg_logits_left = output[:, :, 1]

        pos_logits = torch.zeros_like(seq_aligned, dtype=pos_logits_left.dtype)
        neg_logits = torch.zeros_like(seq_aligned, dtype=neg_logits_left.dtype)
        for i, l in enumerate(lengths):
            if l > 0:
                pos_logits[i, -l:] = pos_logits_left[i, :l]
                neg_logits[i, -l:] = neg_logits_left[i, :l]
        return pos_logits, neg_logits, None

    def predict(self, user_ids, log_seqs, time_seqs, item_indices, min_year):
        log_tensor = torch.as_tensor(log_seqs, device=self.dev, dtype=torch.long)
        time_tensor = torch.as_tensor(time_seqs, device=self.dev, dtype=torch.long)
        if log_tensor.dim() == 1:
            log_tensor = log_tensor.unsqueeze(0)
            time_tensor = time_tensor.unsqueeze(0)
        lengths = (log_tensor > 0).sum(dim=1).tolist()
        log_aligned = self._left_align(log_tensor, lengths)

        batch, seq_len = log_aligned.shape
        src_loc = log_aligned.transpose(0, 1)
        src_user = torch.zeros_like(src_loc)
        src_reg = torch.zeros_like(src_loc)
        src_time = torch.zeros_like(src_loc)

        cand = torch.as_tensor(item_indices, device=self.dev, dtype=torch.long)
        if cand.dim() == 1:
            trg_loc = cand.unsqueeze(1).repeat(1, batch)
        else:
            trg_loc = cand.transpose(0, 1).contiguous()

        src_square_mask = _generate_square_mask(seq_len, src_loc.device)
        src_binary_mask = (src_loc.transpose(0, 1) == 0)

        output = self.backbone(
            src_user,
            src_loc,
            src_reg,
            src_time,
            src_square_mask,
            src_binary_mask,
            trg_loc,
            mem_mask=None,
            ds=lengths,
        )
        logits = output.transpose(0, 1)  # (B, num_candidates)
        return logits.unsqueeze(1)  # match expected shape (B, 1, num_candidates)

    def predict_time(self, log_seqs, time_seqs, min_year, pos_time, return_details=False):
        raise NotImplementedError("GeoSAN baseline does not support time prediction")
