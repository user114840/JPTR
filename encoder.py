# encoder.py
import torch
import torch.nn as nn
import numpy as np
from Baselines.Baseline1.trs_layer import TrsLayer, TrsEncoder
from Baselines.Baseline1.utils import get_mask
from time_utils import timestamp_sequence_to_datetime_sequence_batch


class MultiFreqTimeEncoder(nn.Module):
    """
    Simplified time encoder that keeps only hour/minute/second embeddings and feeds them to the downstream POI branch.
    Output dimension is fixed at 24 (3 * 8) to align with the Mamba branch.
    """

    def __init__(self, args):
        super(MultiFreqTimeEncoder, self).__init__()
        self.dev = args.device
        self.hour_embedding = nn.Embedding(24, 8)
        self.minute_embedding = nn.Embedding(60, 8)
        self.second_embedding = nn.Embedding(60, 8)

    def _extract_time_hms(self, time_seqs):
        secs_day = 24 * 3600
        t = time_seqs.clamp_min(0)
        tod = t % secs_day
        hour = (tod // 3600).long()
        minute = ((tod % 3600) // 60).long()
        second = (tod % 60).long()
        return hour, minute, second

    def forward(self, time_seqs):
        module_device = next(self.parameters()).device
        if isinstance(time_seqs, np.ndarray):
            time_seqs = torch.from_numpy(time_seqs).to(module_device)
        else:
            time_seqs = time_seqs.to(module_device)
        time_seqs = time_seqs.long().contiguous()

        hour, minute, second = self._extract_time_hms(time_seqs)
        encoded_features = torch.cat(
            [
                self.hour_embedding(hour),
                self.minute_embedding(minute),
                self.second_embedding(second),
            ],
            dim=-1,
        )

        mask = (time_seqs.long() > 0).unsqueeze(-1).float()
        encoded_features = encoded_features * mask
        return encoded_features

    @property
    def output_dim(self):
        return (
            self.hour_embedding.embedding_dim
            + self.minute_embedding.embedding_dim
            + self.second_embedding.embedding_dim
        )


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model, dropout_ratio):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, x):
        lookup_table = self.pos_embedding.weight[:x.size(1), :]
        x += lookup_table.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = self.dropout(x)
        return x


class TimeEncoder(nn.Module):
    """
    Time encoder that keeps the original structure; only the Mamba branch is customized.
    """

    def __init__(self, args, n_item):
        super(TimeEncoder, self).__init__()
        self.time_hidden_units1 = args.time_hidden_units1
        self.time_hidden_units2 = args.time_hidden_units2
        self.hidden_units = args.hidden_units
        self.maxlen = args.maxlen
        self.dev = args.device
        self.time_encoder_type = args.time_encoder_type
        self.time_encoder_layers = args.time_encoder_layers
        self.time_encoder_heads = args.time_encoder_heads
        self.time_encoder_dropout = args.time_encoder_dropout

        # POI embedding layer
        self.poi_embedding = nn.Embedding(n_item + 1, args.hidden_units, padding_idx=0)

        # Initialize by encoder type
        if self.time_encoder_type == 'transformer':
            # Transformer time encoder (keeps original logic)
            self.hour_embedding_layer = torch.nn.Embedding(num_embeddings=24, embedding_dim=args.time_hidden_units1)
            self.minute_embedding_layer = torch.nn.Embedding(num_embeddings=60, embedding_dim=args.time_hidden_units2)
            self.second_embedding_layer = torch.nn.Embedding(num_embeddings=60, embedding_dim=args.time_hidden_units2)

            # Transformer layers
            input_dim = args.hidden_units + args.time_hidden_units1 + args.time_hidden_units2 * 2
            self.time_trs_layer = TrsLayer(input_dim, self.time_encoder_heads, args.exp_factor, self.time_encoder_dropout)
            self.time_trs_encoder = TrsEncoder(input_dim, self.time_trs_layer, self.time_encoder_layers)

            # Time predictor
            self.time_predictor = TimePredictor(args, self.time_encoder_type)

        elif self.time_encoder_type == 'mamba':
            # Mamba time encoder with multi-frequency embeddings
            self.multi_freq_encoder = MultiFreqTimeEncoder(args)
            from mamba_adapter import MambaTimePredictor
            self.mamba_predictor = MambaTimePredictor(args, self.multi_freq_encoder)

    def forward(self, time_seqs, poi_seqs, seq_mask=None, poi_context=None):
        """Forward pass for the time encoder"""
        if isinstance(poi_seqs, np.ndarray):
            poi_tensor = torch.from_numpy(poi_seqs).to(self.dev)
        else:
            poi_tensor = torch.as_tensor(poi_seqs, device=self.dev)
        if poi_context is not None and self.time_encoder_type == 'mamba':
            poi_embeddings = poi_context.to(self.dev)
        else:
            poi_embeddings = self.poi_embedding(poi_tensor.long())

        if self.time_encoder_type == 'transformer':
            # Transformer path (unchanged)
            datetime_sequences = timestamp_sequence_to_datetime_sequence_batch(time_seqs)
            input_data = []
            for datetime_sequence in datetime_sequences:
                sequence_data = []
                for dt in datetime_sequence:
                    if dt is None:
                        sequence_data.append([0, 0, 0])
                    else:
                        sequence_data.append([dt.hour, dt.minute, dt.second])
                input_data.append(sequence_data)

            input_data_tensor = torch.tensor(input_data).to(self.dev)

            embedded_hour = self.hour_embedding_layer(input_data_tensor[:, :, 0:1])
            embedded_minute = self.minute_embedding_layer(input_data_tensor[:, :, 1:2])
            embedded_second = self.second_embedding_layer(input_data_tensor[:, :, 2:3])

            embedded_time = torch.cat((embedded_hour, embedded_minute, embedded_second), dim=-1)
            batch_size, sequence_length, class_num, vocab_size = embedded_time.shape
            embedded_time = embedded_time.view(batch_size, sequence_length, -1)

            combined_features = torch.cat((poi_embeddings, embedded_time), dim=2)

            if seq_mask is None:
                seq_mask = torch.ones(combined_features.size(0), combined_features.size(1)).to(self.dev)
            mask = get_mask(seq_mask, bidirectional=False, device=self.dev)
            encoded_output = self.time_trs_encoder(combined_features, mask)

            time_output = self.time_predictor(encoded_output)
            return time_output

        elif self.time_encoder_type == 'mamba':
            # Ensure time sequence has batch dimension
            if isinstance(time_seqs, np.ndarray):
                time_tensor = torch.from_numpy(time_seqs).to(self.dev)
            else:
                time_tensor = torch.as_tensor(time_seqs, device=self.dev)
            if time_tensor.dim() == 1:
                time_tensor = time_tensor.unsqueeze(0)
            # Mamba path returns Von Mises mixture parameters
            return self.mamba_predictor(poi_embeddings, time_tensor)


class POIEncoder(nn.Module):
    """
    Standalone POI encoder that mirrors the original logic.
    """

    def __init__(self, args, n_item):
        super(POIEncoder, self).__init__()
        self.hidden_units = args.hidden_units
        self.time_hidden_units1 = args.time_hidden_units1
        self.time_hidden_units2 = args.time_hidden_units2
        self.maxlen = args.maxlen
        self.dev = args.device

        # POI embedding layers
        self.poi_embedding = nn.Embedding(n_item + 1, args.hidden_units, padding_idx=0)
        self.pos_embedding = PositionalEmbedding(args.maxlen, args.hidden_units, args.dropout_rate)

        # Time feature embeddings
        self.hour_embedding_layer = torch.nn.Embedding(num_embeddings=24, embedding_dim=args.time_hidden_units1)
        self.minute_embedding_layer = torch.nn.Embedding(num_embeddings=60, embedding_dim=args.time_hidden_units2)
        self.second_embedding_layer = torch.nn.Embedding(num_embeddings=60, embedding_dim=args.time_hidden_units2)

        # POI encoder
        self.poi_trs_layer = TrsLayer(
            args.hidden_units + args.time_hidden_units1 * 1 + args.time_hidden_units2 * 2,
            args.num_heads,
            args.exp_factor,
            args.dropout_rate,
        )
        self.poi_trs_encoder = TrsEncoder(
            args.hidden_units + args.time_hidden_units1 * 1 + args.time_hidden_units2 * 2,
            self.poi_trs_layer,
            args.num_blocks,
        )

    def extract_poi_features(self, poi_seqs, time_seqs):
        """Extract POI features and time embeddings"""
        embedded_poi = self.poi_embedding(torch.LongTensor(poi_seqs).to(self.dev))
        embedded_poi = self.pos_embedding(embedded_poi)

        datetime_sequences = timestamp_sequence_to_datetime_sequence_batch(time_seqs)
        input_data = []
        for datetime_sequence in datetime_sequences:
            sequence_data = []
            for dt in datetime_sequence:
                if dt is None:
                    sequence_data.append([0, 0, 0])
                else:
                    sequence_data.append([dt.hour, dt.minute, dt.second])
            input_data.append(sequence_data)

        input_data_tensor = torch.tensor(input_data).to(self.dev)

        embedded_hour = self.hour_embedding_layer(input_data_tensor[:, :, 0:1])
        embedded_minute = self.minute_embedding_layer(input_data_tensor[:, :, 1:2])
        embedded_second = self.second_embedding_layer(input_data_tensor[:, :, 2:3])

        embedded_time = torch.cat((embedded_hour, embedded_minute, embedded_second), dim=-1)
        batch_size, sequence_length, class_num, vocab_size = embedded_time.shape
        embedded_time = embedded_time.view(batch_size, sequence_length, -1)

        combined_features = torch.cat((embedded_poi, embedded_time), dim=2)

        return combined_features

    def forward(self, poi_seqs, time_seqs, seq_mask=None):
        """Forward pass for the POI encoder"""
        features = self.extract_poi_features(poi_seqs, time_seqs)

        if seq_mask is None:
            seq_mask = torch.ones(features.size(0), features.size(1)).to(self.dev)
        mask = get_mask(seq_mask, bidirectional=False, device=self.dev)
        encoded_output = self.poi_trs_encoder(features, mask)

        return encoded_output


class TimePredictor(nn.Module):
    """
    Time predictor used with the Transformer-based time encoder.
    """

    def __init__(self, args, time_encoder_type):
        super(TimePredictor, self).__init__()
        self.dev = args.device

        input_dim = args.hidden_units + args.time_hidden_units1 + args.time_hidden_units2 * 2
        self.fc = nn.Linear(input_dim, 3)
        self.act = nn.Sigmoid()

    def forward(self, encoded_features):
        """Forward pass for time prediction"""
        batch_size, seq_len, feature_dim = encoded_features.shape

        flatten_output = encoded_features.contiguous().view(-1, feature_dim)
        linear_output = self.fc(flatten_output)
        output = linear_output.view(batch_size, seq_len, -1)
        output = self.act(output)

        # Scale to real-world time range
        scale_factors = torch.tensor([23.0, 59.0, 59.0]).to(self.dev)
        output = output * scale_factors.unsqueeze(0).unsqueeze(0)

        return output
