import torch
import numpy as np
from typing import List, Optional
from datetime import datetime
import math


class Preprocessor:
    """
    Data preprocessor that centralizes all preprocessing logic while staying compatible with the original code.
    """

    def __init__(self, config):
        """
        Initialize the preprocessor.

        Args:
            config: configuration object containing maxlen and other parameters
        """
        self.config = config
        self.maxlen = config.maxlen

    def process_time_sequences(self, time_sequences: List[List[int]]) -> torch.Tensor:
        """
        Convert timestamp sequences to model-ready (h, m, s) tensors.

        Args:
            time_sequences: list of timestamp sequences

        Returns:
            torch.Tensor: time feature tensor
        """
        datetime_sequences = self._timestamp_sequence_to_datetime_sequence_batch(time_sequences)
        input_data = []

        for datetime_sequence in datetime_sequences:
            sequence_data = []
            for dt in datetime_sequence:
                if dt is None:
                    sequence_data.append([0, 0, 0])  # padding
                else:
                    sequence_data.append([dt.hour, dt.minute, dt.second])
            input_data.append(sequence_data)

        return torch.tensor(input_data).float()

    def pad_sequences(self, sequences: List[List[int]], value: int = 0) -> np.ndarray:
        """
        Pad sequences to a fixed length while keeping original logic.

        Args:
            sequences: raw sequence list
            value: pad value

        Returns:
            np.ndarray: padded sequences
        """
        padded_sequences = np.zeros((len(sequences), self.maxlen), dtype=np.int32)

        for i, seq in enumerate(sequences):
            if len(seq) == 0:
                continue

            if len(seq) > self.maxlen:
                padded_sequences[i] = seq[-self.maxlen:]
            else:
                padded_sequences[i, -len(seq):] = seq

        return padded_sequences

    def create_time_matrix(self, time_seq: List[int], time_span: int) -> np.ndarray:
        """
        Create a time relation matrix (original computeRePos logic).

        Args:
            time_seq: time sequence
            time_span: time-span threshold

        Returns:
            np.ndarray: time relation matrix
        """
        time_seq = torch.LongTensor(time_seq)
        size = time_seq.shape[0]
        time_matrix = np.zeros([size, size], dtype=np.int32)

        for i in range(size):
            for j in range(size):
                span = abs(time_seq[i] - time_seq[j])
                if span > time_span:
                    time_matrix[i][j] = time_span
                else:
                    time_matrix[i][j] = span

        return time_matrix

    def normalize_time_features(self, time_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize time features.

        Args:
            time_tensor: raw time features

        Returns:
            torch.Tensor: normalized features
        """
        device = time_tensor.device
        weights = torch.tensor([1, 1 / 60, 1 / 3600], dtype=torch.float32).to(device)
        return torch.sum(time_tensor * weights, dim=2)

    def _timestamp_to_datetime(self, timestamp: int) -> Optional[datetime]:
        """Convert a timestamp to datetime; treat 0 as padding."""
        if timestamp == 0:
            return None
        return datetime.fromtimestamp(timestamp)

    def _timestamp_sequence_to_datetime_sequence_batch(self, time_sequences: List[List[int]]) -> List[
        List[Optional[datetime]]]:
        """Batch conversion of timestamp sequences to datetime sequences."""
        result = []
        for seq in time_sequences:
            datetime_seq = []
            for ts in seq:
                datetime_seq.append(self._timestamp_to_datetime(ts))
            result.append(datetime_seq)
        return result

    def compute_circular_time_loss(self, true_time_tensor, pred_time_tensor):
        """
        Compute circular time loss, accounting for temporal periodicity.
        Args:
            true_time_tensor: ground truth time [batch_size, seq_len, 3] (h, m, s)
            pred_time_tensor: predicted time [batch_size, seq_len, 3] (h, m, s)
        Returns:
            loss: circular time loss
        """
        device = true_time_tensor.device

        true_seconds = true_time_tensor[:, :, 0] * 3600 + true_time_tensor[:, :, 1] * 60 + true_time_tensor[:, :, 2]
        pred_seconds = pred_time_tensor[:, :, 0] * 3600 + pred_time_tensor[:, :, 1] * 60 + pred_time_tensor[:, :, 2]

        direct_error = torch.abs(true_seconds - pred_seconds)
        circular_error = 86400 - torch.abs(true_seconds - pred_seconds)  # 24 hours = 86400 seconds

        min_error = torch.min(direct_error, circular_error)

        return min_error.mean()

    def normalize_time_features_circular(self, time_tensor):
        """
        Circularly normalize time features by mapping to the unit circle.
        Args:
            time_tensor: time tensor [batch_size, seq_len, 3] (h, m, s)
        Returns:
            normalized: circular features [batch_size, seq_len, 2] (sin, cos)
        """
        device = time_tensor.device

        total_seconds = time_tensor[:, :, 0] * 3600 + time_tensor[:, :, 1] * 60 + time_tensor[:, :, 2]
        normalized_seconds = total_seconds / 86400.0  # 24 hours = 86400 seconds

        angle = 2 * math.pi * normalized_seconds  # [0, 2π]
        sin_component = torch.sin(angle)
        cos_component = torch.cos(angle)

        return torch.stack([sin_component, cos_component], dim=-1)


class TimeRelationGenerator:
    """
    Generator for time relation matrices, wrapping the original Relation logic.
    """

    def __init__(self, time_span: int):
        self.time_span = time_span

    def generate_relation_matrices(self, user_train_time: dict, usernum: int) -> dict:
        """
        Build time relation matrices for all users.

        Args:
            user_train_time: training time data per user
            usernum: number of users

        Returns:
            dict mapping user id to relation matrix
        """
        data_train = {}
        for user in range(1, usernum + 1):
            if user in user_train_time:
                time_seq = user_train_time[user]
                data_train[user] = self._compute_repos(time_seq, self.time_span)
        return data_train

    def _compute_repos(self, time_seq: List[int], time_span: int) -> np.ndarray:
        """Implementation of the original computeRePos function."""
        time_seq = torch.LongTensor(time_seq)
        size = time_seq.shape[0]
        time_matrix = np.zeros([size, size], dtype=np.int32)

        for i in range(size):
            for j in range(size):
                span = abs(time_seq[i] - time_seq[j])
                time_matrix[i][j] = min(span, time_span)

        return time_matrix
