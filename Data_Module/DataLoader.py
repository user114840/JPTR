import os
import random
import numpy as np
import datetime
import math
from collections import defaultdict
from typing import List, Dict, Tuple

import torch

from .Dataset import TAPTDataset


class DataLoader:
    def __init__(self, dataset_name: str = None, time_gap_threshold: int = 0, min_session_length: int = 0):
        self.dataset_name = dataset_name
        self.time_gap_threshold = time_gap_threshold
        self.min_session_length = min_session_length

    def load_data(self, dataset_name: str = None) -> TAPTDataset:
        """
        Load data and return a Dataset instance.

        Args:
            dataset_name: dataset name

        Returns:
            TAPTDataset: dataset instance
        """
        if dataset_name is not None:
            self.dataset_name = dataset_name

        if self.dataset_name is None:
            raise ValueError("Dataset name must be provided")

        # Call the original data_partition logic
        data_components = self._data_partition(self.dataset_name)

        return TAPTDataset(data_components)

    def _data_partition(self, fname: str) -> List:
        """
        Full implementation of the original data_partition logic.
        """
        usernum = 0
        itemnum = 0
        timenum_set = set()
        User = defaultdict(list)
        UserTime = defaultdict(list)
        user_train = {}
        user_valid = {}
        user_test = {}
        user_train_time = {}
        user_valid_time = {}
        user_test_time = {}
        timestamp_list = []
        poi_info = {}  # Dictionary storing POI info

        # Assume user/item indices start from 1
        with open(f'data/{fname}.txt', 'r') as f:
            for line in f:
                data = line.rstrip().split('\t')
                u = int(data[0])
                i_raw = data[-1]
                time = int(data[1])
                try:
                    i = int(i_raw)
                except ValueError:
                    i = int(float(i_raw))
                lat = float(data[2])
                lng = float(data[3])
                usernum = max(u, usernum)
                itemnum = max(i, itemnum)
                User[u].append(i)
                UserTime[u].append(time)
                if i not in poi_info:
                    poi_info[i] = {'latitude': lat, 'longitude': lng}
                timenum_set.add(time)
                timestamp_list.append(time)

        # Sort timestamps and build timestamp-to-index map
        sorted_timestamps = sorted(timestamp_list)

        # Compute timestamp-related stats
        datetime_sequence = self._timestamp_sequence_to_datetime_sequence(sorted_timestamps)
        input_data = [[dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second] for dt in datetime_sequence]

        input_data_tensor = torch.tensor(input_data)
        min_year = torch.min(input_data_tensor[:, 0]).item()
        max_year = torch.max(input_data_tensor[:, 0]).item()
        num_year = max_year - min_year + 1

        timestamp_to_index = {}
        index = 1
        for timestamp in sorted_timestamps:
            if timestamp not in timestamp_to_index:
                timestamp_to_index[timestamp] = index
                index += 1

        time_gap_threshold = getattr(self, 'time_gap_threshold', 0)
        min_session_length = getattr(self, 'min_session_length', 0)
        if time_gap_threshold and time_gap_threshold > 0:
            User, UserTime = self._split_sequences_by_gap(User, UserTime, time_gap_threshold, min_session_length)
            usernum = len(User)

        for user in User:
            nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
                user_train_time[user] = UserTime[user]
                user_valid_time[user] = []
                user_test_time[user] = []
            else:
                user_train[user] = User[user][:-2]
                user_valid[user] = [User[user][-2]]
                user_test[user] = [User[user][-1]]

                user_train_time[user] = UserTime[user][:-2]
                user_valid_time[user] = [UserTime[user][-2:]]
                user_test_time[user] = [UserTime[user][-1:]]

        timenum = len(timenum_set)

        return [user_train, user_valid, user_test, user_train_time, user_valid_time, user_test_time,
                usernum, itemnum, timenum, min_year, num_year, poi_info]

    def _split_sequences_by_gap(self, user_items, user_times, threshold, min_length):
        new_user_items = {}
        new_user_times = {}
        new_user_id = 1
        min_len = max(1, min_length)
        for user in sorted(user_items.keys()):
            items = user_items[user]
            times = user_times[user]
            if not items:
                continue
            pairs = sorted(zip(times, items), key=lambda x: x[0])
            session_items = []
            session_times = []
            prev_ts = None
            for ts, item in pairs:
                if prev_ts is not None and threshold and (ts - prev_ts) > threshold:
                    if len(session_items) >= min_len:
                        new_user_items[new_user_id] = session_items
                        new_user_times[new_user_id] = session_times
                        new_user_id += 1
                    session_items = []
                    session_times = []
                session_items.append(item)
                session_times.append(ts)
                prev_ts = ts
            if len(session_items) >= min_len:
                new_user_items[new_user_id] = session_items
                new_user_times[new_user_id] = session_times
                new_user_id += 1
        return new_user_items, new_user_times

    def _timestamp_to_datetime(self, timestamp: int) -> datetime.datetime:
        """Convert timestamp to datetime."""
        return datetime.datetime.fromtimestamp(timestamp)

    def _timestamp_sequence_to_datetime_sequence(self, timestamp_sequence: List[int]) -> List[datetime.datetime]:
        """Convert a sequence of timestamps to datetimes."""
        return [self._timestamp_to_datetime(ts) for ts in timestamp_sequence]
