import datetime
from typing import List, Optional


def _timestamp_to_datetime(timestamp: int) -> Optional[datetime.datetime]:
    if timestamp == 0:
        return None
    return datetime.datetime.fromtimestamp(timestamp)


def timestamp_sequence_to_datetime_sequence(timestamp_sequence):
    return [_timestamp_to_datetime(ts) for ts in timestamp_sequence]


def timestamp_sequence_to_datetime_sequence_batch(timestamp_sequences):
    datetime_sequences = []
    for timestamp_sequence in timestamp_sequences:
        datetime_sequence = timestamp_sequence_to_datetime_sequence(timestamp_sequence)
        datetime_sequences.append(datetime_sequence)
    return datetime_sequences


def timestamp_sequence_to_datetime_sequence_batch0(timestamp_sequence):
    return timestamp_sequence_to_datetime_sequence(timestamp_sequence)
