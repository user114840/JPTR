import torch
import numpy as np
from tqdm import tqdm
import datetime


def computeRePos(time_seq, time_span):  # compute relation matrix (Eq. 2)
    time_seq = torch.LongTensor(time_seq).to("cuda:0")
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i] - time_seq[j])
            if span > time_span:        # clip values beyond threshold
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix


def Relation(user_train_time, usernum, time_span):   # return relation matrices for all users
    data_train = dict()
    for user in tqdm(range(1, usernum + 1), desc='Preparing relation matrix'):
        time_seq = user_train_time[user]
        data_train[user] = computeRePos(time_seq, time_span)
    return data_train

def timestamp_to_datetime(timestamp):
    return datetime.datetime.fromtimestamp(timestamp)

def timestamp_sequence_to_datetime_sequence(timestamp_sequence):
    return [timestamp_to_datetime(ts) for ts in timestamp_sequence]

def timestamp_sequence_to_datetime_sequence_batch(timestamp_sequences):
    datetime_sequences = []
    for i in range(len(timestamp_sequences)):
        timestamp_sequence = timestamp_sequences[i]
        #print(timestamp_sequence.shape)
        datetime_sequence = timestamp_sequence_to_datetime_sequence_batch0(timestamp_sequence)
        datetime_sequences.append(datetime_sequence)
    return datetime_sequences

def timestamp_sequence_to_datetime_sequence_batch0(timestamp_sequence):
    datetime_sequence = []
    #print(timestamp_sequence.shape)
    for ts in timestamp_sequence:
        if ts != 0:
            dt = timestamp_to_datetime(ts)
            datetime_sequence.append(dt)
            #print(datetime_sequence)
        else:
            datetime_sequence.append(None)
    #print(len(datetime_sequence))
    return datetime_sequence

