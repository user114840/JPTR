import sys
import copy
import random
import numpy as np
import os
# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args, near_poi_dict=None, epoch=None, vm_topk_mode='min', summary_filename='time_error_summary.csv', output_dir=None):
    [train, valid, test, train_time, valid_time, test_time, usernum, itemnum, timenum, min_year, num_year, poi_info] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    NDCG2 = 0.0
    HT2 = 0.0
    valid_user = 0.0
    valid_user_time = 0.0
    time_diff = 0.0
    print("->")
    error_records = []
    support_time = getattr(model, 'supports_time_prediction', True)
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break


        seq_time = np.zeros([args.maxlen], dtype=np.int32)
        idx_time = args.maxlen - 1
        seq_time[idx_time] = valid_time[u][0][0]
        idx_time -= 1
        for i in reversed(train_time[u]):
            seq_time[idx_time] = i
            idx_time -= 1
            if idx_time == -1: break
        seq_time_target = np.zeros([args.maxlen], dtype=np.int32)
        idx_time_target = args.maxlen - 1
        seq_time_target[idx_time_target] = test_time[u][0][0]
        idx_time_target -= 1
        seq_time_target[idx_time_target] = valid_time[u][0][0]
        idx_time_target -= 1
        for i in reversed(train_time[u]):
            seq_time_target[idx_time_target] = i
            idx_time_target -= 1
            if idx_time_target == -1: break


        #item_idx = near_poi_dict[u]

        poi_list = list(range(1, itemnum + 1))
        poi_list.remove(test[u][0])
        random.shuffle(poi_list)
        poi_list.insert(0, test[u][0])

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [seq_time], poi_list, min_year]])
        predictions = predictions[0][0]
        rank = predictions.argsort().argsort()[0].item()

        # Optionally free temporary variables to release memory
        # del predictions
        # torch.cuda.empty_cache()


        #if(test_time[u][0][0] - valid_time[u][0][0]) <= 172800 * 5:
        #    time_diff += model.predict_time(seq, seq_time, min_year, seq_time_target)
        #    valid_user_time += 1
        if support_time:
            try:
                detailed = model.predict_time(seq, seq_time, min_year, seq_time_target, return_details=True, vm_topk_mode=vm_topk_mode)
            except TypeError:
                detailed = model.predict_time(seq, seq_time, min_year, seq_time_target, return_details=True)
            if isinstance(detailed, dict):
                time_diff += detailed['avg_error']
                seq_len = detailed['seq_length']
                last_error = float(detailed['per_pos_error'][-1])
                error_records.append((seq_len, last_error))
            else:
                time_diff += float(detailed)
                seq_len = int((np.array(seq) != 0).sum())
                error_records.append((seq_len, float(detailed)))
            valid_user_time += 1
        valid_user += 1

        if rank < 5:
            NDCG2 += 1 / np.log2(rank + 2)
            HT2 += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    if support_time and error_records:
        analyze_time_errors(
            error_records,
            args,
            epoch if epoch is not None else -1,
            split_name='test',
            summary_filename=summary_filename,
            output_dir=output_dir,
        )

    time_avg = time_diff / valid_user_time if valid_user_time > 0 else None
    return NDCG / valid_user, HT / valid_user, time_avg, NDCG2 / valid_user, HT2 / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args, epoch):
    [train, valid, test, train_time, valid_time, test_time, usernum, itemnum, timenum, min_year, num_year, poi_info] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    time_diff = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        seq_time = np.zeros([args.maxlen], dtype=np.int32)
        idx_time = args.maxlen - 1
        for i in reversed(train_time[u]):
            seq_time[idx_time] = i
            idx_time -= 1
            if idx_time == -1: break;

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [seq_time], item_idx, min_year]])
        predictions = predictions[0][0]

        rank = predictions.argsort().argsort()[0].item()


        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def analyze_time_errors(error_records, args, epoch, split_name='test', summary_filename='time_error_summary.csv', output_dir=None):
    errors = np.array([e for _, e in error_records], dtype=np.float32)
    percentiles = np.percentile(errors, [50, 75, 90, 95])
    thresholds = {
        'p_lt_5m': (errors < 300).mean(),
        'p_lt_10m': (errors < 600).mean(),
        'p_lt_15m': (errors < 900).mean(),
        'p_lt_20m': (errors < 1200).mean(),
        'p_lt_30m': (errors < 1800).mean(),
        'p_lt_40m': (errors < 2400).mean(),
        'p_lt_50m': (errors < 3000).mean(),
        'p_lt_1h': (errors < 3600).mean(),
        'p_lt_1.5h': (errors < 5400).mean(),
        'p_lt_2h': (errors < 7200).mean(),
        'p_lt_2.5h': (errors < 9000).mean(),
        'p_lt_3h': (errors < 10800).mean()
    }

    analysis_dir = output_dir if output_dir is not None else args.dataset + '_' + args.train_dir
    os.makedirs(analysis_dir, exist_ok=True)

    summary_path = os.path.join(analysis_dir, summary_filename)
    if not os.path.exists(summary_path):
        with open(summary_path, 'w') as f:
            f.write('epoch,split,median_sec,p75_sec,p90_sec,p95_sec,'
                    'p_lt_5m,p_lt_10m,p_lt_15m,p_lt_20m,p_lt_30m,p_lt_40m,p_lt_50m,'
                    'p_lt_1h,p_lt_1.5h,p_lt_2h,p_lt_2.5h,p_lt_3h\n')

    with open(summary_path, 'a') as f:
        f.write(
            f"{epoch},{split_name},{percentiles[0]:.4f},{percentiles[1]:.4f},"
            f"{percentiles[2]:.4f},{percentiles[3]:.4f},"
            f"{thresholds['p_lt_5m']*100:.2f}%,{thresholds['p_lt_10m']*100:.2f}%,"
            f"{thresholds['p_lt_15m']*100:.2f}%,{thresholds['p_lt_20m']*100:.2f}%,"
            f"{thresholds['p_lt_30m']*100:.2f}%,{thresholds['p_lt_40m']*100:.2f}%,"
            f"{thresholds['p_lt_50m']*100:.2f}%,{thresholds['p_lt_1h']*100:.2f}%,"
            f"{thresholds['p_lt_1.5h']*100:.2f}%,{thresholds['p_lt_2h']*100:.2f}%,"
            f"{thresholds['p_lt_2.5h']*100:.2f}%,{thresholds['p_lt_3h']*100:.2f}%\n"
        )

    print("[Time Analysis] "
          f"Median={percentiles[0]:.2f}s, 75%={percentiles[1]:.2f}s, "
          f"90%={percentiles[2]:.2f}s, 95%={percentiles[3]:.2f}s | "
          f"P(<5m)={thresholds['p_lt_5m']*100:.2f}%, "
          f"P(<10m)={thresholds['p_lt_10m']*100:.2f}%, "
          f"P(<15m)={thresholds['p_lt_15m']*100:.2f}%, "
          f"P(<20m)={thresholds['p_lt_20m']*100:.2f}%, "
          f"P(<30m)={thresholds['p_lt_30m']*100:.2f}%, "
          f"P(<40m)={thresholds['p_lt_40m']*100:.2f}%, "
          f"P(<50m)={thresholds['p_lt_50m']*100:.2f}%, "
          f"P(<1h)={thresholds['p_lt_1h']*100:.2f}%, "
          f"P(<1.5h)={thresholds['p_lt_1.5h']*100:.2f}%, "
          f"P(<2h)={thresholds['p_lt_2h']*100:.2f}%, "
          f"P(<2.5h)={thresholds['p_lt_2.5h']*100:.2f}%, "
          f"P(<3h)={thresholds['p_lt_3h']*100:.2f}%")
