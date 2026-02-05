import os
import time
import torch
import argparse
import pickle
import random
import math
import numpy as np
import multiprocessing as mp
from torch.special import i0
from config import Config

from evaluator import *
from Data_Module.DataLoader import DataLoader
from Data_Module.Preprocessor import Preprocessor, TimeRelationGenerator
from Data_Module.warp_sampler import WarpSampler
from model_factory import build_model


if __name__ == '__main__':
    config = Config()
    config.load_from_command_line()
    args = config

    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()

    data_loader = DataLoader(time_gap_threshold=args.time_gap_threshold,
                             min_session_length=args.min_session_length)
    dataset = data_loader.load_data(args.dataset)
    # Validate dataset integrity
    if not dataset.validate():
        print("Warning: dataset integrity validation failed")
        # Continue running or exit depending on user decision
    else:
        print("Dataset validation passed")

    [user_train, user_valid, user_test, user_train_time, user_valid_time, user_test_time, usernum, itemnum, timenum,
     min_year, num_year, poi_info] = dataset
    nearest_pois_dict = {}

    near_poi_dict = {}
    num_batch = len(user_train) // args.batch_size 

    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

    sampler = WarpSampler(user_train, user_train_time, usernum, itemnum, nearest_pois_dict, batch_size=args.batch_size,
                          maxlen=args.maxlen,
                          n_workers=3)

    preprocessor = Preprocessor(args)

    model = build_model(args, usernum, itemnum)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers

    model.train()

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:  # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb
            pdb.set_trace()

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args, near_poi_dict, epoch=0)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
        print('time_diff: ', t_test[2] if t_test[2] is not None else 'N/A (time head not supported)')

    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    mse_criterion = torch.nn.MSELoss()

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()
    min_miss = 0
    min_num = 0

    def compute_circular_time_loss(true_time_tensor, pred_time_tensor):
        """
        Compute circular time loss, accounting for periodicity.
        Args:
            true_time_tensor: ground truth times [batch_size, seq_len, 3] (h, m, s)
            pred_time_tensor: predicted times [batch_size, seq_len, 3] (h, m, s)
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

    def normalize_time_features_circular(time_tensor):
        """
        Normalize time features onto the unit circle (sin, cos).
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

    TWO_PI = 2 * math.pi
    EPS = 1e-8

    def time_tensor_to_angle_minutes(time_tensor):
        total_minutes = time_tensor[:, :, 0] * 60 + time_tensor[:, :, 1] + time_tensor[:, :, 2] / 60.0
        return TWO_PI * (total_minutes / 1440.0)

    def von_mises_topk_times(vm_params, topk=2):
        pi = vm_params['pi']
        mu = vm_params['mu'] % TWO_PI
        k = min(topk, pi.size(-1))
        if k <= 0:
            raise ValueError("topk must be positive")
        topk_pi, indices = torch.topk(pi, k=k, dim=-1)
        topk_mu = torch.gather(mu, -1, indices)
        minutes = topk_mu / TWO_PI * 1440.0
        hours = torch.floor(minutes / 60.0)
        mins = torch.floor(minutes % 60.0)
        secs = (minutes - torch.floor(minutes)) * 60.0
        return torch.stack([hours, mins, secs], dim=-1)

    def von_mises_nll(vm_params, theta, mask):
        theta = theta.unsqueeze(-1)
        log_pi = torch.log(vm_params['pi'] + EPS)
        log_i0 = torch.log(i0(vm_params['kappa']) + EPS)
        log_component = log_pi + vm_params['kappa'] * torch.cos(theta - vm_params['mu']) - math.log(2 * math.pi) - log_i0
        log_prob = torch.logsumexp(log_component, dim=-1)
        return -(log_prob.masked_select(mask)).mean()

    POI_LOSS_WEIGHT = 10.0
    TIME_LOSS_WEIGHT_VM = 1.0
    TIME_LOSS_WEIGHT_REG = 10.0

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break  # just to decrease identition
        for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg, seq_time, pos_time = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg, seq_time, pos_time = np.array(u), np.array(seq), np.array(pos), np.array(neg), np.array(
                seq_time), np.array(pos_time)

            pos_logits, neg_logits, output = model(u, seq, pos, neg, seq_time)
            true_time_tensor = preprocessor.process_time_sequences(pos_time).to(args.device)
            seq_mask_tensor = torch.as_tensor(seq, device=args.device) > 0

            display_time = None
            time_loss = torch.tensor(0.0, device=args.device)
            if output is not None:
                if isinstance(output, dict):
                    vm_params = {k: v for k, v in output.items() if k in ('pi', 'mu', 'kappa')}
                    theta_true = time_tensor_to_angle_minutes(true_time_tensor)
                    time_loss = von_mises_nll(vm_params, theta_true, seq_mask_tensor)
                    topk_pred_times = von_mises_topk_times(vm_params, topk=2).detach()
                    display_time = topk_pred_times[:, :, 0, :]
                else:
                    display_time = output.float()

            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                   device=args.device)
            adam_optimizer.zero_grad()

            indices = np.where(pos != 0)
            poi_loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            poi_loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            if output is None:
                loss_weight = 0.0
            elif isinstance(output, dict):
                loss_weight = TIME_LOSS_WEIGHT_VM
            else:
                true_circular = normalize_time_features_circular(true_time_tensor)
                pred_circular = normalize_time_features_circular(display_time)
                mask = seq_mask_tensor.unsqueeze(-1).float()
                mse_per_pos = (true_circular - pred_circular) ** 2
                masked_mse = (mse_per_pos * mask).sum()
                valid_count = mask.sum().clamp_min(1.0)
                time_loss = masked_mse / valid_count
                loss_weight = TIME_LOSS_WEIGHT_REG

            total_loss = loss_weight * time_loss + POI_LOSS_WEIGHT * poi_loss
            total_loss.backward()
            adam_optimizer.step()

            print("loss in epoch {} iteration {}: time_loss={:.4f}, poi_loss={:.4f}, total_loss={:.4f}".format(
                epoch, step, time_loss.item() if isinstance(time_loss, torch.Tensor) else float(time_loss),
                poi_loss.item(), total_loss.item()))

        if epoch % 100 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args, near_poi_dict, epoch)
            t_valid = evaluate_valid(model, dataset, args, epoch)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
                epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
            print('time_diff: ', t_test[2] if t_test[2] is not None else 'N/A (time head not supported)')

            f.write(str(t_valid) + ' ' + str(t_test) + ' ' + '\n')
            f.flush()
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = '{}.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.baseline if hasattr(args, 'baseline') else 'SASRec',
                                 args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units,
                                 args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    sampler.close()
    print("Done")
