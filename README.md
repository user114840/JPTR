# JPTR: Time-Aware POI Sequential Recommendation

This repository contains a PyTorch implementation of several sequential recommendation baselines with time prediction heads. It supports Transformer-based and Mamba-based time encoders and evaluates both item (POI) ranking and timestamp prediction quality.

## Features
- Five baselines: SASRec, GRU4Rec, Bert4Rec, FMLP4Rec, and GeoSAN (see `model_factory.py`).
- Two time encoders: Transformer regression head or Mamba Von Mises mixture head (set via `--time_encoder_type transformer|mamba`).
- Joint loss: item ranking (BCE) + optional time loss (Von Mises NLL or circular regression).
- Dataset integrity checks and periodic evaluation with NDCG/HR and detailed time-error summaries.

## Data Format
Place a file at `data/<dataset>.txt` with tab-separated fields per line:
```
user_id    timestamp    latitude    longitude    item_id
```
- `user_id`, `item_id` are 1-indexed.
- `timestamp` is Unix time (seconds).
- The loader keeps the last two interactions per user for validation/test; the rest go to training.
- Optional session splitting: set `--time_gap_threshold` (seconds) and `--min_session_length`.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # or install torch, numpy, tqdm, etc.
```

## Training & Evaluation
Run `main.py` with the desired baseline and dataset:
```bash
python main.py --dataset my_dataset --train_dir exp1 --baseline sasrec --device cuda:0
```
Key arguments (see `config.py` for full list):
- `--baseline`: baseline1|baseline2|...|baseline7 (sasrec, gru4rec, caser, bert4rec, tisasrec, fmlp4rec, geosan)
- `--time_encoder_type`: transformer|mamba
- `--batch_size`, `--lr`, `--num_epochs`, `--maxlen`, `--hidden_units`, `--num_blocks`, `--num_heads`
- Time options: `--vm_num_components`, `--vm_min_kappa`, `--vm_max_kappa`, `--time_span`, `--time_gap_threshold`, `--min_session_length`

During training, checkpoints and logs are saved to `<dataset>_<train_dir>/`:
- `args.txt`: snapshot of hyperparameters
- `log.txt`: periodic metrics (NDCG@10/HR@10, time diff if enabled)
- `*.pth`: model weights saved at the final epoch

To run evaluation only on a saved checkpoint:
```bash
python main.py --dataset my_dataset --train_dir exp1 --baseline sasrec --device cuda:0 --state_dict_path <path_to_pth> --inference_only false
```

## Code Structure
- `main.py`: training loop, loss composition, evaluation hooks
- `model_factory.py`: model construction and wrappers for time prediction
- `encoder.py`: POI encoder and time encoder (Transformer or Mamba)
- `mamba_adapter.py`: Mamba-based time predictor with Von Mises mixture head
- `Data_Module/`: data loading, preprocessing, and negative sampling (`DataLoader.py`, `Preprocessor.py`, `warp_sampler.py`)
- `evaluator.py`: metrics and time-error summaries
- `Baselines/`: implementations of the supported recommender models

## Notes
- Time prediction is optional; if a baseline does not support it, only POI ranking loss is used.
- GPU is recommended; set `--device cuda:0` (falls back to CPU if unavailable).
- Adjust `--time_encoder_type mamba` only if the Mamba dependency is installed.

