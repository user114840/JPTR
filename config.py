# config.py
import argparse
import os
from typing import Any, Dict

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

class Config:
    def __init__(self):
        # Keep original project defaults
        self.baseline: str = 'baseline1'
        self.dataset: str = 'ratings_out'
        self.train_dir: str = 'default'
        self.batch_size: int = 128
        self.lr: float = 0.001
        self.maxlen: int = 200
        self.hidden_units: int = 128
        self.exp_factor: int = 4
        self.time_hidden_units1: int = 30
        self.time_hidden_units2: int = 30
        self.num_blocks: int = 3
        self.kernel_size: int = 3
        self.num_epochs: int = 601
        self.num_heads: int = 2
        self.dropout_rate: float = 0.2
        self.l2_emb: float = 1e-6
        self.device: str = 'cpu'
        self.inference_only: bool = False
        self.state_dict_path: str = None
        self.nv: int = 4
        self.nh: int = 16
        self.ac_conv: str = 'relu'
        self.ac_fc: str = 'relu'
        self.time_encoder_type: str = 'transformer'  # transformer, mamba, identity
        self.time_encoder_layers: int = 2
        self.time_encoder_heads: int = 4
        self.time_encoder_dropout: float = 0.1
        self.time_freq_min = 0.001  # min frequency
        self.time_freq_max = 0.1    # max frequency 
        self.time_freq_units = 16   # number of frequencies
        self.include_dow = True     # include day of week
        self.include_hour = True    # include hour
        self.vm_num_components: int = 3
        self.vm_min_kappa: float = 0.1
        self.vm_max_kappa: float = 20.0
        self.time_gap_threshold: int = 0  # seconds; 0 means no split
        self.min_session_length: int = 0  # minimum session length after split
        self.poi_conv_kernel: int = 3
        self.mamba_use_cuda: bool = True
        self.time_span: int = 256
        self._args = None

    def load_from_command_line(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--baseline', default=self.baseline, type=str)
        parser.add_argument('--dataset', default=self.dataset, required=True)
        parser.add_argument('--train_dir', default=self.train_dir, required=True)
        parser.add_argument('--batch_size', default=self.batch_size, type=int)
        parser.add_argument('--lr', default=self.lr, type=float)
        parser.add_argument('--maxlen', default=self.maxlen, type=int)
        parser.add_argument('--hidden_units', default=self.hidden_units, type=int)
        parser.add_argument('--exp_factor', default=self.exp_factor, type=int)
        parser.add_argument('--time_hidden_units1', default=self.time_hidden_units1, type=int)
        parser.add_argument('--time_hidden_units2', default=self.time_hidden_units2, type=int)
        parser.add_argument('--num_blocks', default=self.num_blocks, type=int)
        parser.add_argument('--kernel_size', default=self.kernel_size, type=int)
        parser.add_argument('--num_epochs', default=self.num_epochs, type=int)
        parser.add_argument('--num_heads', default=self.num_heads, type=int)
        parser.add_argument('--dropout_rate', default=self.dropout_rate, type=float)
        parser.add_argument('--l2_emb', default=self.l2_emb, type=float)
        parser.add_argument('--device', default=self.device, type=str)
        parser.add_argument('--inference_only', default=self.inference_only, type=str2bool)
        parser.add_argument('--state_dict_path', default=self.state_dict_path, type=str)
        parser.add_argument('--nv', default=self.nv, type=int)
        parser.add_argument('--nh', default=self.nh, type=int)
        parser.add_argument('--ac_conv', default=self.ac_conv, type=str)
        parser.add_argument('--ac_fc', default=self.ac_fc, type=str)
        parser.add_argument('--time_encoder_type', default=self.time_encoder_type, type=str)
        parser.add_argument('--time_encoder_layers', default=self.time_encoder_layers, type=int)
        parser.add_argument('--time_encoder_heads', default=self.time_encoder_heads, type=int)
        parser.add_argument('--time_encoder_dropout', default=self.time_encoder_dropout, type=float)
        parser.add_argument('--vm_num_components', default=self.vm_num_components, type=int)
        parser.add_argument('--vm_min_kappa', default=self.vm_min_kappa, type=float)
        parser.add_argument('--vm_max_kappa', default=self.vm_max_kappa, type=float)
        parser.add_argument('--time_gap_threshold', default=self.time_gap_threshold, type=int)
        parser.add_argument('--min_session_length', default=self.min_session_length, type=int)
        parser.add_argument('--poi_conv_kernel', default=self.poi_conv_kernel, type=int)
        parser.add_argument('--mamba_use_cuda', default=self.mamba_use_cuda, type=str2bool)
        parser.add_argument('--time_span', default=self.time_span, type=int)

        self._args = parser.parse_args()
        self._update_from_args()

    def _update_from_args(self):
        for key, value in vars(self._args).items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Export parameters as a dict for saving to file"""
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}

    def save_to_file(self, file_path: str):
        """Persist parameters to file, matching original project behavior"""
        with open(file_path, 'w') as f:
            for k, v in self.to_dict().items():
                f.write(f"{k},{v}\n")
