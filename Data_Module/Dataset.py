# data/dataset.py
import torch
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np

class TAPTDataset:
    """
    TAPT dataset wrapper that stays fully compatible with the original code.
    """
    
    def __init__(self, data_components: List):
        # Preserve original structure
        self.user_train = data_components[0]
        self.user_valid = data_components[1]
        self.user_test = data_components[2]
        self.user_train_time = data_components[3]
        self.user_valid_time = data_components[4]
        self.user_test_time = data_components[5]
        self.usernum = data_components[6]
        self.itemnum = data_components[7]
        self.timenum = data_components[8]
        self.min_year = data_components[9]
        self.num_year = data_components[10]
        self.poi_info = data_components[11]
        
        # Keep raw components for compatibility
        self._data_components = data_components
        
        # Initialize statistics
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute basic statistics"""
        self.train_sequences = sum(len(seq) for seq in self.user_train.values())
        self.valid_sequences = sum(len(seq) for seq in self.user_valid.values())
        self.test_sequences = sum(len(seq) for seq in self.user_test.values())
        
        self.avg_sequence_length = self.train_sequences / len(self.user_train) if self.user_train else 0
    
    def validate(self) -> bool:
        """
        Basic data validation
        """
        try:
            # Check essential data structures
            assert isinstance(self.user_train, dict), "user_train should be dict"
            assert isinstance(self.user_valid, dict), "user_valid should be dict"
            assert isinstance(self.user_test, dict), "user_test should be dict"
            assert isinstance(self.poi_info, dict), "poi_info should be dict"
            
            # Check key counts exist
            assert self.usernum > 0, "usernum should be positive"
            assert self.itemnum > 0, "itemnum should be positive"
            
            return True
        except AssertionError as e:
            print(f"Data validation failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Return dataset statistics"""
        return {
            'user_count': self.usernum,
            'item_count': self.itemnum,
            'train_sequences': self.train_sequences,
            'valid_sequences': self.valid_sequences,
            'test_sequences': self.test_sequences,
            'avg_sequence_length': self.avg_sequence_length
        }
    
    def to_legacy_format(self) -> List:
        """
        Return raw data format compatible with original code
        """
        return self._data_components
    
    def __iter__(self):
        """Support tuple-unpacking like the original code"""
        return iter(self._data_components)
