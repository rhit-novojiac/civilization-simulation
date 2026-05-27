import os
import torch
import shutil
from unittest.mock import patch, MagicMock

from ml.training.trainer import DQNTrainer
from ml.models.macro_net import MacroDQN
from ml.models.micro_net import MicroDQN

def test_save_models():
    # Setup mock brains dictionary
    brains = {
        '1': {
            'macro': MacroDQN(),
            'micro': MicroDQN()
        },
        '2': {
            'macro': MacroDQN()
        }
    }
    
    test_dir = "test_models_dir"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    DQNTrainer.save_models(brains, filepath=test_dir)
    
    assert os.path.exists(test_dir)
    assert os.path.exists(os.path.join(test_dir, "species_1_macro.pt"))
    assert os.path.exists(os.path.join(test_dir, "species_1_micro.pt"))
    assert os.path.exists(os.path.join(test_dir, "species_2_macro.pt"))
    assert not os.path.exists(os.path.join(test_dir, "species_2_micro.pt"))
    
    # Cleanup
    shutil.rmtree(test_dir)
