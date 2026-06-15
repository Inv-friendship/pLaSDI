#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LaSDIc Training Script
======================
Main training entry-point script

Run cells with Shift+Enter in VSCode.
"""

#%%
# =============================================================================
# CONFIG - edit only this section
# =============================================================================

# Resume setting
RESUME = False  # True/False toggle

#%%
# =============================================================================
# Setup & Imports
# =============================================================================

import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import LaSDIcConfig, create_default_config
from src.trainer import LaSDIcTrainer
from src.train_utils import TeeLogger

print("✅ Imports complete")

#%%
# =============================================================================
# Load Configuration
# =============================================================================

cfg = create_default_config()

cfg.train.add_training = RESUME

# Rebuild paths
cfg.__post_init__()

print(f"✅ Config loaded")
print(f"   Output: {cfg.out_dir}")
print(f"   Device: {cfg.get_device()}")

#%%
# =============================================================================
# Setup Logging & Trainer
# =============================================================================

# Set up the log file
log_file = cfg.out_dir / "training.log"
log_mode = "a" if RESUME else "w"

# Start TeeLogger
tee = TeeLogger(str(log_file), mode=log_mode)
sys.stdout = tee
sys.stderr = tee

print(f"\n{'='*60}")
print(f" Logging to: {log_file}")
print(f"{'='*60}\n")

# Print config
cfg.print_summary()

print(f"\n{'='*60}")
print(f" LaSDIc Training")
print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")

# Create trainer
trainer = LaSDIcTrainer(cfg)
trainer.setup_data()
trainer.setup_model()

print("✅ Trainer ready")

#%%
# =============================================================================
# Train
# =============================================================================

try:
    trainer.train(resume=cfg.train.add_training)
    
    # Load the best model
    trainer.load_best_model()
    
    print(f"\n{'='*60}")
    print(f" Training Complete!")
    print(f" Train-best saved to: {cfg.ckpt_train_best_path}")
    print(f" Val-best saved to:   {cfg.ckpt_val_best_path}")
    print(f"{'='*60}\n")

except Exception as e:
    print(f"\n{'='*60}")
    print(f" ERROR: Training failed")
    print(f" {e}")
    print(f"{'='*60}\n")
    raise

finally:
    # Close the log file and restore stdout
    sys.stdout = tee.out_terminal
    sys.stderr = tee.err_terminal
    tee.close()
    print(f"Log saved to: {log_file}")



# %%
