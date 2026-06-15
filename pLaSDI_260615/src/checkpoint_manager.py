# -*- coding: utf-8 -*-
"""
Checkpoint Manager
==================
Manages best-model checkpoints, RAM log buffering, and disk flushing.

Checkpoint/log manager separated from trainer.py.
Accumulates data in RAM and periodically flushes to disk to reduce I/O overhead.
"""

import os
from typing import Optional, Dict, Any

import torch
import numpy as np

from .train_utils import append_loss_csv, append_metrics_csv, METRICS_HEADER


class BestTracker:
    """
    Single best-model tracker (train-best or val-best).
    
    Keeps best weights in RAM and saves them to disk on flush.
    """
    
    def __init__(self, label: str, ckpt_path: str):
        """
        Args:
            label: "train" or "val"
            ckpt_path: Checkpoint save path.
        """
        self.label = label
        self.ckpt_path = ckpt_path
        
        self.best_metric = float("inf")
        self.best_epoch = 0
        self.best_weights = None  # AE state_dict
        self.sindy_weights = None  # SINDy model state_dict (for adaptive mode)
        
        self._opt_state = None
        self._sched_state = None
        self._extra = {}
        self._dirty = False  # A new best exists in RAM but has not yet been written to disk
    
    def update(self, metric: float, epoch: int,
               ae_state: dict, sindy_state: Optional[dict] = None,
               opt_state: Optional[dict] = None,
               sched_state: Optional[dict] = None,
               extra: Optional[dict] = None) -> bool:
        """
        Update if the new metric is better than the current best.
        
        Returns:
            True if updated
        """
        if metric < self.best_metric:
            self.best_metric = metric
            self.best_epoch = epoch
            self.best_weights = {k: v.cpu().clone() for k, v in ae_state.items()}
            self.sindy_weights = (
                {k: v.cpu().clone() for k, v in sindy_state.items()}
                if sindy_state else None
            )
            self._opt_state = opt_state
            self._sched_state = sched_state
            self._extra = extra or {}
            self._dirty = True
            return True
        return False
    
    def flush_to_disk(self):
        """Save the best checkpoint in RAM to disk."""
        if not self._dirty or self.best_weights is None:
            return
        
        ckpt = {
            'model_state': self.best_weights,
            'epoch': self.best_epoch,
            'best_type': self.label,
        }
        if self.sindy_weights is not None:
            ckpt['sindy_model_state'] = self.sindy_weights
        if self._opt_state is not None:
            ckpt['opt_state'] = self._opt_state
        if self._sched_state is not None:
            ckpt['sched_state'] = self._sched_state
        if self._extra:
            ckpt.update(self._extra)
        
        torch.save(ckpt, self.ckpt_path)
        self._dirty = False
        print(f"  [Disk] {self.label.capitalize()}-best checkpoint saved (epoch {self.best_epoch})")
    
    def load_weights(self, model, sindy_model=None, device=None):
        """
        Load best weights into the model, preferring RAM and falling back to disk.
        
        Returns:
            True if loaded successfully
        """
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print(f"[Model] Loaded {self.label}-best AE weights from memory")
            if sindy_model is not None and self.sindy_weights is not None:
                sindy_model.load_state_dict(self.sindy_weights)
                print(f"[Model] Loaded {self.label}-best SINDy weights from memory")
            return True
        
        if os.path.exists(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            print(f"[Model] Loaded {self.label}-best AE model from {self.ckpt_path}")
            if sindy_model is not None and 'sindy_model_state' in ckpt:
                sindy_model.load_state_dict(ckpt['sindy_model_state'])
                print(f"[Model] Loaded {self.label}-best SINDy model from {self.ckpt_path}")
            return True
        
        print(f"[Model] Warning: No {self.label}-best weights found")
        return False


class CheckpointManager:
    """
    Integrated checkpoint and log manager.
    
    - Tracks train-best and val-best independently.
    - Buffers loss/metrics CSV rows in RAM and periodically flushes to disk.
    """
    
    def __init__(self, cfg):
        """
        Args:
            cfg: LaSDIcConfig
        """
        self.cfg = cfg
        self.flush_every = cfg.save.disk_save_every
        
        # Best trackers
        self.train_best = BestTracker("train", cfg.ckpt_train_best_path)
        self.val_best = BestTracker("val", cfg.ckpt_val_best_path)
        
        # RAM log buffers
        self._loss_buf = []       # [(epoch, loss), ...]
        self._val_loss_buf = []   # [(epoch, loss), ...]
        self._metrics_buf = []    # [row_dict, ...]
    
    def buffer_loss(self, epoch: int, loss: float):
        """Add train loss to the RAM buffer."""
        self._loss_buf.append((epoch, loss))
    
    def buffer_val_loss(self, epoch: int, loss: float):
        """Add validation loss to the RAM buffer."""
        self._val_loss_buf.append((epoch, loss))
    
    def buffer_metrics(self, row: dict):
        """Add a metrics row to the RAM buffer."""
        self._metrics_buf.append(row)
    
    def should_flush(self, epoch: int) -> bool:
        """Return whether this epoch should trigger a disk flush."""
        return epoch % self.flush_every == 0
    
    def flush_all(self):
        """Flush all RAM buffers to disk."""
        cfg = self.cfg
        
        # Best checkpoints
        self.train_best.flush_to_disk()
        self.val_best.flush_to_disk()
        
        # Loss CSV
        if self._loss_buf:
            for ep, lv in self._loss_buf:
                append_loss_csv(cfg.losscsv_path, ep, lv)
            self._loss_buf.clear()
        
        if self._val_loss_buf:
            for ep, lv in self._val_loss_buf:
                append_loss_csv(cfg.vallosscsv_path, ep, lv)
            self._val_loss_buf.clear()
        
        # Metrics CSV
        if self._metrics_buf:
            for row in self._metrics_buf:
                append_metrics_csv(cfg.metrics_csv_path, row, METRICS_HEADER)
            self._metrics_buf.clear()
    
    def flush_if_needed(self, epoch: int):
        """Conditionally flush based on epoch."""
        if self.should_flush(epoch):
            self.flush_all()
