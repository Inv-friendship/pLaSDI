# -*- coding: utf-8 -*-
"""
LaSDIc Trainer Module
=====================
Main training trainer - integrated AE + SINDyC + physics-loss training

Uses mini-batches (full-batch mode removed)
SINDy is always enabled (toggle removed)
Pretraining removed (train only)

v2.1: added detailed logging and timing profiling
"""

import os
import random
import time
import contextlib
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LambdaLR

from .scaling import PopulationScaler, ControlScaler, TorchScaleHelper
from .atomic_physics import AtomicPhysics, PhysicsLoss
from .sindy_utils import (
    SINDyLossCalculator, split_coefs_torch, zstar_exact_torch,
    eig_extrema, hurwitz_gate_check, hurwitz_penalty_symmetric,
    save_sindy_coefs
)
from .data_utils import (
    load_or_build_pops, build_segment_slices, 
    split_train_val_random_segments, iter_case_minibatches,
    build_local_indices_for_batch, load_state_names,
    load_history_file, guess_history_path, align_controls,
    SteadyStateData
)
from .train_utils import (
    CudaTimer, EpochTimer, TeeLogger,
    get_lr, set_lr, ramp_weight, in_frozen_span,
    append_loss_csv, append_metrics_csv, 
    save_checkpoint, load_checkpoint, METRICS_HEADER,
)
from .checkpoint_manager import CheckpointManager

# Type hint only (config.py is outside src/)
if TYPE_CHECKING:
    from config import LaSDIcConfig


def _nullctx():
    """no-op context manager (alternative: contextlib.nullcontext, Python 3.7+)"""
    return contextlib.nullcontext()


class LaSDIcTrainer:
    """
    Integrated LaSDIc trainer
    
    Autoencoder + SINDyC + physics-loss training
    """
    
    def __init__(self, cfg: "LaSDIcConfig"):
        """
        Args:
            cfg: LaSDIcConfig configuration object
        """
        self.cfg = cfg
        self.device = cfg.get_device()
        self.dtype = cfg.dtype
        
        torch.set_default_dtype(cfg.dtype)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        # Set seed
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        
        # Model (initialized later)
        self.ae = None
        self.ld = None  # SINDyC latent dynamics
        
        # Data
        self.X_frames = None
        self.U_all = None
        self.pop_scaler = None
        self.ctrl_scaler = None
        self.scale_helper = None
        
        # Segments/indices
        self.segment_slices = None
        self.train_idx = None
        self.val_idx = None
        self.train_slices = None
        self.val_slices = None
        
        # Atomic physics
        self.atomic_physics = None
        self.physics_loss = None
        
        # Steady-state
        self.steady_data = None
        
        # CheckpointManager (centralized best-model and log-buffer management)
        # Initialize after setup finalizes cfg
        self.ckpt_mgr = None
        
        # Legacy compatibility (for external code accessing trainer.train_best_metric, etc.)
        self.train_best_weights = None
        self.train_best_metric = float("inf")
        self.train_best_epoch = 0
        self.val_best_weights = None
        self.val_best_metric = float("inf")
        self.val_best_epoch = 0
        self.best_weights = None
        self.best_metric = float("inf")
        self.best_epoch = 0
        
        self.start_epoch_offset = 0
        self.train_loss_curve = []
        
        # Precomputed cache (for periodic saves)
        self._cached_coef_vec = None       # Latest lstsq coef_vec (for the current model)
        self._cached_coef_epoch = -1       # Epoch at which the cache was captured
        self._precomputed_dirty = False    # A new cache exists but has not yet been written to disk
        
        print(f"[OK] Trainer initialized on {self.device}")
    
    def setup_data(self):
        """Load and preprocess data"""
        cfg = self.cfg
        
        # Load populations
        print("[Data] Loading population data...")
        pops = load_or_build_pops(cfg.data_files, cfg.data.nx, cfg.data.data_dir)
        pops = [np.asarray(p, dtype=np.float64) for p in pops]
        pop = np.concatenate(pops, axis=0)
        
        # Apply pop_lim and normalize
        tmp = np.sum(pop, axis=1, keepdims=True)
        pop = pop / tmp + cfg.data.pop_lim
        pop = pop * tmp
        
        # Population scaling
        self.pop_scaler = PopulationScaler(eps=cfg.data.pop_lim, normalize=True)
        W = self.pop_scaler.fit_transform(pop, axis=1)
        W = np.expand_dims(W, axis=1)  # (nt_total, 1, nx)
        
        # AE input shape: (nt_total, 1, 1, nx)
        self.X_frames = torch.tensor(W[:, None, :, :], dtype=self.dtype)
        
        self.nt_total = pop.shape[0]
        self.time_axis = np.arange(self.nt_total) * cfg.data.dt
        
        # Load history (control)
        print("[Data] Loading control variables...")
        U_segments = []
        mu_global = None
        
        for i, f in enumerate(cfg.data_files):
            hpath = cfg.history_files[i] if i < len(cfg.history_files) else guess_history_path(f)
            if hpath is None or not os.path.exists(hpath):
                raise FileNotFoundError(f"[history] Could not find the history file corresponding to '{f}'.")
            
            t_h, U_h = load_history_file(hpath)
            L = pops[i].shape[0]
            t_seg = np.arange(L) * cfg.data.dt
            
            if len(t_h) != L:
                U_h = align_controls(t_h, U_h, t_seg)
            
            mu_i = U_h.shape[1]
            if mu_global is None:
                mu_global = mu_i
            elif mu_i != mu_global:
                raise ValueError(f"[history] Input dimension mismatch: file {i} has mu={mu_i}")
            
            U_segments.append(U_h.astype(np.float64))
        
        U_all_raw = np.concatenate(U_segments, axis=0)
        
        # Control scaling
        self.ctrl_scaler = ControlScaler(eps=1e-300)
        self.U_all = self.ctrl_scaler.fit_transform(U_all_raw)
        self.U_all_raw = U_all_raw
        self.mu = mu_global
        
        print(f"[Data] U_all.shape = {self.U_all.shape}, mu = {mu_global}")
        
        # Scale helper
        self.scale_helper = TorchScaleHelper(self.pop_scaler, self.dtype)
        
        # Segment slices
        self.segment_slices = build_segment_slices(pops)
        self.pops = pops
        self.pop = pop
        
        # Train/Val ë¶"í• 
        (self.train_idx, self.val_idx, 
         self.train_slices, self.val_slices,
         self.train_seg_ids, self.val_seg_ids) = split_train_val_random_segments(
            self.segment_slices, 
            cfg.train.val_ratio,
            seed=cfg.seed
        )
        
        self.n_time_train = len(self.train_idx)
        print(f"[Data] Train: {self.n_time_train}, Val: {len(self.val_idx)}")
        
        # State names
        state_names = load_state_names(cfg.data.names_file, cfg.data.nx)
        
        # Atomic physics
        self.atomic_physics = AtomicPhysics(state_names, cfg.data.nx, self.dtype)
        # PhysicsLoss is initialized in setup_model() (requires dt_eff and fd_type)
        self.physics_loss = None
        
        # Steady-state data
        if cfg.data.steady_enable:
            self.steady_data = SteadyStateData(
                cfg.steady_pop_hist_pairs,
                self.pop_scaler,
                self.ctrl_scaler,
                random_pick=cfg.data.steady_random_pick,
                num_samples=cfg.data.steady_num_samples,
                seed=cfg.data.steady_random_seed,
                pop_lim=cfg.data.pop_lim
            )
        
        print("[Data] Setup complete!")
        
        # Initial dirty flag for precomputed data (save scaler on first flush)
        self._precomputed_dirty = True
    
    def setup_model(self):
        """Initialize model"""
        cfg = self.cfg
        
        # Autoencoder
        from .autoencoder import Autoencoder
        
        self.ae = Autoencoder(
            nx=cfg.data.nx,
            latent_dim=cfg.model.latent_dim,
            hidden_units=cfg.model.hidden,
            activation=cfg.model.activation
        ).to(self.device, dtype=self.dtype)
        
        # dt_eff
        dt_eff = cfg.sindy.dt_eff if cfg.sindy.dt_eff else cfg.data.dt
        self.dt_eff = dt_eff
        
        # Select SINDy model
        self.use_adaptive_sindy = cfg.sindy.use_adaptive
        
        if self.use_adaptive_sindy:
            # Adaptive SINDyC (CoefNet-based)
            from .sindyc_adaptive import AdaptiveSINDyC
            
            self.sindy_model = AdaptiveSINDyC(
                nz=cfg.model.latent_dim,
                mu=2,  # T, density
                hidden_dims=cfg.sindy.adaptive_hidden,
                activation=cfg.sindy.adaptive_activation,
                fd_type=cfg.sindy.fd_type,
                eps=cfg.sindy.adaptive_eps,
                symmetric=cfg.sindy.adaptive_symmetric,
                head_gain=cfg.sindy.adaptive_head_gain
            ).to(self.device, dtype=self.dtype)
            
            self.ld = None
            self.sindy_calc = None
            
            sindy_params = sum(p.numel() for p in self.sindy_model.parameters())
            sym_str = "symmetric" if cfg.sindy.adaptive_symmetric else "asymmetric"
            print(f"[Model] AdaptiveSINDyC: hidden={cfg.sindy.adaptive_hidden}, eps={cfg.sindy.adaptive_eps}, A={sym_str}")
            print(f"[Model] AdaptiveSINDyC params: {sindy_params:,}")
        else:
            # Existing lstsq method
            from .sindyc import SINDyC
            
            self.ld = SINDyC(
                dim=cfg.model.latent_dim,
                nt=self.n_time_train,
                fd_type=cfg.sindy.fd_type,
                use_global_coefs=cfg.sindy.use_global_coefs
            )
            
            self.sindy_calc = SINDyLossCalculator(
                self.ld, cfg.model.latent_dim, dt_eff,
                self.device, self.dtype, cfg.sindy.use_cpu
            )
            
            self.sindy_model = None
            print(f"[Model] SINDyC (lstsq): fd_type={cfg.sindy.fd_type}")
        
        # PhysicsLoss
        self.physics_loss = PhysicsLoss(
            self.atomic_physics, 
            tau=cfg.train.tau,
            fd_type=cfg.sindy.fd_type,
            dt_eff=dt_eff
        )
        print(f"[Model] PhysicsLoss: fd_type={cfg.sindy.fd_type}, dt_eff={dt_eff}")
        
        total_params = sum(p.numel() for p in self.ae.parameters())
        trainable_params = sum(p.numel() for p in self.ae.parameters() if p.requires_grad)
        print(f"[Model] AE params: total={total_params:,} | trainable={trainable_params:,}")
        
        # Initialize CheckpointManager
        self.ckpt_mgr = CheckpointManager(cfg)
        print(f"[Model] CheckpointManager initialized (flush every {cfg.save.disk_save_every} epochs)")
    
    def train(self, resume: bool = True):
        """
        Main training loop
        
        Args:
            resume: whether to resume training
        """
        cfg = self.cfg
        
        # ==================== Additional checkpoint handling ====================
        # Warn when checkpoints exist but resume=False
        if not resume and os.path.exists(cfg.ckpt_train_best_path):
            print(f"\n{'[!] '*20}")
            print(f"WARNING: Checkpoint exists at {cfg.ckpt_train_best_path}")
            print(f"But resume=False! Starting fresh will OVERWRITE the existing checkpoint.")
            print(f"{'[!] '*20}\n")
            
            # User confirmation (optional - commenting this out leaves only the warning)
            import sys
            if hasattr(sys, 'ps1') or 'ipykernel' not in sys.modules:
                # Prompt only in terminal environments (skip in Jupyter)
                try:
                    response = input("Continue and overwrite? (y/N): ")
                    if response.lower() != 'y':
                        print("Training aborted.")
                        return
                except (EOFError, KeyboardInterrupt):
                    print("\nTraining aborted.")
                    return
        # =========================================================
        
        # Optimizer (also includes sindy_model for adaptive mode)
        if self.use_adaptive_sindy:
            opt = torch.optim.Adam([
                {'params': self.ae.parameters(), 'lr': cfg.train.lr},
                {'params': self.sindy_model.parameters(), 'lr': cfg.train.lr},
            ])
        else:
            opt = torch.optim.Adam(self.ae.parameters(), lr=cfg.train.lr)
        # LR scheduler
        sched_type = cfg.train.lr_scheduler.lower()
        if sched_type == "cosine":
            if cfg.train.warmup_epochs > 0:
                warmup_ep = cfg.train.warmup_epochs
                main_ep = max(1, cfg.train.epochs - warmup_ep)
                cosine_main = CosineAnnealingLR(opt, T_max=main_ep, eta_min=cfg.train.min_lr)
                
                def _warmup_lambda(ep):
                    if ep < warmup_ep:
                        alpha = cfg.train.warmup_start_lr / cfg.train.lr
                        return alpha + (1.0 - alpha) * (ep / warmup_ep)
                    return 1.0
                
                warmup_sched = LambdaLR(opt, lr_lambda=_warmup_lambda)
                # Use warmup_sched during warmup, then cosine_main afterward
                scheduler = {"type": "warmup_cosine", "warmup": warmup_sched,
                             "main": cosine_main, "warmup_epochs": warmup_ep}
                print(f"[Scheduler] CosineAnnealing + Warmup({warmup_ep}ep), eta_min={cfg.train.min_lr:.1e}")
            else:
                scheduler = CosineAnnealingLR(opt, T_max=cfg.train.epochs, eta_min=cfg.train.min_lr)
                print(f"[Scheduler] CosineAnnealing, T_max={cfg.train.epochs}, eta_min={cfg.train.min_lr:.1e}")
        elif sched_type == "one_cycle":
            scheduler = OneCycleLR(
                opt, max_lr=cfg.train.max_lr,
                total_steps=cfg.train.epochs,
                pct_start=0.1,
                anneal_strategy='cos',
                final_div_factor=cfg.train.lr / cfg.train.min_lr if cfg.train.min_lr > 0 else 1e4,
            )
            print(f"[Scheduler] OneCycleLR, max_lr={cfg.train.max_lr:.1e}, total={cfg.train.epochs}")
        elif sched_type == "none":
            scheduler = None
            print(f"[Scheduler] None (constant lr={cfg.train.lr:.1e})")
        else:
            raise ValueError(f"Unknown lr_scheduler: {sched_type}. Choose: 'cosine', 'one_cycle', 'none'")
        
        # Load checkpoint (resume from train_best by default)
        if resume and cfg.train.add_training and os.path.exists(cfg.ckpt_train_best_path):
            print(f"[Train] Loading checkpoint: {cfg.ckpt_train_best_path}")
            sched_for_load = scheduler if not isinstance(scheduler, dict) else None
            ckpt = load_checkpoint(cfg.ckpt_train_best_path, self.ae, opt if cfg.train.resume_optimizer else None, 
                                    sched_for_load, self.device)
            self.start_epoch_offset = ckpt.get('epoch', 0)
            if os.path.exists(cfg.losslog_path):
                prev_curve = np.load(cfg.losslog_path).astype(float)
                self.train_loss_curve = prev_curve.tolist()
        
        # Force learning rate
        set_lr(opt, cfg.train.lr)
        current_lr = cfg.train.lr
        
        # Initialize training state (separate train/val bests)
        self.train_best_metric = float("inf")
        self.train_best_epoch = 0
        self.val_best_metric = float("inf")
        self.val_best_epoch = 0
        # Legacy compat
        self.best_metric = float("inf")
        self.best_epoch = 0
        timer = EpochTimer(smooth=100)
        
        # Steady-state tensors
        if self.steady_data and self.steady_data.enabled:
            steady_W_t, steady_U_t = self.steady_data.to_torch(self.device, self.dtype)
        else:
            steady_W_t, steady_U_t = None, None
        
        # Loss functions
        loss_fn = torch.nn.MSELoss()
        
        print(f"\n{'='*60}")
        print(f" Training Start: {cfg.train.epochs} epochs")
        print(f"{'='*60}\n")
        
        # Print training status and back up config
        cfg.print_training_status()
        cfg.backup_config()
        
        train_start_wall = time.time()
        
        for ep in range(1, cfg.train.epochs + 1):
            ep_global = self.start_epoch_offset + ep
            
            # Shuffle mini-batches
            random.seed(ep)
            train_slices_shuffled = self.train_slices.copy()
            random.shuffle(train_slices_shuffled)
            
            # ==================== Timer initialization ====================
            data_timer = CudaTimer()      # Data loading
            ae_timer = CudaTimer()        # AE forward
            phys_timer = CudaTimer()      # Physics-loss calculation
            rate_timer = CudaTimer()      # Rate equation loss
            sindy_timer = CudaTimer()     # SINDy loss
            hurwitz_timer = CudaTimer()   # Hurwitz penalty
            steady_timer = CudaTimer()    # Steady-state loss
            back_timer = CudaTimer()      # Backward
            opt_timer = CudaTimer()       # Optimizer step
            val_timer = CudaTimer()       # Validation
            log_timer = CudaTimer()       # Logging
            
            self.ae.train()
            timer.start()
            
            # Compute ramp-up weights (unified dictionary loop)
            ramp = cfg.train.ramp_config
            lm = cfg.train.loss_mode
            
            # Mapping from loss name to base weight
            _base_weights = {
                "fse": cfg.train.w_fse, "frac": cfg.train.w_frac,
                "ion": cfg.train.w_ion, "zbar": cfg.train.w_zbar,
                "sindy": cfg.sindy.weight, "coef": cfg.sindy.coef_weight,
                "hurwitz": cfg.hurwitz.weight, "steady": cfg.train.steady_weight,
                "rate_W": cfg.train.w_rate_W, "rate_N": cfg.train.w_rate_N,
                "rate_CSD": cfg.train.w_rate_CSD, "rate_Zbar": cfg.train.w_rate_Zbar,
            }
            
            w_dyn = {}
            for name, base_w in _base_weights.items():
                r = ramp[name]
                w_dyn[name] = ramp_weight(ep, r["T"], base_w, r["mode"])
                # Losses in monitor/off mode get zero weight and are excluded from total
                if lm.get(name, "off") != "train":
                    w_dyn[name] = 0.0
            
            # Convenience aliases (compatible with existing code)
            w_fse_dyn = w_dyn["fse"]
            w_frac_dyn = w_dyn["frac"]
            w_ion_dyn = w_dyn["ion"]
            w_zbar_dyn = w_dyn["zbar"]
            w_sindy_dyn = w_dyn["sindy"]
            w_coef_dyn = w_dyn["coef"]
            w_hurwitz_dyn = w_dyn["hurwitz"]
            w_steady_dyn = w_dyn["steady"]
            w_rate_W_dyn = w_dyn["rate_W"]
            w_rate_N_dyn = w_dyn["rate_N"]
            w_rate_CSD_dyn = w_dyn["rate_CSD"]
            w_rate_Zbar_dyn = w_dyn["rate_Zbar"]
            
            # Epoch accumulation variables
            ep_recW, ep_frac, ep_ion, ep_zbar, ep_fse = 0.0, 0.0, 0.0, 0.0, 0.0
            ep_sindy, ep_coef, ep_hurwitz, ep_steady = 0.0, 0.0, 0.0, 0.0
            ep_rate_W, ep_rate_N = 0.0, 0.0
            ep_rate_CSD, ep_rate_Zbar = 0.0, 0.0
            n_samps = 0
            
            sindy_sum, coef_sum, hurwitz_sum, steady_sum = 0.0, 0.0, 0.0, 0.0
            n_batches = 0
            rate_W_sum, rate_N_sum = 0.0, 0.0
            rate_CSD_sum, rate_Zbar_sum = 0.0, 0.0
            
            # Track individual steady-state losses
            ep_steady_fse, ep_steady_frac, ep_steady_ion, ep_steady_zbar = 0.0, 0.0, 0.0, 0.0
            
            # Hurwitz gate variables
            max_real_ep, min_real_ep = -float("inf"), float("inf")
            hurwitz_ok_ep, strong_ok_ep = True, True
            
            # Track condition number (for lstsq steady-state)
            ep_condA_max = 0.0
            ep_condA_min = float("inf")
            ep_steady_skipped = 0   # Number of batches skipped by cond_skip
            
            # ==================== Mini-batch Loop ====================
            for batch_id, slices_batch in enumerate(
                iter_case_minibatches(train_slices_shuffled, cfg.train.batch_size)
            ):
                opt.zero_grad(set_to_none=True)
                
                # Data loading
                data_timer.start()
                idx_batch, local_slices = build_local_indices_for_batch(slices_batch)
                xb = self.X_frames[idx_batch].to(device=self.device, dtype=self.dtype)
                Ub = torch.as_tensor(self.U_all[idx_batch], dtype=self.dtype, device=self.device)
                data_timer.stop()
                
                # AE forward
                ae_timer.start()
                z = self.ae.encoder(xb)
                z_clean = z
                z = z + torch.randn_like(z) * cfg.train.noise
                xr = self.ae.decoder(z)
                ae_timer.stop()
                
                # Physics loss
                phys_timer.start()
                L_recW = loss_fn(xr, xb)
                
                # Fraction conversion (shared by multiple losses; needed if any of frac/ion/zbar/rate is not off)
                need_fraction = any(lm.get(k, "off") != "off" for k in ["frac", "ion", "zbar", "rate_W", "rate_N", "rate_CSD", "rate_Zbar", "steady"])
                
                if need_fraction:
                    F_truth = self.scale_helper.W_to_fraction(xb)
                    F_pred = self.scale_helper.W_to_fraction(xr)
                    F_truth_b = F_truth.view(F_truth.size(0), -1)
                    F_pred_b = F_pred.view(F_pred.size(0), -1)
                else:
                    F_truth_b = F_pred_b = None
                
                # FSE
                need_fse = lm.get("fse", "off") != "off" or lm.get("steady", "off") != "off"
                if need_fse:
                    S_truth = self.scale_helper.get_sum_before_normalize(xb)
                    S_pred = self.scale_helper.get_sum_before_normalize(xr)
                    _ctx_fse = torch.no_grad() if lm.get("fse", "off") == "monitor" else _nullctx()
                    with _ctx_fse:
                        L_fse = loss_fn(S_pred, S_truth)
                else:
                    L_fse = torch.zeros((), dtype=self.dtype, device=self.device)
                
                # Frac
                if lm.get("frac", "off") != "off":
                    _ctx_frac = torch.no_grad() if lm["frac"] == "monitor" else _nullctx()
                    with _ctx_frac:
                        L_frac = self.physics_loss.compute_fraction_loss(F_pred_b, F_truth_b)
                else:
                    L_frac = torch.zeros((), dtype=self.dtype, device=self.device)
                
                # Ion (CSD)
                if lm.get("ion", "off") != "off":
                    _ctx_ion = torch.no_grad() if lm["ion"] == "monitor" else _nullctx()
                    with _ctx_ion:
                        L_ion = self.physics_loss.compute_ion_loss(F_pred_b, F_truth_b)
                else:
                    L_ion = torch.zeros((), dtype=self.dtype, device=self.device)
                
                # Zbar
                if lm.get("zbar", "off") != "off":
                    _ctx_zbar = torch.no_grad() if lm["zbar"] == "monitor" else _nullctx()
                    with _ctx_zbar:
                        L_zbar = self.physics_loss.compute_zbar_loss(F_pred_b, F_truth_b)
                else:
                    L_zbar = torch.zeros((), dtype=self.dtype, device=self.device)
                phys_timer.stop()
                
                # Rate-equation loss (computed per segment) - compute/skip according to loss_mode
                rate_timer.start()
                L_rate_W = torch.zeros((), dtype=self.dtype, device=self.device)
                L_rate_N = torch.zeros((), dtype=self.dtype, device=self.device)
                L_rate_CSD = torch.zeros((), dtype=self.dtype, device=self.device)
                L_rate_Zbar = torch.zeros((), dtype=self.dtype, device=self.device)
                
                any_rate_on = any(lm.get(k, "off") != "off" for k in ["rate_W", "rate_N", "rate_CSD", "rate_Zbar"])
                
                if any_rate_on:
                    # Convert W and Fraction to 2D
                    W_truth_2d = xb[:, 0, 0, :]  # (B, nx)
                    W_pred_2d = xr[:, 0, 0, :] if xr.dim() == 4 else xr.view(xr.size(0), -1)
                    
                    # Get nA (total population)
                    nA_batch = torch.as_tensor(
                        np.sum(self.pop[idx_batch], axis=1, keepdims=True),
                        dtype=self.dtype, device=self.device
                    )
                    
                    # Compute rate-equation loss per segment
                    for sl in local_slices:
                        if sl.stop - sl.start < 3:
                            continue  # Central differences require at least 3 points
                        
                        W_t_seg = W_truth_2d[sl.start:sl.stop]
                        W_p_seg = W_pred_2d[sl.start:sl.stop]
                        nA_seg = nA_batch[sl.start:sl.stop]
                        
                        # Fraction slice (used by rate_N, rate_CSD, and rate_Zbar)
                        need_F_seg = any(lm.get(k, "off") != "off" for k in ["rate_N", "rate_CSD", "rate_Zbar"])
                        if need_F_seg:
                            F_t_seg = F_truth_b[sl.start:sl.stop]
                            F_p_seg = F_pred_b[sl.start:sl.stop]
                        
                        if lm.get("rate_W", "off") != "off":
                            _ctx = torch.no_grad() if lm["rate_W"] == "monitor" else _nullctx()
                            with _ctx:
                                L_rate_W = L_rate_W + self.physics_loss.compute_rate_equation_loss_W(
                                    W_p_seg, W_t_seg, cfg.data.dt
                                )
                        
                        if lm.get("rate_N", "off") != "off":
                            _ctx = torch.no_grad() if lm["rate_N"] == "monitor" else _nullctx()
                            with _ctx:
                                L_rate_N = L_rate_N + self.physics_loss.compute_rate_equation_loss_N(
                                    F_p_seg, F_t_seg, nA_seg, cfg.data.dt
                                )
                        
                        if lm.get("rate_CSD", "off") != "off":
                            _ctx = torch.no_grad() if lm["rate_CSD"] == "monitor" else _nullctx()
                            with _ctx:
                                L_rate_CSD = L_rate_CSD + self.physics_loss.compute_rate_equation_loss_CSD(
                                    F_p_seg, F_t_seg, cfg.data.dt
                                )
                        
                        if lm.get("rate_Zbar", "off") != "off":
                            _ctx = torch.no_grad() if lm["rate_Zbar"] == "monitor" else _nullctx()
                            with _ctx:
                                L_rate_Zbar = L_rate_Zbar + self.physics_loss.compute_rate_equation_loss_Zbar(
                                    F_p_seg, F_t_seg, cfg.data.dt
                                )
                
                rate_W_sum += float(L_rate_W.item())
                rate_N_sum += float(L_rate_N.item())
                rate_CSD_sum += float(L_rate_CSD.item())
                rate_Zbar_sum += float(L_rate_Zbar.item())
                rate_timer.stop()
                
                # SINDy loss
                sindy_timer.start()
                sindy_mode = lm.get("sindy", "off")
                coef_mode = lm.get("coef", "off")
                
                if sindy_mode != "off" or coef_mode != "off":
                    if self.use_adaptive_sindy:
                        L_sindy = torch.zeros((), dtype=self.dtype, device=self.device)
                        L_coef = torch.zeros((), dtype=self.dtype, device=self.device)
                        
                        for sl in local_slices:
                            Z_seg = z_clean[sl.start:sl.stop]
                            U_seg = Ub[sl.start:sl.stop]
                            if sindy_mode != "off":
                                _ctx = torch.no_grad() if sindy_mode == "monitor" else _nullctx()
                                with _ctx:
                                    L_sindy = L_sindy + self.sindy_model.compute_sindy_loss(Z_seg, U_seg, self.dt_eff)
                            if coef_mode != "off":
                                _ctx = torch.no_grad() if coef_mode == "monitor" else _nullctx()
                                with _ctx:
                                    L_coef = L_coef + self.sindy_model.compute_coef_l2_loss(U_seg)
                        
                        sindy_sum += float(L_sindy.item())
                        coef_sum += float(L_coef.item())
                    else:
                        require_grad = (sindy_mode == "train" or coef_mode == "train")
                        L_sindy, L_coef = self.sindy_calc.compute_loss_from_precomputed(
                            z_clean, Ub, local_slices, require_grad=require_grad, reduce="sum"
                        )
                        sindy_sum += float(L_sindy.item())
                        coef_sum += float(L_coef.item())
                else:
                    L_sindy = torch.zeros((), dtype=self.dtype, device=self.device)
                    L_coef = torch.zeros((), dtype=self.dtype, device=self.device)
                sindy_timer.stop()
                
                # Hurwitz + Steady
                hurwitz_pen = torch.zeros((), dtype=self.dtype, device=self.device)
                steady_loss = torch.zeros((), dtype=self.dtype, device=self.device)
                
                # Individual steady-loss variables
                L_fse_s = torch.zeros((), dtype=self.dtype, device=self.device)
                L_frac_s = torch.zeros((), dtype=self.dtype, device=self.device)
                L_ion_s = torch.zeros((), dtype=self.dtype, device=self.device)
                L_zbar_s = torch.zeros((), dtype=self.dtype, device=self.device)
                
                Z2 = z_clean.squeeze(1) if z_clean.dim() == 3 else z_clean
                
                # Hurwitz penalty
                hurwitz_timer.start()
                hurwitz_mode = lm.get("hurwitz", "off")
                
                if self.use_adaptive_sindy:
                    # Adaptive: Hurwitz stability is guaranteed structurally, so no penalty is needed
                    # Still compute eigenvalues for monitoring
                    if cfg.hurwitz.enable and hurwitz_mode != "off":
                        with torch.no_grad():
                            max_real_batch = -float('inf')
                            min_real_batch = float('inf')
                            for sl in local_slices:
                                U_repr = Ub[sl.start:sl.stop].mean(dim=0)
                                eigvals = self.sindy_model.get_eigenvalues(U_repr)
                                max_real = eigvals.real.max().item()
                                min_real = eigvals.real.min().item()
                                max_real_batch = max(max_real_batch, max_real)
                                min_real_batch = min(min_real_batch, min_real)
                            max_real_ep = max(max_real_ep, max_real_batch)
                            min_real_ep = min(min_real_ep, min_real_batch)
                    # hurwitz_pen = 0 (structurally guaranteed)
                else:
                    # Existing lstsq method
                    if cfg.hurwitz.enable and hurwitz_mode != "off":
                        _ctx_h = torch.no_grad() if hurwitz_mode == "monitor" else _nullctx()
                        with _ctx_h:
                            hurwitz_pen, max_real, min_real = self.sindy_calc.compute_hurwitz_penalty(
                                Z2, Ub, cfg.hurwitz.margin, 
                                cfg.hurwitz.gate_enable, cfg.hurwitz.gate_min_real
                            )
                        hurwitz_sum += float(hurwitz_pen.item())
                        max_real_ep = max(max_real_ep, max_real)
                        min_real_ep = min(min_real_ep, min_real)
                hurwitz_timer.stop()
                
                # Steady-state loss
                steady_timer.start()
                steady_mode = lm.get("steady", "off")
                
                if self.steady_data and self.steady_data.enabled and steady_W_t is not None and steady_mode != "off":
                    _ctx_s = torch.no_grad() if steady_mode == "monitor" else _nullctx()
                    
                    with _ctx_s:
                        if self.use_adaptive_sindy:
                            # Adaptive: compare Z_ss and Z*(U_ss)
                            steady_W4 = steady_W_t.unsqueeze(1).unsqueeze(1)
                            z_ss = self.ae.encoder(steady_W4)  # (N, nz)
                            
                            # Steady-state latent loss
                            L_steady_latent = self.sindy_model.compute_steady_loss(z_ss, steady_U_t)
                            
                            # Also compute reconstruction through the decoder
                            z_star = self.sindy_model.get_equilibrium_batch(steady_U_t)
                            steady_Wr = self.ae.decoder(z_star)
                            
                            # Fraction conversion
                            F_truth_s = self.scale_helper.W_to_fraction(steady_W4)
                            F_pred_s = self.scale_helper.W_to_fraction(steady_Wr)
                            F_truth_sb = F_truth_s.view(F_truth_s.size(0), -1)
                            F_pred_sb = F_pred_s.view(F_pred_s.size(0), -1)
                            
                            # FSE
                            S_truth_s = self.scale_helper.get_sum_before_normalize(steady_W4)
                            S_pred_s = self.scale_helper.get_sum_before_normalize(steady_Wr)
                            L_fse_s = loss_fn(S_pred_s, S_truth_s)
                            
                            # Fraction / Ion / Zbar loss
                            L_frac_s = self.physics_loss.compute_fraction_loss(F_pred_sb, F_truth_sb)
                            L_ion_s = self.physics_loss.compute_ion_loss(F_pred_sb, F_truth_sb)
                            L_zbar_s = self.physics_loss.compute_zbar_loss(F_pred_sb, F_truth_sb)
                            
                            # Sum steady loss (including latent loss)
                            steady_loss = (
                                L_steady_latent  # Z_ss ~= Z*(U)
                                + w_fse_dyn * L_fse_s
                                + w_frac_dyn * L_frac_s
                                + w_ion_dyn * L_ion_s
                                + w_zbar_dyn * L_zbar_s
                            )
                        else:
                            # Existing lstsq method
                            ss_cfg = cfg.steady_stability
                            coef_vec = self.sindy_calc.compute_global_coefs(Z2, Ub)
                            a_t, A_t, B_t = split_coefs_torch(coef_vec, cfg.model.latent_dim, self.mu)
                            
                            z_star, condA = zstar_exact_torch(steady_U_t, a_t, A_t, B_t,
                                                              cond_warn=ss_cfg.cond_threshold)
                            
                            # Accumulate cond(A)
                            ep_condA_max = max(ep_condA_max, condA)
                            ep_condA_min = min(ep_condA_min, condA)
                            
                            # Method 2: skip based on cond(A)
                            if ss_cfg.cond_skip_enable and condA > ss_cfg.cond_threshold:
                                ep_steady_skipped += 1
                                # steady_loss = 0 (keep the initialized value)
                            else:
                                # Method 1: z_star clipping
                                if ss_cfg.zstar_clip_enable:
                                    z_star = torch.clamp(z_star,
                                                         -ss_cfg.zstar_clip_val,
                                                         ss_cfg.zstar_clip_val)
                                
                                steady_W4 = steady_W_t.unsqueeze(1).unsqueeze(1)
                                steady_Wr = self.ae.decoder(z_star)
                                
                                # W-space reconstruction loss (for reference)
                                L_recW_s = loss_fn(steady_W4, steady_Wr)
                                
                                # Fraction conversion
                                F_truth_s = self.scale_helper.W_to_fraction(steady_W4)
                                F_pred_s = self.scale_helper.W_to_fraction(steady_Wr)
                                F_truth_sb = F_truth_s.view(F_truth_s.size(0), -1)
                                F_pred_sb = F_pred_s.view(F_pred_s.size(0), -1)
                                
                                # FSE (sum before normalization)
                                S_truth_s = self.scale_helper.get_sum_before_normalize(steady_W4)
                                S_pred_s = self.scale_helper.get_sum_before_normalize(steady_Wr)
                                L_fse_s = loss_fn(S_pred_s, S_truth_s)
                                
                                # Fraction loss
                                L_frac_s = self.physics_loss.compute_fraction_loss(F_pred_sb, F_truth_sb)
                                
                                # Ion / Zbar loss
                                L_ion_s = self.physics_loss.compute_ion_loss(F_pred_sb, F_truth_sb)
                                L_zbar_s = self.physics_loss.compute_zbar_loss(F_pred_sb, F_truth_sb)
                                
                                # Sum steady loss
                                steady_loss = (
                                    w_fse_dyn * L_fse_s
                                    + w_frac_dyn * L_frac_s
                                    + w_ion_dyn * L_ion_s
                                    + w_zbar_dyn * L_zbar_s
                                )
                                
                                # Method 4: steady_loss clamp
                                if ss_cfg.loss_clip_enable:
                                    steady_loss = torch.clamp(steady_loss,
                                                              max=ss_cfg.loss_clip_max)
                    
                    steady_sum += float(steady_loss.item())
                    
                    # Accumulate individual losses
                    ep_steady_fse += float(L_fse_s.item())
                    ep_steady_frac += float(L_frac_s.item())
                    ep_steady_ion += float(L_ion_s.item())
                    ep_steady_zbar += float(L_zbar_s.item())
                steady_timer.stop()
                
                n_batches += 1
                
                # Total loss - only losses in "train" mode are included in backprop
                loss_total = torch.zeros((), dtype=self.dtype, device=self.device)
                
                # rec is always trained (core AE reconstruction)
                if lm.get("rec", "train") == "train":
                    loss_total = loss_total + cfg.train.w_rec * L_recW
                if lm.get("fse", "off") == "train":
                    loss_total = loss_total + w_fse_dyn * L_fse
                if lm.get("frac", "off") == "train":
                    loss_total = loss_total + w_frac_dyn * L_frac
                if lm.get("ion", "off") == "train":
                    loss_total = loss_total + w_ion_dyn * L_ion
                if lm.get("zbar", "off") == "train":
                    loss_total = loss_total + w_zbar_dyn * L_zbar
                if lm.get("sindy", "off") == "train":
                    loss_total = loss_total + w_sindy_dyn * (L_sindy / max(1, self.n_time_train))
                if lm.get("coef", "off") == "train":
                    loss_total = loss_total + w_coef_dyn * (L_coef / max(1, self.n_time_train))
                if lm.get("hurwitz", "off") == "train":
                    loss_total = loss_total + w_hurwitz_dyn * hurwitz_pen
                if lm.get("steady", "off") == "train":
                    loss_total = loss_total + w_steady_dyn * steady_loss
                if lm.get("rate_W", "off") == "train":
                    loss_total = loss_total + w_rate_W_dyn * L_rate_W
                if lm.get("rate_N", "off") == "train":
                    loss_total = loss_total + w_rate_N_dyn * L_rate_N
                if lm.get("rate_CSD", "off") == "train":
                    loss_total = loss_total + w_rate_CSD_dyn * L_rate_CSD
                if lm.get("rate_Zbar", "off") == "train":
                    loss_total = loss_total + w_rate_Zbar_dyn * L_rate_Zbar
                
                # Backward
                back_timer.start()
                loss_total.backward()
                back_timer.stop()
                
                opt_timer.start()
                opt.step()
                opt_timer.stop()
                
                # Accumulate (reduce 13 .item() calls to 1 transfer)
                B = xb.size(0)
                _losses = torch.stack([
                    L_recW, L_frac, L_ion, L_zbar, L_fse,
                    L_sindy, L_coef, hurwitz_pen, steady_loss,
                    L_rate_W, L_rate_N, L_rate_CSD, L_rate_Zbar
                ]).detach().cpu().numpy()
                ep_recW    += float(_losses[0])  * B
                ep_frac    += float(_losses[1])  * B
                ep_ion     += float(_losses[2])  * B
                ep_zbar    += float(_losses[3])  * B
                ep_fse     += float(_losses[4])  * B
                ep_sindy   += float(_losses[5])  * B
                ep_coef    += float(_losses[6])  * B
                ep_hurwitz += float(_losses[7])  * B
                ep_steady  += float(_losses[8])  * B
                ep_rate_W  += float(_losses[9])  * B
                ep_rate_N  += float(_losses[10]) * B
                ep_rate_CSD+= float(_losses[11]) * B
                ep_rate_Zbar+=float(_losses[12]) * B
                n_samps += B
            
            # ==================== Epoch End ====================
            
            # Normalize
            ep_recW /= max(1, n_samps)
            ep_frac /= max(1, n_samps)
            ep_ion /= max(1, n_samps)
            ep_zbar /= max(1, n_samps)
            ep_fse /= max(1, n_samps)
            ep_sindy /= max(1, n_samps)
            ep_coef /= max(1, n_samps)
            ep_hurwitz /= max(1, n_samps)
            ep_steady /= max(1, n_samps)
            ep_rate_W /= max(1, n_samps)
            ep_rate_N /= max(1, n_samps)
            ep_rate_CSD /= max(1, n_samps)
            ep_rate_Zbar /= max(1, n_samps)
            
            sindy_norm = sindy_sum / max(1, self.n_time_train)
            coef_norm = coef_sum / max(1, self.n_time_train)
            hurwitz_avg = hurwitz_sum / max(1, n_batches) if n_batches > 0 else 0.0
            steady_avg = steady_sum / max(1, n_batches) if n_batches > 0 else 0.0
            rate_W_avg = rate_W_sum / max(1, n_batches) if n_batches > 0 else 0.0
            rate_N_avg = rate_N_sum / max(1, n_batches) if n_batches > 0 else 0.0
            rate_CSD_avg = rate_CSD_sum / max(1, n_batches) if n_batches > 0 else 0.0
            rate_Zbar_avg = rate_Zbar_sum / max(1, n_batches) if n_batches > 0 else 0.0
            
            # Individual steady averages
            steady_fse_avg = ep_steady_fse / max(1, n_batches) if n_batches > 0 else 0.0
            steady_frac_avg = ep_steady_frac / max(1, n_batches) if n_batches > 0 else 0.0
            steady_ion_avg = ep_steady_ion / max(1, n_batches) if n_batches > 0 else 0.0
            steady_zbar_avg = ep_steady_zbar / max(1, n_batches) if n_batches > 0 else 0.0
            
            # Hurwitz gate check
            if cfg.hurwitz.gate_enable:
                hurwitz_ok_ep, strong_ok_ep = hurwitz_gate_check(
                    max_real_ep, min_real_ep, cfg.hurwitz.gate_min_real
                )
            
            # Total loss (with weights)
            ep_total = (
                cfg.train.w_rec * ep_recW
                + w_fse_dyn * ep_fse
                + w_frac_dyn * ep_frac
                + w_ion_dyn * ep_ion
                + w_zbar_dyn * ep_zbar
                + w_sindy_dyn * sindy_norm
                + w_coef_dyn * coef_norm
                + w_hurwitz_dyn * hurwitz_avg
                + w_steady_dyn * steady_avg
                + w_rate_W_dyn * rate_W_avg
                + w_rate_N_dyn * rate_N_avg
                + w_rate_CSD_dyn * rate_CSD_avg
                + w_rate_Zbar_dyn * rate_Zbar_avg
            )
            
            # Validation
            val_timer.start()
            ep_val = None
            if self.val_idx is not None and len(self.val_idx) > 0:
                self.ae.eval()
                with torch.no_grad():
                    val_sum, n_val = 0.0, 0
                    for slices_batch in iter_case_minibatches(self.val_slices, cfg.train.batch_size):
                        idx_batch, _ = build_local_indices_for_batch(slices_batch)
                        xbv = self.X_frames[idx_batch].to(device=self.device, dtype=self.dtype)
                        zv = self.ae.encoder(xbv)
                        xrv = self.ae.decoder(zv)
                        lv = loss_fn(xrv, xbv)
                        val_sum += float(lv.item()) * xbv.size(0)
                        n_val += xbv.size(0)
                    ep_val = val_sum / max(1, n_val)
            val_timer.stop()
            
            # Logging (store in CheckpointManager RAM buffer and flush to disk periodically)
            log_timer.start()
            self.train_loss_curve.append(ep_total)
            self.ckpt_mgr.buffer_loss(ep_global, ep_total)
            if ep_val is not None:
                self.ckpt_mgr.buffer_val_loss(ep_global, ep_val)
            
            # Metrics row
            row = {
                "epoch": ep_global,
                "total": ep_total,
                "recW": ep_recW,
                "FSE": ep_fse,
                "frac": ep_frac,
                "ion": ep_ion,
                "zbar": ep_zbar,
                "sindy": ep_sindy,
                "coef": ep_coef,
                "sindy_norm": sindy_norm,
                "coef_norm": coef_norm,
                "hurwitz": hurwitz_avg,
                "steady_raw": steady_avg,
                "lr": current_lr,
                "w_fse": w_fse_dyn,
                "w_frac": w_frac_dyn,
                "w_ion": w_ion_dyn,
                "w_zbar": w_zbar_dyn,
                "val_total": ep_val,
                "steady": ep_steady,
                "w_sindy": w_sindy_dyn,
                "w_coef": w_coef_dyn,
                "w_hurwitz": w_hurwitz_dyn,
                "w_steady": w_steady_dyn,
                "rate_W": rate_W_avg,
                "rate_N": rate_N_avg,
                "rate_CSD": rate_CSD_avg,
                "rate_Zbar": rate_Zbar_avg,
                "w_rate_W": w_rate_W_dyn,
                "w_rate_N": w_rate_N_dyn,
                "w_rate_CSD": w_rate_CSD_dyn,
                "w_rate_Zbar": w_rate_Zbar_dyn,
                "condA_max": ep_condA_max if not self.use_adaptive_sindy else float('nan'),
                "steady_skipped": ep_steady_skipped,
            }
            self.ckpt_mgr.buffer_metrics(row)
            
            # Periodic disk flush
            if ep % cfg.save.disk_save_every == 0 or ep == cfg.train.epochs:
                self.ckpt_mgr.flush_all()
            
            # Compute activated metrics (periodically)
            act_enable = cfg.eval.activated_enable if hasattr(cfg, 'eval') else True
            act_every = cfg.eval.activated_every if hasattr(cfg, 'eval') and cfg.eval.activated_every > 0 else cfg.save.disk_save_every
            if act_enable and (ep % act_every == 0 or ep == cfg.train.epochs):
                try:
                    self._compute_activated_metrics(ep_global)
                except Exception as e:
                    print(f"  [Activated Metrics] Error: {e}")
            log_timer.stop()
            
            # LR scheduling
            if scheduler is not None:
                if isinstance(scheduler, dict) and scheduler["type"] == "warmup_cosine":
                    if ep <= scheduler["warmup_epochs"]:
                        scheduler["warmup"].step()
                    else:
                        scheduler["main"].step()
                else:
                    scheduler.step()
            current_lr = get_lr(opt)
            
            # Best-model selection (track train best and val best independently)
            allow_update = True
            
            if cfg.hurwitz.gate_enable:
                allow_update = hurwitz_ok_ep and strong_ok_ep
            
            min_ep = getattr(cfg.train, 'min_save_epoch', 0)
            
            if allow_update and ep_global >= min_ep:
                # SINDy weights (adaptive only)
                sindy_st = (self.sindy_model.state_dict() 
                           if self.use_adaptive_sindy and self.sindy_model is not None 
                           else None)
                sched_st = (scheduler.state_dict() 
                           if scheduler is not None and not isinstance(scheduler, dict) 
                           else None)
                extra = {}
                if self.use_adaptive_sindy and self.sindy_model is not None:
                    extra['use_adaptive_sindy'] = True
                
                # --- Train best ---
                if self.ckpt_mgr.train_best.update(
                    ep_total, ep_global, self.ae.state_dict(),
                    sindy_state=sindy_st, opt_state=opt.state_dict(),
                    sched_state=sched_st, extra=extra
                ):
                    # Sync legacy compatibility fields
                    self.train_best_weights = self.ckpt_mgr.train_best.best_weights
                    self.train_best_metric = self.ckpt_mgr.train_best.best_metric
                    self.train_best_epoch = self.ckpt_mgr.train_best.best_epoch
                    self.best_weights = self.train_best_weights
                    self.best_metric = self.train_best_metric
                    self.best_epoch = self.train_best_epoch
                
                # --- Val best ---
                if ep_val is not None:
                    if self.ckpt_mgr.val_best.update(
                        ep_val, ep_global, self.ae.state_dict(),
                        sindy_state=sindy_st, opt_state=opt.state_dict(),
                        sched_state=sched_st, extra=extra
                    ):
                        self.val_best_weights = self.ckpt_mgr.val_best.best_weights
                        self.val_best_metric = self.ckpt_mgr.val_best.best_metric
                        self.val_best_epoch = self.ckpt_mgr.val_best.best_epoch
            
            # Flush to disk periodically
            if ep % cfg.save.disk_save_every == 0 or ep == cfg.train.epochs:
                self.ckpt_mgr.flush_all()
                self._flush_precomputed()
            
            # ==================== Detailed logging ====================
            dt_ep = timer.stop_and_update()
            val_str = f"val={ep_val:.4e}" if ep_val is not None else "val=N/A"
            
            # ========== Weighted-loss calculation ==========
            # AE weighted losses
            w_recW = cfg.train.w_rec * ep_recW
            w_fse = w_fse_dyn * ep_fse
            w_frac = w_frac_dyn * ep_frac
            w_ion = w_ion_dyn * ep_ion
            w_zbar = w_zbar_dyn * ep_zbar
            
            # SINDy weighted losses
            w_sindy = w_sindy_dyn * sindy_norm
            w_coef = w_coef_dyn * coef_norm
            
            # Hurwitz weighted loss
            w_hurwitz = w_hurwitz_dyn * hurwitz_avg
            
            # Steady weighted loss (steady_avg is already weighted sum internally)
            w_steady = w_steady_dyn * steady_avg
            
            # Rate equation weighted losses
            w_rate_W = w_rate_W_dyn * rate_W_avg
            w_rate_N = w_rate_N_dyn * rate_N_avg
            w_rate_CSD = w_rate_CSD_dyn * rate_CSD_avg
            w_rate_Zbar = w_rate_Zbar_dyn * rate_Zbar_avg
            
            # Total loss by category (weighted)
            L_AE_total = w_recW + w_fse + w_frac + w_ion + w_zbar
            L_SINDy_total = w_sindy + w_coef
            L_Hurwitz_total = w_hurwitz
            L_Steady_total = w_steady
            L_RateEq_total = w_rate_W + w_rate_N + w_rate_CSD + w_rate_Zbar
            
            # Raw loss by category (unweighted)
            L_AE_raw = ep_recW + ep_fse + ep_frac + ep_ion + ep_zbar
            L_SINDy_raw = sindy_norm + coef_norm
            L_Hurwitz_raw = hurwitz_avg
            L_Steady_raw = steady_avg
            L_RateEq_raw = rate_W_avg + rate_N_avg + rate_CSD_avg + rate_Zbar_avg
            
            # Timing calculation
            t_data = data_timer.value()
            t_ae = ae_timer.value()
            t_phys = phys_timer.value()
            t_rate = rate_timer.value()
            t_sindy = sindy_timer.value()
            t_hurwitz = hurwitz_timer.value()
            t_steady = steady_timer.value()
            t_back = back_timer.value()
            t_opt = opt_timer.value()
            t_val = val_timer.value()
            t_log = log_timer.value()
            t_measured = t_data + t_ae + t_phys + t_rate + t_sindy + t_hurwitz + t_steady + t_back + t_opt + t_val + t_log
            t_total = dt_ep if dt_ep is not None else t_measured
            t_unaccounted = max(0.0, t_total - t_measured)
            pct_unaccounted = 100 * t_unaccounted / t_total if t_total > 0 else 0.0

            if t_total > 0:
                pct_data = 100 * t_data / t_total
                pct_ae = 100 * t_ae / t_total
                pct_phys = 100 * t_phys / t_total
                pct_rate = 100 * t_rate / t_total
                pct_sindy = 100 * t_sindy / t_total
                pct_hurwitz = 100 * t_hurwitz / t_total
                pct_steady = 100 * t_steady / t_total
                pct_back = 100 * t_back / t_total
                pct_opt = 100 * t_opt / t_total
                pct_val = 100 * t_val / t_total
                pct_log = 100 * t_log / t_total
            else:
                pct_data = pct_ae = pct_phys = pct_rate = pct_sindy = 0.0
                pct_hurwitz = pct_steady = pct_back = pct_opt = pct_val = pct_log = 0.0
            
            # ==================== Multiline output ====================
            # mode tag helper
            def _mtag(key):
                m = lm.get(key, "off")
                if m == "monitor": return "[M]"
                if m == "off": return "[X]"
                return ""
            
            print(f"\n{'='*110}")
            print(f"[Epoch {ep_global:6d}] TOTAL={ep_total:.4e} | {val_str} | lr={current_lr:.2e} | train_best={self.train_best_metric:.4e}@{self.train_best_epoch} | val_best={self.val_best_metric:.4e}@{self.val_best_epoch}")
            print(f"{'-'*110}")
            
            # Line 1: weighted total by category (raw in parentheses)
            print(f"  [Summary] AE={L_AE_total:.3e}({L_AE_raw:.2e}) | SINDy={L_SINDy_total:.3e}({L_SINDy_raw:.2e}) | "
                  f"Hurwitz={L_Hurwitz_total:.3e}({L_Hurwitz_raw:.2e}) | Steady={L_Steady_total:.3e}({L_Steady_raw:.2e}) | "
                  f"RateEq={L_RateEq_total:.3e}({L_RateEq_raw:.2e})")
            print(f"{'-'*110}")
            
            # Line 2: AE-related losses - weighted(raw)
            print(f"  [AE]      recW={w_recW:.2e}({ep_recW:.2e}) | FSE{_mtag('fse')}={w_fse:.2e}({ep_fse:.2e}) | "
                  f"frac{_mtag('frac')}={w_frac:.2e}({ep_frac:.2e}) | ion{_mtag('ion')}={w_ion:.2e}({ep_ion:.2e}) | zbar{_mtag('zbar')}={w_zbar:.2e}({ep_zbar:.2e})")
            print(f"            weights: w_rec={cfg.train.w_rec:.1e} | w_fse={w_fse_dyn:.1e} | w_frac={w_frac_dyn:.1e} | w_ion={w_ion_dyn:.1e} | w_zbar={w_zbar_dyn:.1e}")
            
            # Line 3: SINDy-related losses - weighted(raw)
            print(f"  [SINDy]   sindy{_mtag('sindy')}={w_sindy:.2e}({sindy_norm:.2e}) | coef{_mtag('coef')}={w_coef:.2e}({coef_norm:.2e})")
            print(f"            weights: w_sindy={w_sindy_dyn:.1e} | w_coef={w_coef_dyn:.1e}")
            
            # Line 4: Hurwitz loss - weighted(raw)
            if cfg.hurwitz.enable and lm.get("hurwitz", "off") != "off":
                print(f"  [Hurwitz] {_mtag('hurwitz')}penalty={w_hurwitz:.2e}({hurwitz_avg:.2e}) | w_hurwitz={w_hurwitz_dyn:.1e} | max_Re={max_real_ep:.3e} | min_Re={min_real_ep:.3e} | stable={hurwitz_ok_ep}")
            else:
                print(f"  [Hurwitz] OFF")
            
            # Line 5: Steady-state loss - weighted(raw)
            if self.steady_data and self.steady_data.enabled and lm.get("steady", "off") != "off":
                print(f"  [Steady]  {_mtag('steady')}total={w_steady:.2e}({steady_avg:.2e}) | w_steady={w_steady_dyn:.1e}")
                print(f"            fse={steady_fse_avg:.2e} | frac={steady_frac_avg:.2e} | ion={steady_ion_avg:.2e} | zbar={steady_zbar_avg:.2e}")
                # Print condition number / skip only for the lstsq method
                if not self.use_adaptive_sindy:
                        ss_cfg = cfg.steady_stability
                        condA_min_str = f"{ep_condA_min:.2e}" if ep_condA_min < float("inf") else "N/A"
                        print(f"            cond(A): max={ep_condA_max:.2e} | min={condA_min_str} | "
                              f"skipped_batches={ep_steady_skipped}/{n_batches} "
                              f"[clip={ss_cfg.zstar_clip_enable}|skip={ss_cfg.cond_skip_enable}|lossclip={ss_cfg.loss_clip_enable}]")
            else:
                print(f"  [Steady]  OFF")
            
            # Line 6: Rate equation loss - weighted(raw)
            print(f"  [RateEq]  W{_mtag('rate_W')}={w_rate_W:.2e}({rate_W_avg:.2e}) | N{_mtag('rate_N')}={w_rate_N:.2e}({rate_N_avg:.2e}) | "
                  f"CSD{_mtag('rate_CSD')}={w_rate_CSD:.2e}({rate_CSD_avg:.2e}) | Zbar{_mtag('rate_Zbar')}={w_rate_Zbar:.2e}({rate_Zbar_avg:.2e})")
            print(f"            weights: w_W={w_rate_W_dyn:.1e} | w_N={w_rate_N_dyn:.1e} | w_CSD={w_rate_CSD_dyn:.1e} | w_Zbar={w_rate_Zbar_dyn:.1e}")
            
            # Lines 7-8: timing profile (% + actual time)
            print(f"{'-'*110}")
            print(f"  [Time]    total={t_total:.3f}s | measured={t_measured:.3f}s | unaccounted={t_unaccounted:.3f}s({pct_unaccounted:.1f}%)")
            print(f"            data={t_data:.3f}s({pct_data:.1f}%) | AE={t_ae:.3f}s({pct_ae:.1f}%) | phys={t_phys:.3f}s({pct_phys:.1f}%) | rate={t_rate:.3f}s({pct_rate:.1f}%)")
            print(f"            sindy={t_sindy:.3f}s({pct_sindy:.1f}%) | hurwitz={t_hurwitz:.3f}s({pct_hurwitz:.1f}%) | steady={t_steady:.3f}s({pct_steady:.1f}%)")
            print(f"            back={t_back:.3f}s({pct_back:.1f}%) | opt={t_opt:.3f}s({pct_opt:.1f}%) | val={t_val:.3f}s({pct_val:.1f}%) | log={t_log:.3f}s({pct_log:.1f}%)")

            
            # ETA
            print(f"  [ETA]     {timer.eta_str(ep, cfg.train.epochs)}")
            print(f"{'='*110}")
            
            if cfg.hurwitz.gate_enable and not (hurwitz_ok_ep and strong_ok_ep):
                print(f"  [!]  Hurwitz gate BLOCKED (model not saved): OK={hurwitz_ok_ep}, strong={strong_ok_ep}")
        
        # Flush remaining RAM buffer to disk after training ends (safety guard)
        self.ckpt_mgr.flush_all()
        
        # Save precomputed data (for fast evaluate/rollout loading)
        self._save_precomputed()
        
        print(f"\n{'='*60}")
        print(f" Training Complete!")
        print(f" Train-best: {self.train_best_metric:.4e} @ epoch {self.train_best_epoch}")
        print(f" Val-best:   {self.val_best_metric:.4e} @ epoch {self.val_best_epoch}")
        print(f" Total time: {time.time() - train_start_wall:.1f}s")
        print(f"{'='*60}\n")
    
    def _flush_precomputed(self):
        """
        Save cached precomputed data to disk (lightweight)
        
        Save the coef_vec and scaler state cached by _compute_activated_metrics.
        Called together with periodic disk flushes (disk_save_every).
        """
        if not self._precomputed_dirty:
            return
        
        cfg = self.cfg
        save_dict = {}
        
        # 1. Scaler parameters (always saved)
        if self.pop_scaler is not None and self.pop_scaler._fitted:
            save_dict['pop_scaler_state'] = self.pop_scaler.save_state()
        if self.ctrl_scaler is not None and self.ctrl_scaler._fitted:
            save_dict['ctrl_scaler_state'] = self.ctrl_scaler.save_state()
        
        # 2. Train/Val split information
        if self.train_seg_ids is not None:
            save_dict['train_seg_ids'] = np.array(self.train_seg_ids)
        if self.val_seg_ids is not None:
            save_dict['val_seg_ids'] = np.array(self.val_seg_ids)
        if self.segment_slices is not None:
            save_dict['segment_lengths'] = np.array([s.stop - s.start for s in self.segment_slices])
        
        # 3. Cached lstsq coefficients (approximate, based on the current model)
        if self._cached_coef_vec is not None:
            save_dict['sindy_coef_vec_train'] = self._cached_coef_vec
            save_dict['sindy_nz'] = cfg.model.latent_dim
            save_dict['sindy_mu'] = self.mu
            save_dict['sindy_dt_eff'] = self.dt_eff
            save_dict['sindy_coef_epoch'] = self._cached_coef_epoch
        
        save_path = (cfg.out_dir / "precomputed.npz").as_posix()
        np.savez(save_path, **save_dict)
        self._precomputed_dirty = False
        print(f"  [Precomputed] Flushed to disk (coef epoch={self._cached_coef_epoch})")
    
    def _save_precomputed(self):
        """
        Save exact precomputed data based on the best model when training ends
        
        Unlike _flush_precomputed, this loads the best model and recomputes exact coefficients.
        """
        cfg = self.cfg
        save_dict = {}
        
        # 1. Scaler parameters
        if self.pop_scaler is not None and self.pop_scaler._fitted:
            save_dict['pop_scaler_state'] = self.pop_scaler.save_state()
        if self.ctrl_scaler is not None and self.ctrl_scaler._fitted:
            save_dict['ctrl_scaler_state'] = self.ctrl_scaler.save_state()
        
        # 2. Train/Val split information
        if self.train_seg_ids is not None:
            save_dict['train_seg_ids'] = np.array(self.train_seg_ids)
        if self.val_seg_ids is not None:
            save_dict['val_seg_ids'] = np.array(self.val_seg_ids)
        if self.segment_slices is not None:
            save_dict['segment_lengths'] = np.array([s.stop - s.start for s in self.segment_slices])
        
        # 3. SINDy lstsq coefficients (recomputed from the best model)
        if not self.use_adaptive_sindy and self.ld is not None:
            try:
                X_dev = self.X_frames.to(device=self.device, dtype=self.dtype)
                
                # Load train-best model
                self.ckpt_mgr.train_best.load_weights(self.ae, device=self.device)
                self.ae.eval()
                
                with torch.no_grad():
                    Z_all = self.ae.encoder(X_dev)
                    if Z_all.dim() == 4:
                        Z_all = Z_all[:, 0, 0, :]
                    elif Z_all.dim() == 3:
                        Z_all = Z_all[:, 0, :]
                    Z_all_np = Z_all.cpu().numpy()
                
                Z_train_list, U_train_list = [], []
                for i in self.train_seg_ids:
                    sl = self.segment_slices[i]
                    Z_train_list.append(Z_all_np[sl.start:sl.stop])
                    U_train_list.append(self.U_all[sl.start:sl.stop])
                
                Z_tr = torch.tensor(np.vstack(Z_train_list), dtype=self.dtype, device=self.device)
                U_tr = torch.tensor(np.vstack(U_train_list), dtype=self.dtype, device=self.device)
                coef_vec = self.ld.calibrate(Z_tr, U_tr, float(self.dt_eff), compute_loss=False, numpy=True)
                
                save_dict['sindy_coef_vec_train'] = coef_vec
                save_dict['sindy_nz'] = cfg.model.latent_dim
                save_dict['sindy_mu'] = self.mu
                save_dict['sindy_dt_eff'] = self.dt_eff
                
                print(f"  [Precomputed] lstsq coef_vec computed (train-best)")
                
                # Also save val-best
                if os.path.exists(cfg.ckpt_val_best_path):
                    self.ckpt_mgr.val_best.load_weights(self.ae, device=self.device)
                    self.ae.eval()
                    
                    with torch.no_grad():
                        Z_all_v = self.ae.encoder(X_dev)
                        if Z_all_v.dim() == 4:
                            Z_all_v = Z_all_v[:, 0, 0, :]
                        elif Z_all_v.dim() == 3:
                            Z_all_v = Z_all_v[:, 0, :]
                        Z_all_v_np = Z_all_v.cpu().numpy()
                    
                    Z_train_list_v, U_train_list_v = [], []
                    for i in self.train_seg_ids:
                        sl = self.segment_slices[i]
                        Z_train_list_v.append(Z_all_v_np[sl.start:sl.stop])
                        U_train_list_v.append(self.U_all[sl.start:sl.stop])
                    
                    Z_tr_v = torch.tensor(np.vstack(Z_train_list_v), dtype=self.dtype, device=self.device)
                    U_tr_v = torch.tensor(np.vstack(U_train_list_v), dtype=self.dtype, device=self.device)
                    coef_vec_v = self.ld.calibrate(Z_tr_v, U_tr_v, float(self.dt_eff), compute_loss=False, numpy=True)
                    
                    save_dict['sindy_coef_vec_val'] = coef_vec_v
                    print(f"  [Precomputed] lstsq coef_vec computed (val-best)")
                
            except Exception as e:
                print(f"  [Precomputed] Warning: best-model SINDy coef failed: {e}")
                # Fallback: save cached values at least
                if self._cached_coef_vec is not None and 'sindy_coef_vec_train' not in save_dict:
                    save_dict['sindy_coef_vec_train'] = self._cached_coef_vec
                    save_dict['sindy_nz'] = cfg.model.latent_dim
                    save_dict['sindy_mu'] = self.mu
                    save_dict['sindy_dt_eff'] = self.dt_eff
                    print(f"  [Precomputed] Using cached coef_vec (epoch {self._cached_coef_epoch})")
        
        # Saving
        save_path = (cfg.out_dir / "precomputed.npz").as_posix()
        np.savez(save_path, **save_dict)
        self._precomputed_dirty = False
        print(f"  [Precomputed] Final save to {save_path}")
    
    def _compute_activated_metrics(self, epoch: int):
        """
        Compute activated metrics (MRE, MSE) - fraction, CSD, Zbar
        
        Called periodically to print to logs and save CSV files
        """
        cfg = self.cfg
        frac_th = cfg.eval.frac_threshold if hasattr(cfg, 'eval') else 1e-5
        csd_th = cfg.eval.csd_threshold if hasattr(cfg, 'eval') else 1e-5
        
        self.ae.eval()
        
        with torch.no_grad():
            # AE reconstruction on all data
            X_dev = self.X_frames.to(device=self.device, dtype=self.dtype)
            Z_all = self.ae.encoder(X_dev)
            if Z_all.dim() == 4:
                Z_all = Z_all[:, 0, 0, :]
            elif Z_all.dim() == 3:
                Z_all = Z_all[:, 0, :]
            
            W_recon = self.ae.decoder(Z_all)
            if W_recon.dim() == 4:
                W_recon = W_recon[:, 0, 0, :]
            elif W_recon.dim() == 3:
                W_recon = W_recon[:, 0, :]
            
            truth_frac = self.scale_helper.W_to_fraction(X_dev).cpu().numpy().reshape(-1, cfg.data.nx)
            pred_frac_ae = self.scale_helper.W_to_fraction(
                W_recon.unsqueeze(1).unsqueeze(1)
            ).cpu().numpy().reshape(-1, cfg.data.nx)
            
            # SINDy simulation
            Z_all_np = Z_all.cpu().numpy()
            Z_pred_sindy = np.zeros_like(Z_all_np)
            
            if self.use_adaptive_sindy and self.sindy_model is not None:
                from scipy.integrate import solve_ivp
                U_seg_all = torch.tensor(self.U_all, dtype=self.dtype, device=self.device)
                
                for sl in self.segment_slices:
                    L = sl.stop - sl.start
                    z0 = Z_all_np[sl.start]
                    U_seg_t = U_seg_all[sl.start:sl.stop]
                    t_grid = np.linspace(0.0, (L-1)*self.dt_eff, L)
                    _dt_eff = self.dt_eff
                    
                    def _ode(t, z, _U=U_seg_t, _L=L, _dt=_dt_eff):
                        t_idx = int(min(t / _dt, _L-1))
                        U_t = _U[t_idx].unsqueeze(0)
                        with torch.no_grad():
                            a_t, A_t = self.sindy_model.get_coefficients_batch(U_t)
                        a_np = a_t.squeeze().cpu().numpy()
                        A_np = A_t.squeeze().cpu().numpy()
                        return a_np + z @ A_np.T
                    
                    sol = solve_ivp(_ode, [0, (L-1)*self.dt_eff], z0, t_eval=t_grid, method='RK45')
                    Z_pred_sindy[sl.start:sl.stop] = sol.y.T
            
            elif self.ld is not None:
                # lstsq: calibrate globally first
                Z_train_list, U_train_list = [], []
                for i in self.train_seg_ids:
                    sl = self.segment_slices[i]
                    Z_train_list.append(Z_all_np[sl.start:sl.stop])
                    U_train_list.append(self.U_all[sl.start:sl.stop])
                
                Z_tr = torch.tensor(np.vstack(Z_train_list), dtype=self.dtype, device=self.device)
                U_tr = torch.tensor(np.vstack(U_train_list), dtype=self.dtype, device=self.device)
                
                coef_vec = self.ld.calibrate(Z_tr, U_tr, float(self.dt_eff), compute_loss=False, numpy=True)
                
                # Update precomputed cache
                self._cached_coef_vec = coef_vec.copy()
                self._cached_coef_epoch = epoch
                self._precomputed_dirty = True
                
                for sl in self.segment_slices:
                    L = sl.stop - sl.start
                    z0 = Z_all_np[sl.start]
                    U_seg = self.U_all[sl.start:sl.stop]
                    t_grid = np.linspace(0.0, (L-1)*self.dt_eff, L)
                    Z_pred_sindy[sl.start:sl.stop] = self.ld.simulate(coef_vec, z0, t_grid, U=U_seg)
            
            # Decode SINDy predictions
            Z_sindy_t = torch.tensor(Z_pred_sindy, dtype=self.dtype, device=self.device)
            W_sindy = self.ae.decoder(Z_sindy_t)
            if W_sindy.dim() == 4:
                W_sindy = W_sindy[:, 0, 0, :]
            elif W_sindy.dim() == 3:
                W_sindy = W_sindy[:, 0, :]
            pred_frac_sindy = self.scale_helper.W_to_fraction(
                W_sindy.unsqueeze(1).unsqueeze(1)
            ).cpu().numpy().reshape(-1, cfg.data.nx)
        
        self.ae.train()
        
        # Compute metrics
        def _activated_mre_mse(truth, pred, threshold=None):
            if threshold is not None:
                mask = np.abs(truth) >= threshold
                t_a, p_a = truth[mask], pred[mask]
            else:
                t_a, p_a = truth.flatten(), pred.flatten()
            if t_a.size == 0:
                return float('nan'), float('nan')
            mse = float(np.mean((t_a - p_a) ** 2))
            denom = np.where(np.abs(t_a) < 1e-30, 1e-30, np.abs(t_a))
            mre = float(np.mean(np.abs(t_a - p_a) / denom))
            return mre, mse
        
        results = {}
        
        # Fraction
        frac_ae_mre, frac_ae_mse = _activated_mre_mse(truth_frac, pred_frac_ae, frac_th)
        frac_sindy_mre, frac_sindy_mse = _activated_mre_mse(truth_frac, pred_frac_sindy, frac_th)
        results.update({
            'frac_ae_mre': frac_ae_mre, 'frac_ae_mse': frac_ae_mse,
            'frac_sindy_mre': frac_sindy_mre, 'frac_sindy_mse': frac_sindy_mse,
        })
        
        # CSD & Zbar
        if self.atomic_physics and self.atomic_physics.ion_available:
            truth_csd = self.atomic_physics.compute_csd_numpy(truth_frac)
            pred_csd_ae = self.atomic_physics.compute_csd_numpy(pred_frac_ae)
            pred_csd_sindy = self.atomic_physics.compute_csd_numpy(pred_frac_sindy)
            
            csd_ae_mre, csd_ae_mse = _activated_mre_mse(truth_csd, pred_csd_ae, csd_th)
            csd_sindy_mre, csd_sindy_mse = _activated_mre_mse(truth_csd, pred_csd_sindy, csd_th)
            results.update({
                'csd_ae_mre': csd_ae_mre, 'csd_ae_mse': csd_ae_mse,
                'csd_sindy_mre': csd_sindy_mre, 'csd_sindy_mse': csd_sindy_mse,
            })
            
            truth_zbar = self.atomic_physics.compute_zbar_numpy(truth_frac)
            pred_zbar_ae = self.atomic_physics.compute_zbar_numpy(pred_frac_ae)
            pred_zbar_sindy = self.atomic_physics.compute_zbar_numpy(pred_frac_sindy)
            
            zbar_ae_mre, zbar_ae_mse = _activated_mre_mse(truth_zbar, pred_zbar_ae, None)
            zbar_sindy_mre, zbar_sindy_mse = _activated_mre_mse(truth_zbar, pred_zbar_sindy, None)
            results.update({
                'zbar_ae_mre': zbar_ae_mre, 'zbar_ae_mse': zbar_ae_mse,
                'zbar_sindy_mre': zbar_sindy_mre, 'zbar_sindy_mse': zbar_sindy_mse,
            })
        
        # Log to console
        print(f"\n  [Activated Metrics] epoch={epoch}  (frac_th={frac_th:.0e}, csd_th={csd_th:.0e})")
        for k, v in results.items():
            if 'mre' in k:
                print(f"    {k}: {v*100:.2f}%")
            else:
                print(f"    {k}: {v:.4e}")
        
        # Append to CSV
        csv_path = (self.cfg.out_dir / "activated_metrics.csv").as_posix()
        header = "epoch," + ",".join(sorted(results.keys()))
        from .train_utils import append_csv_header_if_needed
        append_csv_header_if_needed(csv_path, header)
        vals = ",".join(f"{results[k]:.12e}" for k in sorted(results.keys()))
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{vals}\n")
        
        return results
    
    def _flush_best_to_disk(self, cfg):
        """Legacy compatibility: delegate to CheckpointManager"""
        if self.ckpt_mgr is not None:
            self.ckpt_mgr.train_best.flush_to_disk()
            self.ckpt_mgr.val_best.flush_to_disk()
    
    def _flush_logs_to_disk(self, cfg):
        """Legacy compatibility: delegate to CheckpointManager"""
        if self.ckpt_mgr is not None:
            self.ckpt_mgr.flush_all()
    
    def load_best_model(self, best_type: str = "train"):
        """Load best model (delegated to CheckpointManager)
        
        Args:
            best_type: "train" or "val" - which best model to load
        """
        if self.ckpt_mgr is not None:
            tracker = self.ckpt_mgr.val_best if best_type == "val" else self.ckpt_mgr.train_best
            sindy = self.sindy_model if self.use_adaptive_sindy else None
            tracker.load_weights(self.ae, sindy, device=self.device)
        else:
            # If ckpt_mgr is unavailable (before setup_model), load directly from disk
            ckpt_path = self.cfg.ckpt_val_best_path if best_type == "val" else self.cfg.ckpt_train_best_path
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                self.ae.load_state_dict(ckpt['model_state'])
                print(f"[Model] Loaded {best_type}-best from {ckpt_path}")
            else:
                print(f"[Model] Warning: No {best_type}-best weights found")
