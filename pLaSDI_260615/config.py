# -*- coding: utf-8 -*-
"""
pLaSDI Configuration Module
============================
Module for managing all hyperparameters, paths, and configuration values.

Future extension notes:
- U scaling: currently uses W-style scaling, but T and density can use separate scalers later.
- SINDyC structure: currently linear dZ/dt = a + A·Z + B·U; can be extended to A = A(U).
"""

import os
import random
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

import torch
import numpy as np


# =============================================================================
# Loss Mode: "train" / "monitor" / "off"
# =============================================================================
VALID_LOSS_MODES = {"train", "monitor", "off"}

def validate_loss_modes(modes: Dict[str, str]):
    """Validate the loss_mode dictionary."""
    for key, mode in modes.items():
        if mode not in VALID_LOSS_MODES:
            raise ValueError(
                f"loss_mode['{key}'] = '{mode}' is invalid. "
                f"Allowed values: {VALID_LOSS_MODES}"
            )

@dataclass 
class SaveConfig:
    """Save-related settings."""
    save_root: str = './runs/dim3'  # Result output directory
    
    # Disk save interval in epochs (CSV/metrics, etc.)
    # Store in RAM every epoch and flush to disk at this interval.
    disk_save_every: int = 100

@dataclass
class DataConfig:
    """Data-related settings."""
    data_dir: str = "../../data_storage/lithography_4ns"
    history_dir: str = "../../data_storage/lithography_4ns"
    
    # State names (required for CSD and Zbar calculation)
    names_file: str = "name_total.txt"
    label_subset_file: str = "name_bound.txt"
    use_state_names: bool = True
    
    # Data dimensions
    nt: int = 400           # Number of time steps per file (reference only)
    nx: int = 1583          # Number of states (levels)
    dt: float = 0.01e-9     # [s]
    pop_lim: float = 1e-50  # Lower population bound
    
    # Steady-state data
    steady_enable: bool = True
    steady_ae: bool = False  # Whether to add SS to AE reconstruction (usually better disabled)
    steady_pop_file: str = "../../data_storage/steady_data/steady_population.txt"
    steady_hist_file: str = "../../data_storage/steady_data/steady_history.txt"
    steady_random_pick: bool = False
    steady_num_samples: int = 200
    steady_random_seed: int = 42


@dataclass
class ModelConfig:
    """Model architecture settings."""
    latent_dim: int = 3
    hidden: List[int] = field(default_factory=lambda: [800, 400, 200, 100, 20])
    activation: str = "mish"


@dataclass
class SINDyConfig:
    """SINDy-related settings."""
    weight: float = 1e-2
    coef_weight: float = 0.0
    fd_type: str = "sbp12"          # Time-derivative finite-difference stencil
    dt_eff: Optional[float] = 1e-3  # Effective dt for SINDy
    use_global_coefs: bool = True   # Used by the lstsq mode
    use_cpu: bool = False           # Whether to run lstsq calibration on CPU
    
    # Adaptive SINDy settings (CoefNet-based)
    use_adaptive: bool = False                    # True: AdaptiveSINDyC, False: existing lstsq mode
    adaptive_hidden: List[int] = field(default_factory=lambda: [32, 32])  # CoefNet hidden layers
    adaptive_eps: float = 0.0                    # Hurwitz margin ε (A = -P·Pᵀ - ε·I)
    adaptive_activation: str = "Mish"             # CoefNet activation function
    adaptive_symmetric: bool = False               # True: A symmetric, False: A = -(P·Pᵀ + S) asymmetric
    adaptive_head_gain: float = 100   # Xavier initialization gain for CoefNet output heads (head_a, head_P, head_S)
                                    # Larger values produce larger initial coefficients. gain=10 is recommended if lstsq coefficients are O(10-100).

@dataclass
class HurwitzConfig:
    """Hurwitz stability-related settings."""
    enable: bool = True
    margin: float = 0.0             # Re(λ_max) + margin ≤ 0
    weight: float = 1e-4
    gate_enable: bool = True        # Prevent saving unstable models
    gate_min_real: float = 0.0      # Minimum eigenvalue threshold


@dataclass
class SteadyStabilityConfig:
    """
    Numerical stabilization settings for the steady-state loss.
    
    Three safeguards to prevent z* = -A⁻¹(a + BU) from blowing up
    when the lstsq-derived A coefficients are ill-conditioned.
    
    Method 1 (zstar_clip): clamp z* based on the expected encoder output range.
    Method 2 (cond_skip): set the batch steady loss to 0 if cond(A) exceeds the threshold.
    Method 4 (loss_clip): clamp steady_loss itself to max_loss.
    """
    # --- Method 1: z* clipping ---
    zstar_clip_enable: bool = True   # If True, clamp z_star to [-zstar_clip_val, +zstar_clip_val]
    zstar_clip_val: float = 400.0     # Clamp range based on the expected max absolute encoder output

    # --- Method 2: skip based on cond(A) ---
    cond_skip_enable: bool = True    # If True, steady loss = 0 when cond(A) > cond_threshold
    cond_threshold: float = 1e3      # Condition number threshold

    # --- Method 4: loss clamp ---
    loss_clip_enable: bool = True    # If True, clamp steady_loss to max_loss
    loss_clip_max: float = 1e-2       # Maximum steady_loss value




@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Basic settings
    epochs: int = 100000
    batch_size: int = 40            # Case-wise minibatch
    lr: float = 1e-5
    tau: float = 1e-100             # fraction mask threshold
    
    # Loss weights (AE + Physics)
    w_rec: float = 1.0
    w_fse: float = 1e-3
    w_frac: float = 1e1
    w_ion: float = 1e-2
    w_zbar: float = 1e-4
    
    # Rate-equation weights
    w_rate_W: float = 0.0       # W-space rate equation loss
    w_rate_N: float = 0.0       # N-space rate equation loss
    w_rate_CSD: float = 0.0     # CSD rate equation loss
    w_rate_Zbar: float = 0.0    # Zbar rate equation loss
    
    # Steady-state weight
    steady_weight: float = 1e-1
    
    # Loss mode settings
    # "train"   : compute and include in backprop
    # "monitor" : compute only for logging (no_grad)
    # "off"     : do not compute
    loss_mode: Dict[str, str] = field(default_factory=lambda: {
        "rec":      "train",
        "fse":      "train",
        "frac":     "train",
        "ion":      "train",
        "zbar":     "train",
        "sindy":    "train",
        "coef":     "train",
        "hurwitz":  "train",
        "steady":   "train",
        "rate_W":   "off",
        "rate_N":   "off",
        "rate_CSD": "off",
        "rate_Zbar":"off",
    })
    
    # Ramp-up settings [exp_late, exp_early, exp_slow, cos]
    ramp_config: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "fse":     {"T": 10000,  "mode": "exp_early"},
        "frac":    {"T": 10000, "mode": "exp_early"},
        "ion":     {"T": 10000, "mode": "exp_early"},
        "zbar":    {"T": 10000, "mode": "exp_early"},
        "sindy":   {"T": 1000,   "mode": "exp_early"},
        "coef":    {"T": 1000,     "mode": "exp_early"},
        "hurwitz": {"T": 10000,   "mode": "exp_early"},
        "steady":  {"T": 10000,   "mode": "exp_early"},
        "rate_W":  {"T": 1000,  "mode": "exp_early"},
        "rate_N":  {"T": 1000,  "mode": "exp_early"},
        "rate_CSD":  {"T": 1000,  "mode": "exp_early"},
        "rate_Zbar": {"T": 1000,  "mode": "exp_early"},
    })
    
    # Noise applied to AE
    noise: float = 0.1
    
    # LR scheduling
    lr_scheduler: str = "cosine"    # "cosine", "one_cycle", "none"
    min_lr: float = 1e-5            # cosine: eta_min
    
    # Warmup settings (used by cosine)
    warmup_epochs: int = 0          # 0 disables warmup
    warmup_start_lr: float = 1e-7   # Warmup start LR
    
    # OneCycleLR only
    max_lr: float = 1e-3            # one_cycle: maximum LR (lr is used as the initial value)
    
    # Best-model save restriction
    min_save_epoch: int = 10000               # Do not save best models before this epoch (0 disables the restriction)
    
    # Validation
    val_ratio: float = 0.2
    val_split_mode: str = "random_segments"  # "random_segments" recommended
    # select_best_by was removed; both train-best and val-best are saved
    
    # Resume
    add_training: bool = False
    resume_optimizer: bool = False


@dataclass
class EvalConfig:
    """Evaluation metric settings."""
    # Activated metrics threshold
    # Compute MRE/MSE only on truth values above the threshold
    frac_threshold: float = 1e-3      # Fraction activation threshold
    csd_threshold: float = 1e-3       # CSD activation threshold
    
    # Activated metrics during training
    activated_enable: bool = True     # If False, do not compute activated metrics during training
    activated_every: int = 100        # Evaluation interval in epochs (0 uses disk_save_every)


@dataclass
class LaSDIcConfig:
    """Main config class that combines all settings."""
    save: SaveConfig = field(default_factory=SaveConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sindy: SINDyConfig = field(default_factory=SINDyConfig)
    hurwitz: HurwitzConfig = field(default_factory=HurwitzConfig)
    steady_stability: SteadyStabilityConfig = field(default_factory=SteadyStabilityConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    # Seed
    seed: int = 42
    
    # Case selection
    num_cases_total: int = 185      # Total number of available cases
    num_cases_use: int = 185        # Number of cases to use (None means all)
    case_select_mode: str = "random"  # "random", "first", "last", "manual"
    manual_case_ids: List[int] = field(default_factory=lambda: [42, 43, 44, 59, 60])  # Case numbers used when case_select_mode="manual"
    
    # Device
    cuda_device: str = "0"
    dtype: torch.dtype = torch.float64
    
    def __post_init__(self):
        """Compute paths and validate settings without side effects."""
        # Validate loss modes
        validate_loss_modes(self.train.loss_mode)
        
        # Compute paths (directory creation happens in setup())
        self.out_dir = Path(self.save.save_root)
        
        # File paths
        self.ckpt_train_best_path = (self.out_dir / "model_train_best.pt").as_posix()
        self.ckpt_val_best_path = (self.out_dir / "model_val_best.pt").as_posix()
        self.opt_ckpt_path = (self.out_dir / "train.opt.pt").as_posix()
        self.losslog_path = (self.out_dir / "train_loss.npy").as_posix()
        self.vallosslog_path = (self.out_dir / "train_val_loss.npy").as_posix()
        self.losscsv_path = (self.out_dir / "train_losslog.csv").as_posix()
        self.vallosscsv_path = (self.out_dir / "train_vallosslog.csv").as_posix()
        self.metrics_csv_path = (self.out_dir / "train_metrics.csv").as_posix()
        self.metrics_path = (self.out_dir / "train_metrics.npz").as_posix()
        self.sindy_coefs_path = (self.out_dir / "train_sindy_coefs.npz").as_posix()
        
        # Generate case numbers (use a local Random to avoid mutating global random state)
        rng = random.Random(self.seed)
        all_case_ids = list(range(self.num_cases_total))
        
        # Determine the number of cases to use
        n_use = self.num_cases_use if self.num_cases_use else self.num_cases_total
        n_use = min(n_use, self.num_cases_total)
        
        # Case selection mode
        if self.case_select_mode == "random":
            self.case_numbers = rng.sample(all_case_ids, n_use)
        elif self.case_select_mode == "first":
            self.case_numbers = all_case_ids[:n_use]
        elif self.case_select_mode == "last":
            self.case_numbers = all_case_ids[-n_use:]
        elif self.case_select_mode == "manual":
            if self.manual_case_ids is None:
                raise ValueError("manual_case_ids must be specified when case_select_mode='manual'.")
            self.case_numbers = [i for i in self.manual_case_ids if i < self.num_cases_total]
        else:
            raise ValueError(f"Unknown case_select_mode: {self.case_select_mode}")
        
        # Data file paths
        self.data_files = [
            os.path.join(self.data.data_dir, f"density_population_seg{i}.txt") 
            for i in self.case_numbers
        ]
        self.history_files = [
            os.path.join(self.data.history_dir, f"historyfile_seg{i}.txt")        
            for i in self.case_numbers
        ]
        
        # Steady-state file pairs
        self.steady_pop_hist_pairs = [
            (self.data.steady_pop_file, self.data.steady_hist_file),
        ]
        
        # Automatic setup for backward compatibility
        self.setup()
    
    def setup(self):
        """
        Initialization with side effects (directory creation, output).
        
        This is called automatically from __post_init__, so existing code works unchanged.
        Tests may mock this method after __post_init__, or set out_dir to a tempdir
        before calling setup().
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Config] Using {len(self.case_numbers)}/{self.num_cases_total} cases (mode={self.case_select_mode})")
    
    def get_device(self) -> torch.device:
        """Return the device."""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_device
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def get_gpu_info(self) -> str:
        """Return a GPU information string."""
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            mem_total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
            return f"GPU {idx}: {name} ({mem_total:.1f} GB)"
        return "CPU only"
    
    def backup_config(self, config_source_path: str = None):
        """
        Back up config.py to the output directory.
        
        For resumed training (add_training=True), save with a timestamp.
        For new training, save as config.py.
        
        Args:
            config_source_path: Source path of config.py. If None, search automatically.
        """
        # Find the config source file
        if config_source_path is None:
            # Search common locations
            candidates = [
                Path("config.py"),
                Path(__file__).parent / "config.py",
                Path.cwd() / "config.py",
            ]
            for c in candidates:
                if c.exists():
                    config_source_path = str(c)
                    break
        
        if config_source_path is None or not Path(config_source_path).exists():
            print(f"[Config] Warning: config source not found, skip backup")
            return
        
        if self.train.add_training:
            # Resumed training: include a timestamp in the filename
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_name = f"config_resume_{ts}.py"
        else:
            dest_name = "config.py"
        
        dest = self.out_dir / dest_name
        shutil.copy2(config_source_path, dest)
        print(f"[Config] Backed up → {dest}")
    
    def print_training_status(self):
        """Print the full status before training starts (GPU, loss modes, weights, etc.)."""
        lm = self.train.loss_mode
        
        print(f"\n{'='*70}")
        print(f" Training Configuration Status")
        print(f"{'='*70}")
        
        # Device / GPU
        print(f"\n  [Device]  {self.get_gpu_info()}")
        print(f"            dtype = {self.dtype}")
        
        # Model
        print(f"\n  [Model]   latent_dim = {self.model.latent_dim}")
        print(f"            hidden     = {self.model.hidden}")
        print(f"            activation = {self.model.activation}")
        
        # SINDy mode
        sindy_type = "AdaptiveSINDyC" if self.sindy.use_adaptive else "lstsq"
        print(f"            SINDy      = {sindy_type}")
        
        # Loss modes table
        print(f"\n  [Loss Modes]")
        print(f"  {'Loss':<14} {'Mode':<10} {'Weight':<12} {'Ramp-T':<8} {'Ramp-mode':<10}")
        print(f"  {'-'*54}")
        
        # Information for each loss
        loss_info = [
            ("rec",      lm.get("rec", "train"),      self.train.w_rec,           None,   None),
            ("fse",      lm.get("fse", "train"),      self.train.w_fse,           "fse",  "fse"),
            ("frac",     lm.get("frac", "train"),     self.train.w_frac,          "frac", "frac"),
            ("ion",      lm.get("ion", "train"),      self.train.w_ion,           "ion",  "ion"),
            ("zbar",     lm.get("zbar", "train"),     self.train.w_zbar,          "zbar", "zbar"),
            ("sindy",    lm.get("sindy", "train"),    self.sindy.weight,          "sindy","sindy"),
            ("coef",     lm.get("coef", "train"),     self.sindy.coef_weight,     "coef", "coef"),
            ("hurwitz",  lm.get("hurwitz", "train"),  self.hurwitz.weight,        "hurwitz","hurwitz"),
            ("steady",   lm.get("steady", "train"),   self.train.steady_weight,   "steady","steady"),
            ("rate_W",   lm.get("rate_W", "off"),     self.train.w_rate_W,        "rate_W","rate_W"),
            ("rate_N",   lm.get("rate_N", "off"),     self.train.w_rate_N,        "rate_N","rate_N"),
            ("rate_CSD", lm.get("rate_CSD", "off"),   self.train.w_rate_CSD,      "rate_CSD","rate_CSD"),
            ("rate_Zbar",lm.get("rate_Zbar", "off"),  self.train.w_rate_Zbar,     "rate_Zbar","rate_Zbar"),
        ]
        
        for name, mode, weight, ramp_key, _ in loss_info:
            # Display by mode
            if mode == "train":
                mode_str = "TRAIN"
            elif mode == "monitor":
                mode_str = "monitor"
            else:
                mode_str = "OFF"
            
            w_str = f"{weight:.1e}"
            
            if ramp_key and ramp_key in self.train.ramp_config:
                rc = self.train.ramp_config[ramp_key]
                rT = str(rc["T"])
                rm = rc["mode"]
            else:
                rT = "-"
                rm = "-"
            
            print(f"  {name:<14} {mode_str:<10} {w_str:<12} {rT:<8} {rm:<10}")
        
        # Training info
        print(f"\n  [Train]   epochs     = {self.train.epochs}")
        print(f"            batch_size = {self.train.batch_size}")
        print(f"            lr         = {self.train.lr}")
        print(f"            scheduler  = {self.train.lr_scheduler}")
        print(f"            add_train  = {self.train.add_training}")
        print(f"            best_save  = train_best + val_best (dual)")
        
        # Steady / Hurwitz / SteadyStability
        print(f"\n  [Steady]  enable     = {self.data.steady_enable}")
        ss = self.steady_stability
        print(f"  [SteadyStab] zstar_clip={ss.zstar_clip_enable}(±{ss.zstar_clip_val:.1e}) | "
              f"cond_skip={ss.cond_skip_enable}(thr={ss.cond_threshold:.1e}) | "
              f"loss_clip={ss.loss_clip_enable}(max={ss.loss_clip_max:.1e})")
        print(f"  [Hurwitz] enable     = {self.hurwitz.enable}")
        if self.hurwitz.enable:
            print(f"            gate       = {self.hurwitz.gate_enable}")
        
        print(f"\n  [Save]    root       = {self.save.save_root}")
        print(f"{'='*70}\n")
    
    def print_summary(self):
        """Print a configuration summary."""
        print("=" * 60)
        print(" LaSDIc Configuration Summary")
        print("=" * 60)
        print(f"[Data]")
        print(f"  data_dir      = {self.data.data_dir}")
        print(f"  nx={self.data.nx}, dt={self.data.dt}")
        print(f"  steady_enable = {self.data.steady_enable}")
        print(f"[Model]")
        print(f"  latent_dim    = {self.model.latent_dim}")
        print(f"  hidden        = {self.model.hidden}")
        print(f"  activation    = {self.model.activation}")
        print(f"[SINDy]")
        print(f"  weight        = {self.sindy.weight}")
        print(f"  fd_type       = {self.sindy.fd_type}")
        print(f"  dt_eff        = {self.sindy.dt_eff}")
        print(f"[Eval]")
        print(f"  frac_threshold= {self.eval.frac_threshold}")
        print(f"  csd_threshold = {self.eval.csd_threshold}")
        print(f"[Hurwitz]")
        print(f"  enable        = {self.hurwitz.enable}")
        print(f"  weight        = {self.hurwitz.weight}")
        print(f"  gate_enable   = {self.hurwitz.gate_enable}")
        print(f"[Train]")
        print(f"  epochs        = {self.train.epochs}")
        print(f"  batch_size    = {self.train.batch_size}")
        print(f"  lr            = {self.train.lr}")
        print(f"  lr_scheduler  = {self.train.lr_scheduler}")
        print(f"  min_lr        = {self.train.min_lr}")
        print(f"  val_ratio     = {self.train.val_ratio}")
        print(f"[Save]")
        print(f"  save_root     = {self.save.save_root}")
        print("=" * 60)


def create_default_config() -> LaSDIcConfig:
    """Create a config with default settings."""
    return LaSDIcConfig()
