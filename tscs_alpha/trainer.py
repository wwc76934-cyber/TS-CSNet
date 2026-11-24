"""Strict annual rolling trainer with exact purge gap and 3-seed ensembling.

For every test year Y in [2017..2025]:
  - Test dates: weekly/biweekly signal dates in year Y.
  - Purge gap : every training/validation sample must have label_end_date <= first_test_signal_date,
                guaranteeing zero overlap between training labels and test period.
  - Val split : last `val_weeks` of the lookback pool (also purged from training labels)
                used strictly for early stopping by validation RankIC.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import copy
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr

from .config import ExperimentConfig
from .dataset import PanelData, load_panel
from .losses import CompositeICRankLoss, CrossSectionalMSELoss
from .models import GRUScorer, MLPScorer, TSCSNet, count_parameters


@dataclass
class ModelRunResult:
    model_name: str
    num_params: int
    oos_scores: np.ndarray          # (K, M) float32; NaN outside OOS test years
    gate_history: Dict[str, float]  # signal_date -> mean gate value (TSCSNet only)
    yearly_val_ic: Dict[int, float] # test_year -> best validation RankIC
    elapsed_seconds: float


def compute_rolling_splits(
    panel: PanelData,
    test_year: int,
    train_years_window: int,
    val_weeks: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute strictly purged (train_idx, val_idx, test_idx) schedule indices for test_year."""
    sig = np.asarray(panel.signal_dates, dtype="<U10")
    end = np.asarray(panel.label_end_dates, dtype="<U10")

    test_idx = np.where(np.char.startswith(sig, f"{test_year}-"))[0]
    if len(test_idx) == 0:
        raise ValueError(f"No schedule dates found for test year {test_year}")

    first_test_sig = str(sig[test_idx[0]])
    lookback_start = f"{test_year - train_years_window}-01-01"

    # Candidate samples whose entire 10-day label window finishes on or before first_test_sig
    cand_mask = (
        (sig >= lookback_start)
        & (end != "")
        & (end <= first_test_sig)
        & (panel.label_valid.sum(axis=1) >= 50)
    )
    cand_idx = np.where(cand_mask)[0]
    if len(cand_idx) < 30:
        raise ValueError(f"Insufficient candidate samples ({len(cand_idx)}) for test_year={test_year}")

    n_val = min(val_weeks, max(12, len(cand_idx) // 4))
    val_idx = cand_idx[-n_val:]
    first_val_sig = str(sig[val_idx[0]])

    # Enforce exact purge gap between train labels and validation start
    train_idx = np.array([k for k in cand_idx if str(end[k]) <= first_val_sig], dtype=np.int64)
    if len(train_idx) < 20:
        raise ValueError(f"Insufficient purged train samples ({len(train_idx)}) for test_year={test_year}")

    return train_idx, val_idx, test_idx


def _zscore_1d(x: np.ndarray) -> np.ndarray:
    m = float(np.mean(x))
    s = float(np.std(x))
    if s < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - m) / s).astype(np.float32)


def _rank_ic(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 10 or np.std(pred) < 1e-10 or np.std(target) < 1e-10:
        return 0.0
    r = spearmanr(pred, target).statistic
    return 0.0 if not np.isfinite(r) else float(r)


def _build_model_and_loss(model_name: str, cfg: ExperimentConfig) -> Tuple[torch.nn.Module, torch.nn.Module]:
    seq_dim = len(cfg.seq_features)
    stat_dim = len(cfg.static_features)
    tab_dim = stat_dim + 3 * seq_dim

    if model_name == "M2_MLP_MSE":
        return (
            MLPScorer(in_dim=tab_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout),
            CrossSectionalMSELoss(),
        )
    if model_name == "M3_GRU_MSE":
        return (
            GRUScorer(seq_dim=seq_dim, static_dim=stat_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout),
            CrossSectionalMSELoss(),
        )
    if model_name == "M4_GRU_ICRank":
        return (
            GRUScorer(seq_dim=seq_dim, static_dim=stat_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout),
            CompositeICRankLoss(ranknet_weight=cfg.ranknet_weight, num_pairs=cfg.ranknet_pairs),
        )
    if model_name == "M5_TSCSNet":
        return (
            TSCSNet(
                seq_dim=seq_dim,
                static_dim=stat_dim,
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
            ),
            CompositeICRankLoss(ranknet_weight=cfg.ranknet_weight, num_pairs=cfg.ranknet_pairs),
        )
    raise ValueError(f"Unknown neural model: {model_name}")


def run_ridge_rolling(panel: PanelData, cfg: ExperimentConfig) -> ModelRunResult:
    """Closed-form rolling Ridge regression baseline on fair 24-dim tabular features."""
    t0 = time.time()
    n_sched, max_m = panel.valid_mask.shape
    oos_scores = np.full((n_sched, max_m), np.nan, dtype=np.float32)
    yearly_val_ic: Dict[int, float] = {}

    n_feat = panel.X_tab.shape[2]
    eye = np.eye(n_feat, dtype=np.float64)

    for year in cfg.test_years:
        train_idx, val_idx, test_idx = compute_rolling_splits(
            panel, year, cfg.train_years_window, cfg.val_weeks
        )
        # Evaluate validation IC on train_idx -> val_idx first for transparent reporting
        x_tr_list, y_tr_list = [], []
        for k in train_idx:
            m = panel.label_valid[k]
            x_tr_list.append(panel.X_tab[k, m])
            y_tr_list.append(panel.y_norm[k, m])
        X_tr = np.concatenate(x_tr_list, axis=0).astype(np.float64)
        y_tr = np.concatenate(y_tr_list, axis=0).astype(np.float64)
        w_val = np.linalg.solve(X_tr.T @ X_tr + cfg.ridge_alpha * eye, X_tr.T @ y_tr)

        val_ics = []
        for k in val_idx:
            m = panel.label_valid[k]
            val_ics.append(_rank_ic(panel.X_tab[k, m] @ w_val, panel.y_raw[k, m]))
        yearly_val_ic[year] = float(np.mean(val_ics))

        # Refit Ridge on full purged lookback (train + val) before scoring test_year
        x_full_list, y_full_list = [X_tr], [y_tr]
        for k in val_idx:
            m = panel.label_valid[k]
            x_full_list.append(panel.X_tab[k, m].astype(np.float64))
            y_full_list.append(panel.y_norm[k, m].astype(np.float64))
        X_full = np.concatenate(x_full_list, axis=0)
        y_full = np.concatenate(y_full_list, axis=0)
        w_full = np.linalg.solve(X_full.T @ X_full + cfg.ridge_alpha * eye, X_full.T @ y_full)

        for k in test_idx:
            m = panel.valid_mask[k]
            preds = (panel.X_tab[k, m].astype(np.float64) @ w_full).astype(np.float32)
            oos_scores[k, m] = _zscore_1d(preds)

    return ModelRunResult(
        model_name="M1_Ridge",
        num_params=n_feat,
        oos_scores=oos_scores,
        gate_history={},
        yearly_val_ic=yearly_val_ic,
        elapsed_seconds=time.time() - t0,
    )


def _train_single_seed_year(
    model_name: str,
    panel: PanelData,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: ExperimentConfig,
    seed: int,
) -> Tuple[Dict[int, np.ndarray], Dict[str, float], float, int]:
    """Train one neural model on one rolling year with validation RankIC early stopping."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model, loss_fn = _build_model_and_loss(model_name, cfg)
    n_params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.1
    )

    # Pre-slice tensors for train/val cross-sections so epochs have zero numpy overhead
    use_rank_target = model_name in ("M4_GRU_ICRank", "M5_TSCSNet")
    target_mat = panel.y_rank if use_rank_target else panel.y_norm
    train_batches = []
    for k in train_idx:
        m = panel.label_valid[k]
        train_batches.append((
            torch.from_numpy(panel.X_seq[k, m]),
            torch.from_numpy(panel.X_stat[k, m]),
            torch.from_numpy(panel.X_tab[k, m]),
            torch.from_numpy(target_mat[k, m]),
        ))

    val_batches = []
    for k in val_idx:
        m = panel.label_valid[k]
        val_batches.append((
            torch.from_numpy(panel.X_seq[k, m]),
            torch.from_numpy(panel.X_stat[k, m]),
            torch.from_numpy(panel.X_tab[k, m]),
            panel.y_raw[k, m],
        ))

    best_val_ic = -1e9
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    accum = max(1, cfg.grad_accum_dates)

    for _epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(train_batches))
        optimizer.zero_grad()

        step_count = 0
        for pos, idx in enumerate(order):
            x_seq, x_stat, x_tab, y_t = train_batches[int(idx)]
            scores = model(x_seq, x_stat, x_tab)
            loss = loss_fn(scores, y_t) / accum
            loss.backward()
            step_count += 1

            is_last = (pos == len(order) - 1)
            if step_count == accum or is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                step_count = 0

        scheduler.step()

        # Validation RankIC evaluation
        model.eval()
        val_ics = []
        with torch.no_grad():
            for x_seq, x_stat, x_tab, y_raw_np in val_batches:
                pred = model(x_seq, x_stat, x_tab).numpy()
                val_ics.append(_rank_ic(pred, y_raw_np))
        mean_val_ic = float(np.mean(val_ics))

        if mean_val_ic > best_val_ic + 1e-4:
            best_val_ic = mean_val_ic
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break

    # Restore best validation checkpoint and score OOS test year
    model.load_state_dict(best_state)
    model.eval()
    test_preds: Dict[int, np.ndarray] = {}
    gate_vals: Dict[str, float] = {}

    with torch.no_grad():
        for k in test_idx:
            m = panel.valid_mask[k]
            x_seq = torch.from_numpy(panel.X_seq[k, m])
            x_stat = torch.from_numpy(panel.X_stat[k, m])
            x_tab = torch.from_numpy(panel.X_tab[k, m])
            scores_t, gate_mean = model(x_seq, x_stat, x_tab, return_gate=True)
            test_preds[int(k)] = _zscore_1d(scores_t.numpy())
            if np.isfinite(gate_mean):
                gate_vals[str(panel.signal_dates[k])] = gate_mean

    return test_preds, gate_vals, best_val_ic, n_params


_WORKER_PANEL: PanelData | None = None


def _init_worker(cfg: ExperimentConfig) -> None:
    global _WORKER_PANEL
    torch.set_num_threads(1)
    _WORKER_PANEL = load_panel(cfg)


def _worker_train_job(
    args: Tuple[str, int, int, ExperimentConfig]
) -> Tuple[int, int, Dict[int, np.ndarray], Dict[str, float], float, int]:
    model_name, year, seed, cfg = args
    assert _WORKER_PANEL is not None
    panel = _WORKER_PANEL
    train_idx, val_idx, test_idx = compute_rolling_splits(
        panel, year, cfg.train_years_window, cfg.val_weeks
    )
    preds, gates, val_ic, n_params = _train_single_seed_year(
        model_name=model_name,
        panel=panel,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        cfg=cfg,
        seed=seed + year,
    )
    return year, seed, preds, gates, val_ic, n_params


def run_neural_rolling(
    model_name: str,
    panel: PanelData,
    cfg: ExperimentConfig,
    verbose: bool = True,
    max_workers: int | None = None,
) -> ModelRunResult:
    """Run full 9-year annual rolling training + 3-seed ensemble for a PyTorch model.

    Uses isolated worker processes with 1 PyTorch thread each for deterministic,
    multi-core CPU parallelism across (year, seed) jobs.
    """
    t0 = time.time()
    n_sched, max_m = panel.valid_mask.shape
    oos_scores = np.full((n_sched, max_m), np.nan, dtype=np.float32)
    gate_accum: Dict[str, List[float]] = {}
    yearly_val_ic: Dict[int, float] = {}
    n_params = 0

    jobs = [(model_name, year, seed, cfg) for year in cfg.test_years for seed in cfg.seeds]
    n_workers = max_workers if max_workers is not None else min(6, max(1, (os.cpu_count() or 2) // 2))

    year_seed_preds: Dict[int, Dict[int, List[np.ndarray]]] = {y: {} for y in cfg.test_years}
    year_val_ics: Dict[int, List[float]] = {y: [] for y in cfg.test_years}

    if n_workers > 1 and cfg.cache_path.exists():
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for year, _seed, preds, gates, val_ic, n_p in pool.map(_worker_train_job, jobs):
                n_params = n_p
                year_val_ics[year].append(val_ic)
                for k_int, arr in preds.items():
                    year_seed_preds[year].setdefault(k_int, []).append(arr)
                for d_str, g_val in gates.items():
                    gate_accum.setdefault(d_str, []).append(g_val)
    else:
        torch.set_num_threads(1)
        for year in cfg.test_years:
            train_idx, val_idx, test_idx = compute_rolling_splits(
                panel, year, cfg.train_years_window, cfg.val_weeks
            )
            for seed in cfg.seeds:
                preds, gates, val_ic, n_params = _train_single_seed_year(
                    model_name=model_name,
                    panel=panel,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    cfg=cfg,
                    seed=seed + year,
                )
                year_val_ics[year].append(val_ic)
                for k_int, arr in preds.items():
                    year_seed_preds[year].setdefault(k_int, []).append(arr)
                for d_str, g_val in gates.items():
                    gate_accum.setdefault(d_str, []).append(g_val)

    for year in cfg.test_years:
        _, _, test_idx = compute_rolling_splits(panel, year, cfg.train_years_window, cfg.val_weeks)
        for k in test_idx:
            m = panel.valid_mask[k]
            avg_pred = np.mean(np.stack(year_seed_preds[year][int(k)], axis=0), axis=0)
            oos_scores[k, m] = _zscore_1d(avg_pred)
        yearly_val_ic[year] = float(np.mean(year_val_ics[year]))
        if verbose:
            print(f"  [{model_name}] Year {year}: val_RankIC={yearly_val_ic[year]:+.4f}")

    gate_history = {d: float(np.mean(v)) for d, v in sorted(gate_accum.items())}
    return ModelRunResult(
        model_name=model_name,
        num_params=n_params,
        oos_scores=oos_scores,
        gate_history=gate_history,
        yearly_val_ic=yearly_val_ic,
        elapsed_seconds=time.time() - t0,
    )
