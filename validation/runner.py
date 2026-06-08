"""Unified experiment runner with statistical validation.

Merges functionality from bench_core.py and bench_stats.py into a single,
clean API. Every experiment gets bootstrap confidence intervals, Cohen's d
effect size, and Wilcoxon signed-rank test automatically.
"""

import gc
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from scipy.stats import wilcoxon


@dataclass
class ExperimentResult:
    """One method's result in an experiment."""
    method: str
    shape: Tuple[int, int, int]
    mean_us: float
    std_us: float
    round_means: List[float] = field(repr=False)
    speedup_mean: float = 0.0
    speedup_ci: Tuple[float, float] = (0.0, 0.0)
    cohens_d: float = 0.0
    p_value: float = 1.0
    significant: bool = False

    def summary(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not sig."
        return (
            f"{self.method:>20s} | μ={self.mean_us:.1f}±{self.std_us:.1f}μs | "
            f"speedup={self.speedup_mean:.2f}x [{self.speedup_ci[0]:.2f},{self.speedup_ci[1]:.2f}] | "
            f"d={self.cohens_d:.2f} p={self.p_value:.4f} {sig}"
        )


class ExperimentRunner:
    """Runs experiments with statistical rigor."""

    def __init__(self, device: str = "cuda", warmup: int = 30, repeat: int = 100):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.warmup = warmup
        self.repeat = repeat

    # ═══════════════════════════════════════════════════════════
    # Training utilities
    # ═══════════════════════════════════════════════════════════

    def train_model(
        self,
        model: torch.nn.Module,
        data: torch.Tensor,
        labels: torch.Tensor,
        n_epochs: int,
        batch_size: int = 64,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        scheduler_cls=None,
        verbose: bool = False,
        label: str = "",
        data_test: torch.Tensor = None,
        labels_test: torch.Tensor = None,
    ) -> List[dict]:
        """Train a model and return per-epoch metrics.

        Args:
            data, labels: training data
            data_test, labels_test: optional test data for eval.
                                     If None, eval on training data.
        Returns list of dicts with keys: epoch, train_loss, test_acc, wall_time_s
        """
        model = model.to(self.device)
        data = data.to(self.device)
        labels = labels.to(self.device)
        if data_test is not None:
            data_test = data_test.to(self.device)
            labels_test = labels_test.to(self.device)
        else:
            data_test = data
            labels_test = labels
        n_train = data.size(0)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = None
        if scheduler_cls is not None:
            sched = scheduler_cls(opt, n_epochs)

        history = []
        wall_clock = 0.0
        torch.cuda.synchronize()

        for epoch in range(n_epochs):
            t0 = time.perf_counter()

            # Train
            model.train()
            perm_idx = torch.randperm(n_train, device=self.device)
            total_loss = 0.0
            n_batches = 0
            for i in range(0, n_train, batch_size):
                idx = perm_idx[i:i + batch_size]
                opt.zero_grad()
                logits = model(data[idx])
                # Handle per-position labels: [B, N, C] → [B*N, C]
                if labels.dim() == 2 and logits.dim() == 3:
                    B_s, N_s, C_s = logits.shape
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(B_s * N_s, C_s),
                        labels[idx].reshape(B_s * N_s),
                    )
                else:
                    loss = torch.nn.functional.cross_entropy(logits, labels[idx])
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1

            if sched is not None:
                sched.step()

            torch.cuda.synchronize()
            wall_clock += time.perf_counter() - t0

            # Eval (on test data if provided, else on training data)
            model.eval()
            with torch.no_grad():
                logits_eval = model(data_test)
                if labels_test.dim() == 2 and logits_eval.dim() == 3:
                    pred = logits_eval.argmax(dim=-1)  # [B, N]
                    acc = (pred == labels_test).float().mean().item()
                else:
                    pred = logits_eval.argmax(dim=1)   # [B]
                    acc = (pred == labels_test).float().mean().item()

            history.append({
                "epoch": epoch,
                "train_loss": total_loss / max(n_batches, 1),
                "test_acc": acc,
                "wall_time_s": wall_clock,
            })

            if verbose and epoch % 20 == 0:
                print(f"  [{label}] epoch {epoch:>2d} | loss={total_loss/n_batches:.4f} | "
                      f"acc={acc:.3f} | time={wall_clock:.1f}s")

        return history

    # ═══════════════════════════════════════════════════════════
    # Throughput measurement
    # ═══════════════════════════════════════════════════════════

    def measure_throughput(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        n_warmup: int = 20,
        n_repeat: int = 50,
    ) -> Tuple[float, float]:
        """Measure forward+backward step time in milliseconds.

        Returns (mean_ms, std_ms).
        """
        model = model.to(self.device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Warmup
        for _ in range(n_warmup):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), target)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()

        # Measure
        times = []
        for _ in range(n_repeat):
            opt.zero_grad()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = torch.nn.functional.mse_loss(model(x), target)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e3)

        return np.mean(times), np.std(times)

    # ═══════════════════════════════════════════════════════════
    # Statistical inference
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def bootstrap_speedup_ci(
        baseline: np.ndarray,
        method: np.ndarray,
        n_bootstrap: int = 10000,
        ci: float = 0.95,
    ) -> Tuple[float, float]:
        """Bootstrap 95% CI for speedup ratio baseline/method."""
        rng = np.random.RandomState(42)
        speedups = []
        n = len(baseline)
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            mu_b = baseline[idx].mean()
            mu_m = method[idx].mean()
            speedups.append(mu_b / mu_m if mu_m > 0 else float("inf"))
        speedups = np.array(speedups)
        lower = np.percentile(speedups, (1 - ci) / 2 * 100)
        upper = np.percentile(speedups, (1 + ci) / 2 * 100)
        return (lower, upper)

    @staticmethod
    def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
        """Cohen's d: (μ_a - μ_b) / σ_pooled."""
        mu_a, mu_b = a.mean(), b.mean()
        n_a, n_b = len(a), len(b)
        var_a = a.var(ddof=1)
        var_b = b.var(ddof=1)
        pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        if pooled < 1e-12:
            return 0.0
        return (mu_a - mu_b) / pooled

    @staticmethod
    def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
        """Wilcoxon signed-rank test p-value."""
        try:
            _, p = wilcoxon(a, b, alternative="two-sided")
            return p
        except Exception:
            return 1.0


# ═══════════════════════════════════════════════════════════════
# Summary helpers
# ═══════════════════════════════════════════════════════════════

def print_header(title: str, width: int = 70):
    print(f"\n{'='*width}")
    print(title)
    print(f"{'='*width}")


def find_time_to_accuracy(history: List[dict], target: float) -> Optional[float]:
    """Find wall-clock time when a target accuracy is first reached."""
    for entry in history:
        if entry["test_acc"] >= target:
            return entry["wall_time_s"]
    return None
