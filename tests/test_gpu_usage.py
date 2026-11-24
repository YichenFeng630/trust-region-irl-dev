"""Tests for trust_region_irl.gpu_usage: GPU recording with nvidia-smi around a block of code.

    python -m pytest tests/test_gpu_usage.py -q
"""
import shutil

import jax
import jax.numpy as jnp
import pytest

from trust_region_irl.gpu_usage import GpuUsage


@pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="needs nvidia-smi")
def test_gpu_usage(tmp_path):
    x = jnp.ones((2048, 2048))
    with GpuUsage(tmp_path / "gpu.csv", interval_ms=100) as gpu:
        for _ in range(30):
            x = jnp.tanh(x @ x / 2048)
        jax.block_until_ready(x)
    s = gpu.summary
    for k in ("gpu_name", "gpu_driver", "wall_s", "samples", "utilization_mean_pct", "gpu_memory_used_peak_mib", "process_memory_mib", "power_mean_w", "energy_wh"):
        assert k in s, k
    assert s["samples"] >= 1 and s["energy_wh"] >= 0
    assert len(open(tmp_path / "gpu.csv").read().splitlines()) >= 2
