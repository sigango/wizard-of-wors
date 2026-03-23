#!/usr/bin/env python3
"""
Quick runtime estimator for Wizard of Wor PPO (object-centric).

Runs:
1) A short warmup run (to trigger JAX compilation)
2) A timed benchmark run (default ~1M timesteps)
3) Prints steps/sec and ETA for a full run (default 200M timesteps)

Run from repository root, e.g.:
  XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=metal PYTHONPATH=src \
  python scripts/benchmarks/benchmark_wizard_ppo_object_eta.py
"""

from __future__ import annotations

import argparse
import copy
import math
import time

import jax

from train_jaxatari_agent import train_ppo_with_jaxatari


def format_seconds(total_seconds: float) -> str:
    total_seconds = int(round(total_seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_object_config(
    num_envs: int,
    num_steps: int,
    num_minibatches: int,
    update_epochs: int,
) -> dict:
    return {
        "ENV_NAME_JAXATARI": "wizardofwor",
        "ENV_TYPE": "jax",
        "TOTAL_TIMESTEPS": 20_000_000,
        "LR": 3e-4,
        "NUM_ENVS": num_envs,
        "NUM_STEPS": num_steps,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "NUM_MINIBATCHES": num_minibatches,
        "UPDATE_EPOCHS": update_epochs,
        "CLIP_EPS": 0.1,
        "CLIP_VF_EPS": 0.1,
        "ENT_COEF": 0.005,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu",
        "ANNEAL_LR": True,
        "SEED": 42,
        "BUFFER_WINDOW": 4,
        "FRAMESKIP": 4,
        "REPEAT_ACTION_PROBABILITY": 0.25,
        "JAX_OBS_MODE": "object",
        "LOG_INTERVAL_UPDATES": 1000,
        "VISUALIZE_AFTER_TRAINING": False,
        "SAVE_VIZ_VIDEO": False,
        "USE_WANDB": False,
        "WANDB_MODE": "disabled",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-steps", type=int, default=1_000_000)
    parser.add_argument("--full-steps", type=int, default=200_000_000)
    parser.add_argument("--warmup-updates", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=3)
    args = parser.parse_args()

    config = build_object_config(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
    )
    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]

    warmup_steps = steps_per_update * max(1, args.warmup_updates)
    bench_updates = max(1, math.ceil(args.benchmark_steps / steps_per_update))
    bench_steps = bench_updates * steps_per_update

    print("=== Backend Check ===")
    print(f"JAX version: {jax.__version__}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print()

    print("=== Warmup Run (compile) ===")
    warm_cfg = copy.deepcopy(config)
    warm_cfg["TOTAL_TIMESTEPS"] = warmup_steps
    print(f"Warmup timesteps: {warmup_steps:,}")
    train_ppo_with_jaxatari(warm_cfg)
    print("Warmup done.")
    print()

    print("=== Timed Benchmark Run ===")
    bench_cfg = copy.deepcopy(config)
    bench_cfg["TOTAL_TIMESTEPS"] = bench_steps
    print(f"Benchmark timesteps: {bench_steps:,} ({bench_updates} updates)")

    t0 = time.time()
    train_ppo_with_jaxatari(bench_cfg)
    elapsed = time.time() - t0

    sps = bench_steps / elapsed
    eta_seconds = args.full_steps / sps

    print()
    print("=== Results ===")
    print(f"Elapsed: {elapsed:.2f} s")
    print(f"Throughput: {sps:,.2f} steps/s")
    print(f"Estimated time for {args.full_steps:,} steps: {eta_seconds/3600:.2f} h ({format_seconds(eta_seconds)})")


if __name__ == "__main__":
    main()
