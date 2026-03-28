#!/usr/bin/env python3
"""Generate a self-contained Wizard of Wor PPO report bundle.

This script consumes run metadata produced by run_report_wizard.sh and creates:
- a bundle directory with copied run artifacts,
- summary CSV tables (metrics + runtime),
- a markdown report with required sections.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

BASELINE_REFERENCE = {
    "title": "Fast and Data-Efficient Training of Rainbow: an Experimental Study on Atari",
    "authors": "Dominik Schmidt, Thomas Schmied",
    "year": "2021",
    "doi": "10.48550/arXiv.2111.10247",
    "url": "https://arxiv.org/pdf/2111.10247.pdf",
    "table_note": "Table 2 (53 Atari games, average over 3 seeds).",
    # Row extracted from the PDF text (WizardOfWor line).
    "wizard_row": [563.5, 4756.5, 2727.9, 3393.3, 7878.6, 17862.5, 15518.6],
}


@dataclass
class RunRecord:
    preset: str
    obs_mode: str
    source_result_dir: Path
    source_model_path: Path
    copied_result_dir: Path


@dataclass
class RunSummary:
    preset: str
    obs_mode: str
    run_dir_rel: str
    model_rel: Optional[str]
    final_timestep: Optional[float]
    final_mean_reward: Optional[float]
    eval_tag: Optional[str]
    eval_mean_reward: Optional[float]
    eval_std_reward: Optional[float]
    eval_num_episodes: Optional[int]
    total_timesteps_cfg: Optional[int]
    num_updates_cfg: Optional[int]
    est_steps_per_sec: Optional[float]
    compile_sec: Optional[float]
    runtime_sec: Optional[float]
    total_wall_sec: Optional[float]
    train_stage_sec: Optional[float]
    eval_stage_sec: Optional[float]
    visualize_stage_sec: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Wizard of Wor PPO report bundle")
    parser.add_argument("--manifest-csv", required=True, help="Path to wizard_run_manifest_*.csv")
    parser.add_argument("--timings-csv", required=True, help="Path to wizard_stage_timing_*.csv")
    parser.add_argument("--output-dir", required=True, help="Output report bundle directory")
    return parser.parse_args()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _latest_file(files: List[Path]) -> Optional[Path]:
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _copy_run_dirs(manifest_rows: List[Dict[str, str]], output_dir: Path) -> List[RunRecord]:
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    records: List[RunRecord] = []
    used_names: Dict[str, int] = {}

    for row in manifest_rows:
        preset = row.get("preset", "unknown_preset")
        obs_mode = row.get("obs_mode", "unknown")
        src_result_dir = Path(row.get("result_dir", "")).expanduser()
        src_model = Path(row.get("model_path", "")).expanduser()

        if not src_result_dir.exists():
            print(f"[WARN] Missing result_dir, skip: {src_result_dir}")
            continue

        base_name = f"{obs_mode}_{src_result_dir.name}"
        if base_name in used_names:
            used_names[base_name] += 1
            base_name = f"{base_name}_{used_names[base_name]}"
        else:
            used_names[base_name] = 1

        dst_result_dir = runs_dir / base_name
        if dst_result_dir.exists():
            shutil.rmtree(dst_result_dir)
        shutil.copytree(src_result_dir, dst_result_dir)

        records.append(
            RunRecord(
                preset=preset,
                obs_mode=obs_mode,
                source_result_dir=src_result_dir.resolve(),
                source_model_path=src_model.resolve() if src_model.exists() else src_model,
                copied_result_dir=dst_result_dir.resolve(),
            )
        )

    return records


def _collect_stage_timings(
    timing_rows: List[Dict[str, str]], run_records: List[RunRecord]
) -> Dict[str, Dict[str, float]]:
    by_result_dir: Dict[str, Dict[str, float]] = {}
    valid_keys = {str(r.source_result_dir): r for r in run_records}

    for row in timing_rows:
        result_dir = row.get("result_dir", "")
        stage = row.get("stage", "")
        duration = _safe_float(row.get("duration_seconds"))
        if not result_dir or not stage or duration is None:
            continue
        if result_dir not in valid_keys:
            # tolerate path string mismatches due to symlinks; compare resolved paths
            try:
                resolved = str(Path(result_dir).expanduser().resolve())
            except Exception:
                resolved = result_dir
            if resolved not in valid_keys:
                continue
            result_dir = resolved
        by_result_dir.setdefault(result_dir, {})[stage] = duration

    return by_result_dir


def _read_metrics_csv(run_dir: Path) -> Dict[str, Optional[float]]:
    metrics_path = run_dir / "training_metrics_ppo_distrax.csv"
    if not metrics_path.exists():
        return {"final_timestep": None, "final_mean_reward": None}

    rows = _read_csv(metrics_path)
    if not rows:
        return {"final_timestep": None, "final_mean_reward": None}

    last_row = rows[-1]
    return {
        "final_timestep": _safe_float(last_row.get("timesteps")),
        "final_mean_reward": _safe_float(last_row.get("mean_rewards")),
    }


def _read_eval_summary(run_dir: Path) -> Dict[str, Optional[float]]:
    candidates = sorted(run_dir.glob("*_summary_*.json"))
    if not candidates:
        return {
            "eval_tag": None,
            "eval_mean_reward": None,
            "eval_std_reward": None,
            "eval_num_episodes": None,
        }

    preferred = [
        p
        for p in candidates
        if p.name.startswith("eval_from_checkpoint_summary")
        or p.name.startswith("post_train_eval_summary")
    ]
    chosen = _latest_file(preferred) or _latest_file(candidates)
    if chosen is None:
        return {
            "eval_tag": None,
            "eval_mean_reward": None,
            "eval_std_reward": None,
            "eval_num_episodes": None,
        }

    try:
        payload = json.loads(chosen.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "eval_tag": None,
            "eval_mean_reward": None,
            "eval_std_reward": None,
            "eval_num_episodes": None,
        }

    return {
        "eval_tag": str(payload.get("tag", "")) or None,
        "eval_mean_reward": _safe_float(str(payload.get("mean_reward", ""))),
        "eval_std_reward": _safe_float(str(payload.get("std_reward", ""))),
        "eval_num_episodes": _safe_int(str(payload.get("num_episodes", ""))),
    }


def _read_timing_summary(run_dir: Path) -> Dict[str, Optional[float]]:
    json_path = run_dir / "training_timing_summary.json"
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        return {
            "total_timesteps": _safe_int(str(payload.get("total_timesteps", ""))),
            "num_updates": _safe_int(str(payload.get("num_updates", ""))),
            "estimated_steps_per_sec": _safe_float(str(payload.get("estimated_steps_per_sec", ""))),
            "estimated_compile_sec": _safe_float(str(payload.get("estimated_compile_sec", ""))),
            "estimated_runtime_sec": _safe_float(str(payload.get("estimated_runtime_sec", ""))),
            "total_wall_time_sec": _safe_float(str(payload.get("total_wall_time_sec", ""))),
        }

    csv_path = run_dir / "training_timing_summary.csv"
    rows = _read_csv(csv_path)
    if not rows:
        return {
            "total_timesteps": None,
            "num_updates": None,
            "estimated_steps_per_sec": None,
            "estimated_compile_sec": None,
            "estimated_runtime_sec": None,
            "total_wall_time_sec": None,
        }

    row = rows[-1]
    return {
        "total_timesteps": _safe_int(row.get("total_timesteps")),
        "num_updates": _safe_int(row.get("num_updates")),
        "estimated_steps_per_sec": _safe_float(row.get("estimated_steps_per_sec")),
        "estimated_compile_sec": _safe_float(row.get("estimated_compile_sec")),
        "estimated_runtime_sec": _safe_float(row.get("estimated_runtime_sec")),
        "total_wall_time_sec": _safe_float(row.get("total_wall_time_sec")),
    }


def _relative_to_bundle(path: Path, bundle_dir: Path) -> str:
    try:
        return str(path.relative_to(bundle_dir))
    except ValueError:
        return str(path)


def _write_summary_csv(path: Path, rows: List[RunSummary]) -> None:
    fieldnames = [
        "preset",
        "obs_mode",
        "run_dir_rel",
        "model_rel",
        "final_timestep",
        "final_mean_reward",
        "eval_tag",
        "eval_mean_reward",
        "eval_std_reward",
        "eval_num_episodes",
        "total_timesteps_cfg",
        "num_updates_cfg",
        "est_steps_per_sec",
        "compile_sec",
        "runtime_sec",
        "total_wall_sec",
        "train_stage_sec",
        "eval_stage_sec",
        "visualize_stage_sec",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _write_runtime_csv(path: Path, rows: List[RunSummary]) -> None:
    fieldnames = [
        "obs_mode",
        "preset",
        "compile_sec",
        "runtime_sec",
        "total_wall_sec",
        "train_stage_sec",
        "eval_stage_sec",
        "visualize_stage_sec",
        "total_timesteps_cfg",
        "est_steps_per_sec",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "obs_mode": row.obs_mode,
                    "preset": row.preset,
                    "compile_sec": row.compile_sec,
                    "runtime_sec": row.runtime_sec,
                    "total_wall_sec": row.total_wall_sec,
                    "train_stage_sec": row.train_stage_sec,
                    "eval_stage_sec": row.eval_stage_sec,
                    "visualize_stage_sec": row.visualize_stage_sec,
                    "total_timesteps_cfg": row.total_timesteps_cfg,
                    "est_steps_per_sec": row.est_steps_per_sec,
                }
            )


def _collect_figures(run_dir: Path) -> Dict[str, List[Path]]:
    plots = sorted(run_dir.glob("training_plots_*.png"))
    videos = sorted(run_dir.glob("agent_visualization_*.mp4"))
    metric_csv = sorted(run_dir.glob("training_metrics_*.csv"))
    summaries = sorted(run_dir.glob("*_summary_*.json"))
    return {
        "plots": plots,
        "videos": videos,
        "metric_csv": metric_csv,
        "summaries": summaries,
    }


def _build_markdown(bundle_dir: Path, run_summaries: List[RunSummary], run_records: List[RunRecord]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    lines.append("# Wizard of Wor PPO Report Bundle")
    lines.append("")
    lines.append(f"Generated at: **{now}**")
    lines.append("")
    lines.append("## 1) Scope and Script Workflow")
    lines.append("")
    lines.append("This bundle is generated from `scripts/benchmarks/run_report_wizard.sh` for **PPO only** with **Object + Pixel** observations (no mods pipeline).")
    lines.append("")
    lines.append("Pipeline summary:")
    lines.append("1. Train PPO for each selected preset (Object, Pixel).")
    lines.append("2. Save checkpoint + training metrics (`npz/csv`) + training plot (`png`).")
    lines.append("3. Run evaluation from checkpoint and save summary (`json/csv`).")
    lines.append("4. Optionally run visualization and save `mp4` video.")
    lines.append("5. Copy all run artifacts into this bundle and generate this markdown report.")
    lines.append("")

    lines.append("## 2) Run Inventory")
    lines.append("")
    lines.append("| obs_mode | preset | run_dir | model |")
    lines.append("|---|---|---|---|")
    if run_summaries:
        for row in run_summaries:
            model = row.model_rel if row.model_rel is not None else "N/A"
            lines.append(f"| {row.obs_mode} | {row.preset} | `{row.run_dir_rel}` | `{model}` |")
    else:
        lines.append("| N/A | N/A | N/A | N/A |")
    lines.append("")

    lines.append("## 3) Baseline Reference (Paper)")
    lines.append("")
    lines.append(
        f"Reference: {BASELINE_REFERENCE['authors']} ({BASELINE_REFERENCE['year']}), "
        f"*{BASELINE_REFERENCE['title']}*, DOI: {BASELINE_REFERENCE['doi']}"
    )
    lines.append("")
    lines.append(f"- PDF: {BASELINE_REFERENCE['url']}")
    lines.append(f"- Note: {BASELINE_REFERENCE['table_note']}")
    lines.append("- Extracted WizardOfWor row from the paper table:")
    row_vals = BASELINE_REFERENCE["wizard_row"]
    lines.append(
        f"  `WizardOfWor {row_vals[0]} {row_vals[1]} {row_vals[2]} {row_vals[3]} {row_vals[4]} {row_vals[5]} {row_vals[6]}`"
    )
    lines.append("")
    lines.append("Comparison anchors used in this report:")
    lines.append(f"- Random baseline anchor: **{row_vals[0]}**")
    lines.append(f"- Human baseline anchor: **{row_vals[1]}**")
    lines.append("")

    lines.append("## 4) Training and Evaluation Results")
    lines.append("")
    lines.append("| obs_mode | final_timestep | final_train_mean_reward | eval_tag | eval_mean ± std | eval_episodes |")
    lines.append("|---|---:|---:|---|---:|---:|")
    if run_summaries:
        for row in run_summaries:
            eval_pair = (
                f"{_fmt(row.eval_mean_reward, 3)} +/- {_fmt(row.eval_std_reward, 3)}"
                if row.eval_mean_reward is not None
                else "N/A"
            )
            lines.append(
                f"| {row.obs_mode} | {_fmt(row.final_timestep, 0)} | {_fmt(row.final_mean_reward, 3)} | "
                f"{row.eval_tag or 'N/A'} | {eval_pair} | {row.eval_num_episodes if row.eval_num_episodes is not None else 'N/A'} |"
            )
    else:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A |")
    lines.append("")

    lines.append("## 5) Compile/Runtime Table")
    lines.append("")
    lines.append("| obs_mode | compile_sec_est | runtime_sec_est | train_wall_sec | train_stage_sec | eval_stage_sec | visualize_stage_sec | total_timesteps | steps_per_sec_est |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    if run_summaries:
        for row in run_summaries:
            lines.append(
                f"| {row.obs_mode} | {_fmt(row.compile_sec, 3)} | {_fmt(row.runtime_sec, 3)} | {_fmt(row.total_wall_sec, 3)} | "
                f"{_fmt(row.train_stage_sec, 3)} | {_fmt(row.eval_stage_sec, 3)} | {_fmt(row.visualize_stage_sec, 3)} | "
                f"{row.total_timesteps_cfg if row.total_timesteps_cfg is not None else 'N/A'} | {_fmt(row.est_steps_per_sec, 3)} |"
            )
    else:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    lines.append("")
    lines.append("Raw runtime CSV: `compile_runtime_table.csv`")
    lines.append("")

    lines.append("## 6) Figures and Artifacts")
    lines.append("")
    if run_records:
        for record in run_records:
            run_dir = record.copied_result_dir
            rel_run_dir = _relative_to_bundle(run_dir, bundle_dir)
            figure_assets = _collect_figures(run_dir)
            lines.append(f"### {record.obs_mode.upper()} ({record.preset})")
            lines.append("")
            lines.append(f"Artifacts folder: `{rel_run_dir}`")
            lines.append("")

            for plot in figure_assets["plots"]:
                rel_plot = _relative_to_bundle(plot, bundle_dir)
                lines.append(f"- Training plot: `{rel_plot}`")
                lines.append(f"  ![Training plot {record.obs_mode}]({rel_plot})")

            if figure_assets["videos"]:
                for video in figure_assets["videos"]:
                    rel_video = _relative_to_bundle(video, bundle_dir)
                    lines.append(f"- Evaluation/visualization video: `{rel_video}`")
            else:
                lines.append("- Evaluation/visualization video: not found in this run folder.")

            for metric_csv in figure_assets["metric_csv"]:
                rel_metric = _relative_to_bundle(metric_csv, bundle_dir)
                lines.append(f"- Metrics CSV: `{rel_metric}`")

            for summary in figure_assets["summaries"]:
                rel_summary = _relative_to_bundle(summary, bundle_dir)
                lines.append(f"- Eval summary JSON: `{rel_summary}`")

            lines.append("")
    else:
        lines.append("No runs were copied, so no figures are available.")
        lines.append("")

    lines.append("## 7) Repro Command")
    lines.append("")
    lines.append("```bash")
    lines.append("bash scripts/benchmarks/run_report_wizard.sh --algo ppo --obs both --with-visualize")
    lines.append("```")
    lines.append("")
    lines.append("## 8) Notes")
    lines.append("")
    lines.append("- This report intentionally excludes the mods pipeline.")
    lines.append("- If visualization is disabled, MP4 files will be absent by design.")
    lines.append("- Compile time is estimated from first-update overhead; values are approximate.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    manifest_csv = Path(args.manifest_csv).expanduser().resolve()
    timings_csv = Path(args.timings_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_csv(manifest_csv)
    timing_rows = _read_csv(timings_csv)

    run_records = _copy_run_dirs(manifest_rows, output_dir)
    stage_timings = _collect_stage_timings(timing_rows, run_records)

    run_summaries: List[RunSummary] = []

    for record in run_records:
        run_dir = record.copied_result_dir
        metrics = _read_metrics_csv(run_dir)
        eval_summary = _read_eval_summary(run_dir)
        timing_summary = _read_timing_summary(run_dir)
        stage_summary = stage_timings.get(str(record.source_result_dir), {})

        model_path = run_dir / "ppo_distrax_model_params.npz"
        model_rel = _relative_to_bundle(model_path, output_dir) if model_path.exists() else None

        run_summaries.append(
            RunSummary(
                preset=record.preset,
                obs_mode=record.obs_mode,
                run_dir_rel=_relative_to_bundle(run_dir, output_dir),
                model_rel=model_rel,
                final_timestep=metrics.get("final_timestep"),
                final_mean_reward=metrics.get("final_mean_reward"),
                eval_tag=eval_summary.get("eval_tag"),
                eval_mean_reward=eval_summary.get("eval_mean_reward"),
                eval_std_reward=eval_summary.get("eval_std_reward"),
                eval_num_episodes=eval_summary.get("eval_num_episodes"),
                total_timesteps_cfg=timing_summary.get("total_timesteps"),
                num_updates_cfg=timing_summary.get("num_updates"),
                est_steps_per_sec=timing_summary.get("estimated_steps_per_sec"),
                compile_sec=timing_summary.get("estimated_compile_sec"),
                runtime_sec=timing_summary.get("estimated_runtime_sec"),
                total_wall_sec=timing_summary.get("total_wall_time_sec"),
                train_stage_sec=stage_summary.get("train"),
                eval_stage_sec=stage_summary.get("eval"),
                visualize_stage_sec=stage_summary.get("visualize"),
            )
        )

    # Keep rows stable for readability.
    run_summaries.sort(key=lambda x: (x.obs_mode, x.preset, x.run_dir_rel))

    manifest_snapshot = output_dir / "run_manifest_snapshot.csv"
    timings_snapshot = output_dir / "stage_timings_snapshot.csv"
    shutil.copy2(manifest_csv, manifest_snapshot)
    shutil.copy2(timings_csv, timings_snapshot)

    summary_csv = output_dir / "training_eval_summary.csv"
    runtime_csv = output_dir / "compile_runtime_table.csv"
    _write_summary_csv(summary_csv, run_summaries)
    _write_runtime_csv(runtime_csv, run_summaries)

    report_md = output_dir / "wizardofwor_report.md"
    report_md.write_text(_build_markdown(output_dir, run_summaries, run_records), encoding="utf-8")

    print(f"Report markdown: {report_md}")
    print(f"Run copies: {output_dir / 'runs'}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Runtime CSV: {runtime_csv}")


if __name__ == "__main__":
    main()
