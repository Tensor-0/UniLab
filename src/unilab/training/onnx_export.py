"""ONNX export and onnxruntime numeric verification helpers for play entrypoints."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


class ObsManifestUnavailableError(RuntimeError):
    """The env cannot describe its policy observation layout (shape, not contract).

    Raised when the env exposes no ``observation_manager`` (lightweight fakes, non
    manager-based envs) or when the policy observation group cannot be identified.
    Callers treat this as "manifest skipped", not as a broken artifact — see
    ``write_obs_manifest`` for why the two cases are kept apart.
    """


def _resolve_policy_obs_layout(env: Any) -> tuple[str, list[tuple[str, int]]]:
    """Return ``(group_name, [(term_name, dim), ...])`` for the policy observation group.

    The term order is the YAML declaration order (``ObservationManager`` appends in
    ``cfg.terms`` order, see ``managers/observation_manager.py``), and each length is
    the flattened per-term slice width including any history stacking — i.e. exactly
    the widths used when the group is concatenated.
    """
    manager = None
    tried: list[str] = []
    for candidate in (
        env,
        getattr(env, "unwrapped", None),
        getattr(env, "wrapped_env", None),
        getattr(env, "env", None),
    ):
        if candidate is None:
            continue
        tried.append(type(candidate).__name__)
        manager = getattr(candidate, "observation_manager", None)
        if manager is not None:
            break
    if manager is None:
        raise ObsManifestUnavailableError(
            "write_obs_manifest needs an env exposing `observation_manager`; tried "
            f"{tried}. Pass the manager-based env itself."
        )

    cfg = getattr(env, "_cfg", None)
    if cfg is None:
        cfg = getattr(env, "cfg", None)
    group = getattr(cfg, "policy_observation_group", None)
    if group is None:
        # Fall back to whatever group the manager actually has, but only when unambiguous.
        groups = list(manager.active_terms)
        if len(groups) != 1:
            raise ObsManifestUnavailableError(
                "env config has no `policy_observation_group` and the manager exposes "
                f"{groups}; cannot pick the policy group unambiguously."
            )
        group = groups[0]

    names = list(manager.active_terms[group])
    dims = [int(np.prod(d)) for d in manager.group_obs_term_dim[group]]
    if len(names) != len(dims):
        raise ValueError(
            f"Observation group '{group}' reports {len(names)} term names but {len(dims)} dims."
        )
    return group, list(zip(names, dims, strict=True))


def _onnx_input_total(onnx_path: str) -> tuple[str, list[int], int]:
    """Return ``(input_name, shape, element_count)`` of a single-input ONNX graph."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"{onnx_path} has {len(inputs)} inputs; expected a single-input policy.")
    shape = [int(d) if isinstance(d, int) and d > 0 else 1 for d in inputs[0].shape]
    total = 1
    for d in shape:
        total *= d
    return inputs[0].name, shape, total


def write_obs_manifest(
    onnx_path: str | None,
    env: Any,
    *,
    task_name: str | None = None,
    algo: str | None = None,
    root_dir: str | Path | None = None,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write ``obs_manifest.json`` describing the policy input vector, next to the ONNX graph.

    Why this exists: ``run_config.json`` snapshots ``env.observations`` (term order) but
    **not the per-term dimensions**, so a deploy-side layout check cannot verify widths —
    only order. The manifest closes that gap, and gives the deploy side one machine
    comparable string (``signature``) to diff against its own ``obs_layouts``.

    The layout is read back from the *built* env rather than re-derived from config, so it
    reflects what the policy is actually fed.

    Args:
        onnx_path: Exported policy graph to cross-check and write beside. Pass ``None`` to
            snapshot a task *before* training (then ``out_path`` is required).
        env: Built environment exposing ``observation_manager``.
        task_name: Registry task name, for provenance.
        algo: Algorithm name (``flashsac``, ``sac``, ...), for provenance.
        root_dir: Git repo root used to record the producing commit; defaults to cwd.
        out_path: Explicit manifest destination; defaults to ``obs_manifest.json`` beside
            ``onnx_path``.

    Returns:
        The manifest payload as written.

    Raises:
        ValueError: if the env's obs layout disagrees with the ONNX input element count —
            the exported graph and the environment are not the same contract. Callers must
            NOT swallow this: it means the artifact itself is inconsistent.
        ObsManifestUnavailableError: if the env cannot describe a layout at all. Callers may
            downgrade this to a loud warning — it says nothing about the artifact.
    """
    group, terms = _resolve_policy_obs_layout(env)
    obs_dim = sum(d for _, d in terms)

    if onnx_path is not None:
        input_name, input_shape, onnx_total = _onnx_input_total(onnx_path)
        if obs_dim != onnx_total:
            raise ValueError(
                f"Observation layout disagrees with the ONNX graph: group '{group}' totals "
                f"{obs_dim} values ({[f'{n}:{d}' for n, d in terms]}) but "
                f"{Path(onnx_path).name} expects {onnx_total} ({input_name}{input_shape}). "
                "The exported policy and the environment are not the same contract — fix "
                "before shipping this artifact."
            )
        onnx_info = {
            "file": Path(onnx_path).name,
            "input_name": input_name,
            "input_shape": input_shape,
        }
    else:
        onnx_info = None

    # Secondary check: the env's own group->dim spec must agree with the term sum.
    spec = getattr(env, "obs_groups_spec", None)
    if isinstance(spec, dict) and "obs" in spec and int(spec["obs"]) != obs_dim:
        raise ValueError(
            f"obs_groups_spec['obs']={int(spec['obs'])} but group '{group}' sums to {obs_dim}."
        )

    git_info: dict[str, Any] = {}
    try:
        from unilab.training.experiment import get_git_info

        git_info = get_git_info(root_dir or Path.cwd())
    except Exception:  # pragma: no cover - provenance is best-effort
        git_info = {}

    payload: dict[str, Any] = {
        "schema": "unilab-obs-manifest/1",
        "created_utc": datetime.now(UTC).isoformat(),
        "task_name": task_name,
        "algo": algo,
        "git_commit": git_info.get("commit"),
        "git_branch": git_info.get("branch"),
        "git_dirty": git_info.get("dirty"),
        "policy_observation_group": group,
        "obs_dim": obs_dim,
        "terms": [{"name": n, "dim": d} for n, d in terms],
        "signature": "|".join(f"{n}:{d}" for n, d in terms),
        "onnx": onnx_info,
    }

    if out_path is None:
        if onnx_path is None:
            raise ValueError("write_obs_manifest needs either onnx_path or out_path.")
        out_path = Path(onnx_path).with_name("obs_manifest.json")
    manifest_path = Path(out_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote observation manifest to {manifest_path}\n"
        f"  group={group}  obs_dim={obs_dim}  terms={len(terms)}\n"
        f"  signature={payload['signature']}"
    )
    return payload


def export_policy_onnx(
    export_module: torch.nn.Module,
    onnx_path: str,
    export_inputs: tuple[torch.Tensor, ...],
    *,
    input_names: list[str],
    output_names: list[str] | None = None,
    opset_version: int = 17,
) -> None:
    """Export ``export_module`` to ``onnx_path`` and print the artifact path.

    Args:
        export_module: Module traced by ``torch.onnx.export``.
        onnx_path: Destination file path for the exported graph.
        export_inputs: Positional example inputs matching ``input_names``.
        input_names: ONNX input names, aligned positionally with ``export_inputs``.
        output_names: ONNX output names; defaults to ``["action"]``.
        opset_version: ONNX opset version; defaults to 17.
    """
    if output_names is None:
        output_names = ["action"]
    with torch.inference_mode():
        torch.onnx.export(
            export_module,
            export_inputs,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
        )
    print(f"Exported actor ONNX to {onnx_path}")


def verify_policy_onnx(
    export_module: torch.nn.Module,
    onnx_path: str,
    verify_inputs: tuple[torch.Tensor, ...],
    *,
    input_names: list[str],
    max_diff_tol: float = 1e-4,
) -> tuple[float, float]:
    """Compare PyTorch and ONNX Runtime outputs on identical inputs.

    Runs ``export_module`` and the exported graph at ``onnx_path`` on
    ``verify_inputs``, prints the max/mean absolute difference, and prints a
    warning when the max difference exceeds ``max_diff_tol``.

    Args:
        export_module: Module that was exported to ``onnx_path``. Tuple outputs
            are compared on their first element.
        onnx_path: Exported ONNX graph to verify.
        verify_inputs: Positional inputs matching ``input_names``.
        input_names: ONNX input names, aligned positionally with ``verify_inputs``.
        max_diff_tol: Absolute max-difference tolerance before warning.

    Returns:
        ``(max_diff, mean_diff)`` between PyTorch and ONNX Runtime outputs.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        pt_output = export_module(*verify_inputs)
        if isinstance(pt_output, tuple):
            pt_output = pt_output[0]
        pt_np = pt_output.cpu().numpy()
    onnx_inputs = {
        name: value.cpu().numpy().astype(np.float32)
        for name, value in zip(input_names, verify_inputs, strict=True)
    }
    onnx_output = sess.run(None, onnx_inputs)[0]
    max_diff = float(np.max(np.abs(pt_np - onnx_output)))
    mean_diff = float(np.mean(np.abs(pt_np - onnx_output)))
    print(f"ONNX vs PyTorch — max_diff: {max_diff:.2e}, mean_diff: {mean_diff:.2e}")
    if max_diff > max_diff_tol:
        print("WARNING: ONNX output diverges from PyTorch!")
    else:
        print("ONNX export verified OK.")
    return max_diff, mean_diff
