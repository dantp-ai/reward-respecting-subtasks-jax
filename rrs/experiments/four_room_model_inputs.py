"""Pin the validated Milestone 7 policies used by every Milestone 8 model run."""

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp

from rrs.experiments.four_room_options import OPTION_NAMES, PARAMETERS, SEEDS, STEPS
from rrs.experiments.model_learning import FixedOptions

SNAPSHOT = Path("artifacts/four_room_options/options_step_1000000.json.gz")
SNAPSHOT_SHA256 = "2e800352b546604d862af4fa73ad9786e243b3b5f503a20a991e10580ff7ec40"
OPTION_SEED = 7000


def ensure_snapshot(path=None, figures_dir=Path("figures")):
    """Regenerate only an absent default; never overwrite a supplied/corrupt input."""
    if path is not None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    if not SNAPSHOT.exists():
        print("Regenerating missing canonical Milestone 7 options", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "rrs.experiments.four_room_options",
                "--output-dir",
                str(SNAPSHOT.parent),
                "--figures-dir",
                str(figures_dir),
            ],
            check=True,
        )
    return SNAPSHOT


def load_options(path, data):
    encoded = path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != SNAPSHOT_SHA256:
        raise ValueError(
            "Option snapshot SHA-256 differs from the frozen Milestone 7 input"
        )
    payload = json.loads(gzip.decompress(encoded))
    expected = {
        "schema_version": 1,
        "step": STEPS,
        "seeds": list(SEEDS),
        "option_names": list(OPTION_NAMES),
        "learning_parameters": PARAMETERS._asdict(),
        "feature_positions": [
            list(data.positions[i]) for i in data.nonterminal_indices
        ],
        "policy_positions": [list(p) for p in data.positions],
        "critic_axes": ["run", "option", "feature"],
        "actor_axes": ["run", "option", "action", "feature"],
        "policy_axes": ["run", "option", "state", "action"],
        "stopping_axes": ["run", "option", "state"],
        "main_task_weights": [0.0] * len(data.nonterminal_indices),
        "hallway_bonus": 1.0,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Option snapshot protocol, axes or state/feature order differs"
        )
    run = payload["seeds"].index(OPTION_SEED)
    pi = jnp.array(payload["policy_probabilities"][run])
    beta = jnp.array(payload["stopping"][run])
    n = len(data.positions)
    if pi.shape != (4, n, 4) or beta.shape != (4, n):
        raise ValueError("Option snapshot policy dimensions differ")
    if not bool(
        jnp.isfinite(pi).all()
        & (pi >= 0).all()
        & (jnp.abs(pi.sum(axis=-1) - 1) <= 1e-6).all()
        & ((beta == 0) | (beta == 1)).all()
    ):
        raise ValueError("Invalid frozen policy or stopping probabilities")
    indices = jnp.array(data.nonterminal_indices)
    primitive_pi = jnp.broadcast_to(jnp.eye(4)[:, :, None], (4, 4, len(indices)))
    options = FixedOptions(
        jnp.concatenate((primitive_pi, pi[:, indices].transpose(0, 2, 1))),
        jnp.concatenate(
            (jnp.ones((4, len(indices))), beta[:, indices].astype(jnp.float32))
        ),
    )
    source = {
        "path": str(path),
        "sha256": digest,
        "code_revision": "e24dc66e5f9a490afd1fc3fe681c087478cbb827",
        "seed": OPTION_SEED,
        "steps": STEPS,
        "learning_parameters": PARAMETERS._asdict(),
        "critic_weights": payload["critic_weights"][run],
        "actor_weights": payload["actor_weights"][run],
    }
    return options, source
