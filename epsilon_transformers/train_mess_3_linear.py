"""
Small-scale training run for Linear_Mess3 (or plain Mess3).

IMPORTANT (package resolution):
If `epsilon_transformers` also happens to be `pip install -e`'d from some
other checkout in this same Python environment, a plain
`import epsilon_transformers` could silently load that other copy instead
of this repo's. The sys.path.insert below forces this repo's package to
take priority for this process, regardless of cwd or how the script is
invoked (`python train_mess_3_linear.py`, `python -m ...`, etc).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import epsilon_transformers  # noqa: E402  (import after sys.path fix)
assert Path(epsilon_transformers.__file__).is_relative_to(REPO_ROOT), (
    f"Loaded epsilon_transformers from {epsilon_transformers.__file__}, "
    f"expected it under {REPO_ROOT}. The sys.path fix above did not work "
    f"as intended."
)

from epsilon_transformers.training.configs.model_configs import RawModelConfig
from epsilon_transformers.training.configs.training_configs import (
    LoggingConfig,
    OptimizerConfig,
    PersistanceConfig,
    ProcessDatasetConfig,
    TrainConfig,
    NGramAnalysisConfig,
    MarkovKLAnalysisConfig,
    SimplexAnalysisConfig,
    AnalysisConfig,
)
from epsilon_transformers.training.train import train_model


# ============================================================================
# Model Configuration (small: 1 layer, tiny d_model, for a quick smoke test)
# ============================================================================

model_config = RawModelConfig(
    d_vocab=3,
    d_model=32,
    n_ctx=8,
    d_head=16,
    n_head=2,
    d_mlp=128,
    n_layers=1,
)


# ============================================================================
# Optimizer Configuration
# ============================================================================

optimizer_config = OptimizerConfig(
    optimizer_type="adam",
    learning_rate=1e-3,
    weight_decay=0,
)


# ============================================================================
# Dataset Configuration
# ============================================================================
# Set USE_LINEAR=False to train on plain Mess3 instead of Linear_Mess3.
USE_LINEAR = True

dataset_config = ProcessDatasetConfig(
    process="Linear_Mess3" if USE_LINEAR else "Mess3",
    process_params={"x": 0.05, "a": 0.85},
    batch_size=32,
    num_tokens=20480,       # small scale: ~80 train batches
    sequence_length=model_config.n_ctx,
    test_split=0.05,
    chunk_size=512,
    test_batch_size=256,
    start_state_idx=None,   # let the process compute its own steady state
)


# ============================================================================
# Persistence Configuration
# ============================================================================
run_name = f"smoke_test_{dataset_config.process.lower()}"

persistance_config = PersistanceConfig(
    location="local",
    collection_location=REPO_ROOT / "epsilon_transformers" / "models" / run_name,
    checkpoint_every_n_tokens=5120,  # every 20 train batches
)


# ============================================================================
# Logging Configuration
# ============================================================================
# wandb is off by default for this local smoke test so it can run without a
# network connection / API key. Flip to True (and set WANDB_API_KEY in the
# environment) once you're ready to log a real run.
logging_config = LoggingConfig(
    project_name="epstrans",
    wandb=False,
    run_name=run_name,
    relative_loss=True,
)


# ============================================================================
# Analysis Configuration
# ============================================================================
analysis_config = AnalysisConfig(
    analysis_batch_size=100,
    ngram_analysis=NGramAnalysisConfig(
        enabled=True,
        n_values=[1, 2, 3],
        return_per_position=False,
    ),
    markov_kl_analysis=MarkovKLAnalysisConfig(
        enabled=True,
        return_per_position=False,
    ),
    simplex_analysis=SimplexAnalysisConfig(
        enabled=True,
        hook_point="blocks.0.hook_resid_post",
        num_samples_for_probe=200,
    ),
)


# ============================================================================
# Complete Training Configuration
# ============================================================================
mock_config = TrainConfig(
    model=model_config,
    optimizer=optimizer_config,
    dataset=dataset_config,
    persistance=persistance_config,
    logging=logging_config,
    analysis=analysis_config,
    verbose=True,
    seed=42,
    do_eval=True,
)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    try:
        train_model(mock_config)
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nFix: Either:")
        print("  1. Set wandb=True and wandb_api_key in LoggingConfig")
        print("  2. OR set WANDB_API_KEY environment variable")
        raise
