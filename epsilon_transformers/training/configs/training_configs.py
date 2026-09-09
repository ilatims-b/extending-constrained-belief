from typing import Literal, Optional, List as PyList, Dict, Any,Union
from pydantic import BaseModel, field_validator, model_validator
import pathlib
import torch
from torch.utils.data import DataLoader
import wandb
import os
import dotenv
import math
from dataclasses import dataclass, asdict, field

from epsilon_transformers.persistence import Persister
from epsilon_transformers.process.processes import PROCESS_REGISTRY
from epsilon_transformers.process.dataset import (
    ProcessDataset,
    process_dataset_collate_fn,
)
from epsilon_transformers.training.configs.base_config import Config
from epsilon_transformers.training.configs.model_configs import RawModelConfig


# ============================================================================
# EXISTING CONFIGS (DO NOT MODIFY please)
# ============================================================================

Optimizer = torch.optim.Adam | torch.optim.SGD | torch.optim.AdamW


class OptimizerConfig(Config):
    optimizer_type: Literal["sgd", "adam","adamw"]
    learning_rate: float
    weight_decay: float

    def from_model(self, model: torch.nn.Module, device: torch.device) -> Optimizer:
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW    
        else:
            raise ValueError(
                f"{self.optimizer_type} is not a valid optimizer_type. Must be 'adam' or 'sgd' or adamw"
            )
        return optimizer(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )


class PersistanceConfig(Config):
    location: Literal["local"]
    collection_location: pathlib.Path | str
    checkpoint_every_n_tokens: int

    def init(self) -> Persister:
        save_dir = str(self.collection_location)
        return Persister(save_dir=save_dir) 

@dataclass
class TruncationConfig:
    enabled: bool = False
    keep_lengths: Optional[list[int]] = None



class ProcessDatasetConfig(Config):
    """Dataset configuration."""
    process: str
    process_params: dict[str, Any]
    batch_size: int
    sequence_length: int
    num_tokens: int
    test_split: float
    test_batch_size: Optional[int]=None
    gpu_generation:bool=True
    chunk_size: int=2048
    start_state_idx: Optional[int]=None
    steady_state: Optional[list[float]]=None
    truncation: Optional[TruncationConfig] = None

    vocab_map:Optional[Union[list[int],dict[int,int]]]=None


    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if v <= 0:
            raise ValueError("batch_size must be > 0")
        return v
 
    def to_dataloader(self, sequence_length: int, train: bool,device:Optional[torch.device]=None,suffix_eval: bool = False) -> DataLoader:
        """Create dataloader from config."""
        # Use sequence_length from config by default
        seq_len = sequence_length
        total_tokens_target = (
            self.num_tokens
            if train
            else math.ceil(self.num_tokens * self.test_split)
            )
        num_samples=total_tokens_target // seq_len
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        
        if train:
            current_batch_size=self.batch_size
            current_chunk_size=self.chunk_size
        else:
            current_batch_size=self.test_batch_size if self.test_batch_size is not None else self.batch_size 
            current_chunk_size=self.test_batch_size if self.test_batch_size is not None else self.chunk_size
        
        enable_truncation = False
        truncation_choices = None

        if suffix_eval:
            enable_truncation = True
            truncation_choices = (
                self.truncation.keep_lengths
                if self.truncation and self.truncation.keep_lengths
                else [seq_len, seq_len - 1, seq_len - 2]
            )
        elif self.truncation and self.truncation.enabled:
            enable_truncation = True
            truncation_choices = self.truncation.keep_lengths

        dataset = ProcessDataset(
            process_name=self.process,
            process_params=self.process_params,
            sequence_length=seq_len,
            device=device if self.gpu_generation else torch.device("cpu"),
            chunk_size=self.chunk_size,
            num_samples=num_samples,
            start_state_idx=self.start_state_idx,
            steady_state=self.steady_state,
            vocab_map=self.vocab_map
        )

        dataset.enable_truncation = enable_truncation
        dataset.truncation_choices = truncation_choices
        
        
        print(f"[Info] Created {'train' if train else 'test'} dataloader with {num_samples} samples, "
          f"sequence_length={seq_len}, batch_size={current_batch_size}, "
          f"chunk_size={current_chunk_size}, "
          f"device={device if self.gpu_generation else 'cpu'}")
        
        
        return DataLoader(
            dataset=dataset,
            collate_fn=process_dataset_collate_fn,
            batch_size=current_batch_size,
        )


@dataclass
class Log:
    """
    Revised Log class that stores lists of values to avoid sum/avg confusion.
    """
    config: "LoggingConfig"
    
    # Store lists of floats instead of sums
    metrics: Dict[str, Dict[str, PyList[float]]] = field(default_factory=lambda: {"train": {}, "test": {}})

    def reset(self):
        self.metrics = {"train": {}, "test": {}}

    def update_metrics(self, train_or_test: Literal["train", "test"], loss: float, metric_name: str = 'loss'):
        if train_or_test not in self.metrics:
            self.metrics[train_or_test] = {}
        if metric_name not in self.metrics[train_or_test]:
            self.metrics[train_or_test][metric_name] = []
            
        self.metrics[train_or_test][metric_name].append(float(loss))

    def get_aggregated_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate averages for all stored metrics."""
        agg = {}
        for split, metrics_dict in self.metrics.items():
            agg[split] = {}
            for name, values in metrics_dict.items():
                if values:
                    agg[split][name] = sum(values) / len(values)
                else:
                    agg[split][name] = 0.0
        return agg

    def persist(self):
        """Log averaged metrics to WandB."""
        if self.config.wandb:
            agg_metrics = self.get_aggregated_metrics()
            flat = {}
            for split, md in agg_metrics.items():
                for name, val in md.items():
                    flat[f"{split}/{name}"] = val
            
            if flat:
                wandb.log(flat)

class LoggingConfig(Config):
    local: pathlib.Path | None = None
    wandb: bool = True
    project_name: str | None = None
    wandb_api_key: str | None = None
    run_name: str | None = None
    relative_loss: bool = True

    @field_validator("project_name")
    @classmethod
    def validate_wandb_config(cls, v, info):
        if info.data.get("wandb", False) and not v:
            raise ValueError("project_name must be provided if wandb is enabled")
        return v

    def close(self):
        if self.wandb: wandb.finish()

    def init(self) -> Log:
        return Log(config=self)


@dataclass
class NGramAnalysisConfig:
    """Configuration for n-gram KL divergence analysis during validation."""
    enabled: bool = True
    n_values: list[int] = field(default_factory=lambda: [1, 2, 3])
    return_per_position: bool = True

    def __post_init__(self):
        """Validate n_values."""
        if not all(isinstance(n, int) and n >= 1 for n in self.n_values):
            raise ValueError("n_values must be list of positive integers")
        if max(self.n_values) > 5:
            raise ValueError("n_values > 5 may be too slow, recommended max is 3")


@dataclass
class MarkovKLAnalysisConfig:
    """Configuration for Markov process KL divergence analysis during validation."""
    enabled: bool = True
    return_per_position: bool = True


@dataclass
class KLAnalysisConfig:
    """Configuration for all KL analysis metrics."""
    ngram_analysis: NGramAnalysisConfig = field(default_factory=NGramAnalysisConfig)
    markov_kl_analysis: MarkovKLAnalysisConfig = field(default_factory=MarkovKLAnalysisConfig)


@dataclass
class SimplexAnalysisConfig:
    """configuration for simplex mse analyis during validation"""
    enabled:bool=True
    hook_point:str='blocks.0.hook_resid_post'
    num_samples_for_probe: int=1000

class AnalysisConfig(Config):
    """Configuration for all analysis metrics."""
    ngram_analysis: NGramAnalysisConfig = field(default_factory=NGramAnalysisConfig)
    markov_kl_analysis: MarkovKLAnalysisConfig = field(default_factory=MarkovKLAnalysisConfig)
    simplex_analysis: SimplexAnalysisConfig = field(default_factory=SimplexAnalysisConfig)    
    analysis_batch_size:Optional[int]=None
# ============================================================================
# TrainConfig WITH KL Analysis
# ============================================================================

class TrainConfig(Config):
    model: RawModelConfig
    optimizer: OptimizerConfig
    dataset: ProcessDatasetConfig
    persistance: PersistanceConfig
    logging: LoggingConfig
    seed: int
    verbose: bool
    truncation: Optional[TruncationConfig] = None
    do_eval: bool=True
    
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    @property
    def kl_analysis(self):
        return self.analysis

    @model_validator(mode="after")
    def validate_model(self):
        """Validate model vocab matches process vocab (if process is registered)."""
        dataset_process = self.dataset.process

        if dataset_process and dataset_process in PROCESS_REGISTRY:
            try:
                process_instance=PROCESS_REGISTRY[dataset_process](vocab_map=self.dataset.vocab_map,**self.dataset.process_params)
                process_vocab_len = process_instance.vocab_len
                if self.model.d_vocab != process_vocab_len:
                    raise ValueError(
                        f"Model's d_vocab ({self.model.d_vocab}) doesn't match "
                        f"dataset process's vocab_len ({process_vocab_len})"
                    )
            except KeyError:
                # Process not registered, skip validation
                print(f"[Warning] Process '{dataset_process}' not in PROCESS_REGISTRY, skipping vocab validation")
        elif dataset_process:
            print(f"[Warning] Process '{dataset_process}' not found in PROCESS_REGISTRY")
            print(f"[Warning] Available processes: {list(PROCESS_REGISTRY.keys())}")
        
        return self

    def init_logger(self) -> Log:
        """Initialize logger with optional wandb support."""
        if self.logging.wandb:
            dotenv.load_dotenv()
            
            # Try to get API key from config first, then environment
            wandb_api_key = self.logging.wandb_api_key or os.environ.get("WANDB_API_KEY", None)
            
            if wandb_api_key is None:
                raise ValueError(
                    "To use wandb, provide wandb_api_key in config or set WANDB_API_KEY environment variable"
                )
            
            wandb.login(key=wandb_api_key)
            
            resume_id = os.environ.get("WANDB_RUN_ID")
            resume_mode = os.environ.get("WANDB_RESUME", "allow")
            
            if resume_id:
                print(f"Config detected resume request for Run ID: {resume_id}")
                wandb.init(
                    project=self.logging.project_name, 
                    id=resume_id, 
                    resume=resume_mode,
                    # We pass the config here so W&B can log any NEW overrides we made
                    config=self.model_dump() 
                )
            else:
                # Standard fresh run
                wandb.init(
                    project=self.logging.project_name, 
                    name=self.logging.run_name, 
                    config=self.model_dump()
                )

            
            # wandb.init(project=self.logging.project_name, name=self.logging.run_name, config=self.model_dump())
        
        if self.logging.local is not None:
            raise NotImplementedError()
        
        return self.logging.init()

