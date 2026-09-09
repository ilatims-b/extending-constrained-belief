import fire
import pathlib
import random
import re
import numpy as np
import time
from pyparsing import Opt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os
import dotenv
from tqdm import tqdm
from typing import Tuple, Optional
from torch.utils.data import DataLoader
from epsilon_transformers.process.MixedStateTree import MixedStateTree
from epsilon_transformers.process.processes import Trun_Mess3
from epsilon_transformers.persistence import Persister
from epsilon_transformers.training.configs.training_configs import (
    TrainConfig,
    Log,
    ProcessDatasetConfig,
)
from epsilon_transformers.analysis.kl_analysis import MarkovKLAnalyzer, compute_markov_kl_divergence
from epsilon_transformers.analysis.ngram_analysis import NGramAnalyzer, compute_ngram_kl_divergence
from epsilon_transformers.analysis.simplex_analysis import SimplexAnalyzer
# from epsilon_transformers.process import Process

from epsilon_transformers.process.processes import PROCESS_REGISTRY


from epsilon_transformers.visualization.plots import _project_to_simplex
from epsilon_transformers.analysis.activation_analysis import get_beliefs_for_transformer_inputs

def get_process_object(config: ProcessDatasetConfig):
    """Return an instantiated Process object given name and parameters."""
    if config.process not in PROCESS_REGISTRY:
        raise ValueError(f"Process {config.process} not found in registry")
    return PROCESS_REGISTRY[config.process](**config.process_params,vocab_map=config.vocab_map)


def _set_random_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _calculate_tokens_trained(batch_size: int, sequence_len: int, batch_idx: int) -> int:
    """Calculate total tokens trained up to this batch."""
    tokens_per_batch = batch_size * sequence_len
    total_tokens_trained = (batch_idx + 1) * tokens_per_batch
    return total_tokens_trained


def _check_if_action_batch(
    perform_action_every_n_tokens: int,
    batch_size: int,
    sequence_len: int,
    batch_idx: int,
) -> bool:
    """Check if this batch should trigger a checkpoint/evaluation."""
    tokens_per_batch = batch_size * sequence_len
    assert (
        perform_action_every_n_tokens >= tokens_per_batch
    ), "perform_action_every_n_tokens must be >= tokens_per_batch"
    perform_action_every_n_batches = perform_action_every_n_tokens // tokens_per_batch
    return (batch_idx + 1) % perform_action_every_n_batches == 0



def _setup_analyzers(
    config: TrainConfig,
    vocab_size: int,
    device:torch.device
) -> Tuple[Optional[NGramAnalyzer], Optional[MarkovKLAnalyzer], Optional[SimplexAnalyzer]]:
    """Initialize KL divergence analyzers if enabled."""
    ngram_analyzer = None
    markov_analyzer = None
    simplex_analyzer = None
    
    analysis_cfg=getattr(config,'analysis',None)
    if analysis_cfg is None and hasattr(config,'kl_analysis'):
        analysis_cfg=config.kl_analysis

    if analysis_cfg:
        if analysis_cfg.ngram_analysis.enabled:
            n_values = analysis_cfg.ngram_analysis.n_values
            ngram_analyzer = NGramAnalyzer(vocab_size=vocab_size, n_grams=n_values) 
        if analysis_cfg.markov_kl_analysis.enabled:
            markov_analyzer = MarkovKLAnalyzer(vocab_size=vocab_size)
        if hasattr(analysis_cfg,'simplex_analysis') and analysis_cfg.simplex_analysis.enabled: 
            simplex_analyzer = SimplexAnalyzer(
                hook=analysis_cfg.simplex_analysis.hook_point, device=device
            )
    return ngram_analyzer, markov_analyzer, simplex_analyzer 

def _compute_myopic_entropy(val_process:object, n_ctx: int, device: torch.device, start_state_idx: Optional[int] = None) -> torch.Tensor:
    """Compute theoretical minimum (myopic) cross-entropy per position for given process."""
    #MSP_tree = mixed_state_tree(process, n_ctx + 1)
    mixed_state_tree = val_process.derive_mixed_state_presentation(depth=n_ctx + 1,start_state_idx=start_state_idx)
    #MSP_transition_matrix = mixed_state_tree.build_msp_transition_matrix()
    #block_entropy = mixed_state_tree.block_entropy
    myopic_entropy_rate = mixed_state_tree.myopic_entropy
    minimum_cross_entropy = myopic_entropy_rate 
    print(f"myopic entropy rates:{minimum_cross_entropy}")
    return torch.tensor(minimum_cross_entropy, dtype=torch.float32, device=device)

def _compute_relative_losses(loss_tensor: torch.Tensor, minimum_cross_entropy: Optional[torch.Tensor],token_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean loss and relative loss per position.
    loss_tensor: (batch, seq_len) 
    """
    if token_mask is not None:
        # avoid division by zero
        mask_f = token_mask.float()
        per_position_loss = (loss_tensor * mask_f).sum(dim=0) / mask_f.sum(dim=0).clamp(min=1.0)
    else:
        per_position_loss = loss_tensor.mean(dim=0)

   
    #print(per_position_loss)
    if minimum_cross_entropy is not None:
        relative_loss = per_position_loss / minimum_cross_entropy
        mean_loss=per_position_loss.mean()
        return mean_loss, relative_loss
    else:  
        mean_loss = per_position_loss.mean()
        relative_loss = None
        return mean_loss, relative_loss


def _compute_validation_metrics(
    model,
    eval_dataloader:DataLoader,
    device: torch.device,
    log: Log,
    config: TrainConfig,
    ngram_analyzer: Optional[NGramAnalyzer] = None,
    markov_analyzer: Optional[MarkovKLAnalyzer] = None,
    simplex_analyzer: Optional[SimplexAnalyzer] = None,
    val_process: Optional[object] = None,
    return_per_position: bool = True,
    minimum_cross_entropy:Optional[torch.Tensor]=None,
    start_state_idx: Optional[int] = None
) -> Log:
    """Compute validation metrics including loss and KL divergences."""

    model.eval()

    criterion=nn.CrossEntropyLoss(reduction="none")

    all_logits = []
    all_sequences = []
    num_batches = 0 

    t_start_eval=time.time()
    print("[eval] starting loop")

    with torch.no_grad():
        t_model_fwd=0.0
        for i, batch in enumerate(eval_dataloader):
            t0 = time.time()

            input_data, target_data, prefix_mask, suffix_mask = batch

            input_data,target_data = input_data.to(device), target_data.to(device)
            prefix_mask = prefix_mask.to(device)
            suffix_mask = suffix_mask.to(device)
            if model.cfg.pad_token_id is not None:
                PAD_TOKEN = model.cfg.pad_token_id
                truncated_input = input_data.clone()
                truncated_input[suffix_mask] = PAD_TOKEN
            else:
                truncated_input = input_data    
            logits=model(truncated_input, return_type="logits")
            loss=criterion(logits.view(-1,logits.size(-1)),target_data.view(-1))
            loss=loss.view(input_data.shape[0],input_data.shape[1])
            if minimum_cross_entropy is not None:
                mean_loss, relative_loss=_compute_relative_losses(loss,minimum_cross_entropy,prefix_mask)
                
                #  Log per-batch metrics (Log class will average them later)
                log.update_metrics("test", loss=mean_loss.item(), metric_name="loss")
                log.update_metrics("test", loss=relative_loss.mean().item(), metric_name="relative_loss")
                for i, rel_val in enumerate(relative_loss):
                    log.update_metrics("test", loss=rel_val.item(), metric_name=f"relative_loss_{i}")
            else:
                mean_loss,_=_compute_relative_losses(loss,minimum_cross_entropy=None)
                log.update_metrics("test", loss=mean_loss.item(), metric_name="loss")        
            masked_logits = logits * prefix_mask.unsqueeze(-1)
            masked_sequences = truncated_input * prefix_mask

            # Collect for KL analysis
            all_logits.append(masked_logits)
            all_sequences.append(masked_sequences)

            num_batches += 1
            t1 = time.time()
            t_model_fwd += (t1 - t0)
        print(f"[eval] model forward time: {t_model_fwd:.3f} seconds over {num_batches} batches")    
    analysis_batch_size=None
    if analysis_config := getattr(config, 'analysis', None):
        analysis_batch_size = analysis_config.analysis_batch_size
    if analysis_batch_size is None:
        analysis_batch_size=config.dataset.test_batch_size    
    if simplex_analyzer is not None:
        t_simplex_start=time.time()
        try:
            _,_,_,mse = simplex_analyzer.compute_simplex_mse(model)
            log.update_metrics("test", metric_name="simplex_mse", loss=mse)
            print(f"[eval] Simplex MSE: {mse:.6f} ({time.time() - t_simplex_start:.3f}s)")
        except Exception as e:
            print(f"[eval] Simplex Analysis Failed: {e}")

    if (ngram_analyzer is not None or markov_analyzer is not None) and len(all_logits)>0:
        t_concat_start=time.time()
        all_logits_tensor=torch.cat(all_logits,dim=0)
        all_sequences_tensor=torch.cat(all_sequences,dim=0)
        print(f"[eval] concat: {time.time()-t_concat_start:.3f}s")
        print(f'[kl]computing kl metrics on {len(all_sequences_tensor)} sequences')
        if ngram_analyzer is not None:
            if not ngram_analyzer.prob_tables and ngram_analyzer.count_tables:
                print("[kl] warning: prob_tables are empty, rebuilding from count_tables before computing kl")
                ngram_analyzer.build_prob_tables_from_counts()
            t_ngram_start=time.time()
            if ngram_analyzer.device != device:
                print(f"[kl] moving ngram analyzer to {device}")
                for n in ngram_analyzer.prob_tables:
                    ngram_analyzer.prob_tables[n]=ngram_analyzer.prob_tables[n].to(device)
                ngram_analyzer.device=device
            ngram_metrics=compute_ngram_kl_divergence(
                all_logits_tensor,
                all_sequences_tensor,
                ngram_analyzer=ngram_analyzer,
                n_values=ngram_analyzer.n_grams,
                return_per_position=return_per_position,
                batch_size=analysis_batch_size
            )    
            for metric_name, metric_value in ngram_metrics.items():
                log.update_metrics("test", metric_name=metric_name, loss=metric_value)
            print(f"[kl] ngram kl time: {time.time()-t_ngram_start:.3f}s")

        if markov_analyzer is not None and val_process is not None:
            t_markov_start=time.time()
            markov_metrics=compute_markov_kl_divergence(
                all_logits_tensor,
                all_sequences_tensor,
                process=val_process,
                analyzer=markov_analyzer,
                return_per_position=return_per_position,
                start_state_idx=start_state_idx,
                batch_size=analysis_batch_size
            )
            for metric_name, metric_value in markov_metrics.items():
                log.update_metrics("test", metric_name=metric_name, loss=metric_value)
            print(f"[kl] markov kl time: {time.time()-t_markov_start:.3f}s")
    print(f"[eval] total eval time: {time.time()-t_start_eval:.3f}s")
    return log                    

def _evaluate_log_and_persist(
    persister,
    model,
    optimizer,
    verbose: bool,
    log: Log,
    config: TrainConfig,
    device: torch.device,
    tokens_trained: int,
    dataset_config: ProcessDatasetConfig,
    ngram_analyzer: Optional[NGramAnalyzer] = None,
    markov_analyzer: Optional[MarkovKLAnalyzer] = None,
    simplex_analyzer: Optional[SimplexAnalyzer] = None,
    val_process: Optional[object] = None,
    return_per_position: bool = True,
    minimum_cross_entropy: Optional[torch.Tensor]=None,
    do_eval: bool = True

):
    """Evaluate model, log metrics, and persist checkpoint."""
    if do_eval:
        eval_dataloader = dataset_config.to_dataloader(
            sequence_length=model.cfg.n_ctx, train=False
        )
        with torch.no_grad():
            _compute_validation_metrics(
                model=model,
                eval_dataloader=eval_dataloader,
                device=device,
                log=log,
                config=config,
                ngram_analyzer=ngram_analyzer,
                markov_analyzer=markov_analyzer,
                simplex_analyzer=simplex_analyzer,
                val_process=val_process,
                return_per_position=return_per_position,
                minimum_cross_entropy=minimum_cross_entropy,
                start_state_idx=dataset_config.start_state_idx
                
            )
    aggregated_metrics=log.get_aggregated_metrics()
    if verbose:
        train_loss = aggregated_metrics.get("train", {}).get("loss", 0.0)
        test_loss = aggregated_metrics.get("test", {}).get("loss", 0.0)
        tqdm.write(f"[Step {tokens_trained}] Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    # 3. Persist Metrics to CSV
    if "train" in aggregated_metrics:
        persister.save_metrics_to_csv("train", aggregated_metrics["train"], tokens_trained)
    if "test" in aggregated_metrics:
        persister.save_metrics_to_csv("test", aggregated_metrics["test"], tokens_trained)
        
    # 4. Persist to WandB and Save Checkpoint
    log.persist() # Logs averages to WandB
    
    persister.save_model(
    model,
    tokens_trained,
    metadata={
        **aggregated_metrics,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    )

    log.reset()


def evaluate_suffix_only(model, dataloader, device,suffix_eval: bool = False) -> Optional[float]:
    
    if not suffix_eval:
        return None
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_data, target_data, prefix_mask, suffix_mask = batch
            input_data = input_data.to(device)
            target_data = target_data.to(device)
            suffix_mask = suffix_mask.to(device)

            logits = model(input_data, return_type="logits")

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target_data.view(-1),
            ).view(target_data.shape)

            suffix_loss = loss[suffix_mask]
            if suffix_loss.numel() > 0:
                total_loss += suffix_loss.sum().item()
                total_tokens += suffix_loss.numel()

    return total_loss / max(total_tokens, 1)




def train_model(config: TrainConfig, run_id: str = None,save_dir: str = None, return_per_position: bool = True) -> Tuple:
    """Train transformer model with KL analysis metrics."""
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    print(f"[Training] Using device: {device}")
    
    _set_random_seed(config.seed)

    if run_id:
        print(f"Setting up environment to RESUME run: {run_id}")
        os.environ["WANDB_RUN_ID"] = run_id
        os.environ["WANDB_RESUME"] = "must"  
    
    # Initialize logger and persister(handles wandb setup)
    log = config.init_logger()
    persister=config.persistance.init()
    persister.save_config(config.model_dump())
    
    # Initialize model and optimizer
    model = config.model.to_hooked_transformer(device=device, seed=config.seed)
    model.cfg.pad_token_id = config.model.pad_token_id
    optimizer = config.optimizer.from_model(model=model, device=device)

    tokens_trained_so_far = 0

    if run_id:
        save_dir = pathlib.Path(save_dir)
        all_files = list(save_dir.glob("*.pt"))
        valid_checkpoints = []

        def parse_tokens(path: pathlib.Path) -> int:
            
            short_match = re.match(r"^(\d+)\.pt$", path.name)
            if short_match:
                return int(short_match.group(1))
            
            
            long_match = re.search(r"tokens_(\d+)", path.name)
            if long_match:
                return int(long_match.group(1))
            
            return -1

        for p in all_files:
            
            if "ngram_counts" in p.name or "train_config" in p.name:
                continue
            
            if parse_tokens(p) != -1:
                valid_checkpoints.append(p)

        print(f"[Persister] Found {len(valid_checkpoints)} checkpoints in {save_dir}")
        checkpoints=  sorted(valid_checkpoints, key=parse_tokens) 
        
        latest = checkpoints[-1]
        checkpoint = torch.load(latest, map_location=device)
        print("[Resume] The checkpoints are loaded")
        
        if checkpoint is not None:
            print("[Resume] Restoring model & optimizer (checkpoints loaded)")

            model.load_state_dict(checkpoint["model_state_dict"])

            optimizer_state = checkpoint["metadata"].get("optimizer_state_dict")

            if optimizer_state is None:
                raise RuntimeError("Optimizer state missing in checkpoint metadata")
            optimizer.load_state_dict(optimizer_state)

            tokens_trained_so_far = checkpoint.get("tokens_trained", 0)
            print(f"[Resume] Resumed from {tokens_trained_so_far} tokens")
    print(f"[Training] Creating dataloaders...")
    # Create data loaders
    train_dataloader = config.dataset.to_dataloader(
        sequence_length=model.cfg.n_ctx, train=True
    )

    print(f"[Training] Dataloaders created")
    
    # Initialize KL analyzers
    val_process = get_process_object(config.dataset)
    if config.logging.relative_loss:
        minimum_cross_entropy = _compute_myopic_entropy(val_process, model.cfg.n_ctx, device, start_state_idx=config.dataset.start_state_idx)
    else:
        minimum_cross_entropy = None

    ngram_analyzer, markov_analyzer, simplex_analyzer = _setup_analyzers(
        config=config,
        vocab_size=model.cfg.d_vocab,
        device=device
    )


    num_samples=None
    if hasattr(config.analysis, 'simplex_analysis'):
        num_samples=config.analysis.simplex_analysis.num_samples_for_probe
    if simplex_analyzer is not None:
        simplex_analyzer.setup_from_tree(process=val_process,depth=model.cfg.n_ctx + 1,num_samples=num_samples,start_state_idx=config.dataset.start_state_idx)

    last_action_batch_tokens=0#for ngram analyzer

    model.train()
    
    train_sequences_since_last_action=[]
    start_batch = tokens_trained_so_far // (
    config.dataset.batch_size * model.cfg.n_ctx
    )
    tokens_per_batch = config.dataset.batch_size * model.cfg.n_ctx

    # Training loop
    for batch_idx, (input_data, target_data,prefix_mask,suffix_mask) in enumerate(
        tqdm(train_dataloader, desc="Train Loop")
    ):
        if batch_idx < start_batch:
            continue
        t0 = time.time()
        input_data = input_data.to(device)
        # print(input_data.size(0))
        target_data = target_data.to(device)
        prefix_mask = prefix_mask.to(device)
        suffix_mask = suffix_mask.to(device)
        if ngram_analyzer is not None:
            train_sequences_since_last_action.append(input_data)
        if model.cfg.pad_token_id is not None:
            print(f"[Training] Applying padding token for suffix masking")
            PAD_TOKEN = model.cfg.pad_token_id
            truncated_input = input_data.clone()
            truncated_input[suffix_mask] = PAD_TOKEN
        else:
            truncated_input = input_data    

        logits = model(truncated_input, return_type="logits")
        criterion=nn.CrossEntropyLoss(reduction="none")
        loss_per_token = criterion(logits.view(-1, logits.size(-1)), target_data.view(-1))
        loss_per_token = loss_per_token.view(input_data.size(0), input_data.size(1))
        mean_loss, _ = _compute_relative_losses(loss_per_token, minimum_cross_entropy,prefix_mask)

        log.update_metrics(train_or_test="train", loss=mean_loss.item(),metric_name="loss")
        if config.logging.wandb and (batch_idx + 1) % 10 == 0:
            wandb.log({
                "train/step_loss": mean_loss.item(),
                "tokens_trained": tokens_trained_so_far

            },
            step=tokens_trained_so_far
        )
        optimizer.zero_grad()
        mean_loss.backward()
        optimizer.step()
        
        t1 = time.time()
        print(f"[TIMING] Train batch took {t1 - t0:.3f} seconds")

        tokens_trained_so_far +=prefix_mask.sum().item()

        
        # Checkpoint and evaluation
        if _check_if_action_batch(
            perform_action_every_n_tokens=config.persistance.checkpoint_every_n_tokens,
            batch_size=config.dataset.batch_size,
            batch_idx=batch_idx,
            sequence_len=model.cfg.n_ctx,
        ):
            model.eval()
            t0 = time.time()
            if ngram_analyzer is not None and len(train_sequences_since_last_action)>0:
                print(f"[train] merging ngram analyzer count tables from {last_action_batch_tokens} to {tokens_trained_so_far}")
                new_train_tensor=torch.cat(train_sequences_since_last_action,dim=0)
                temp_analyzer=NGramAnalyzer(vocab_size=model.cfg.d_vocab,n_grams=ngram_analyzer.n_grams)
                temp_analyzer.build_from_sequences(new_train_tensor)#on same device as new_train_tensor
                if last_action_batch_tokens==0:
                    ngram_analyzer.count_tables=temp_analyzer.count_tables
                else:
                    ngram_analyzer.merge_ngram_tables(temp_analyzer.count_tables)    
                persister.save_ngram_data(ngram_analyzer,tokens_trained=tokens_trained_so_far)
                print(f"[train] ngram analyzer count tables merged and saved")
                
                # Free memory
                del new_train_tensor
                del train_sequences_since_last_action
                train_sequences_since_last_action = []
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Update last action batch tracker
                last_action_batch_tokens = tokens_trained_so_far   

            _evaluate_log_and_persist(
                persister=persister,
                model=model,
                optimizer=optimizer,
                log=log,
                verbose=config.verbose,
                config=config,
                device=device,
                dataset_config=config.dataset,
                tokens_trained=tokens_trained_so_far,
                ngram_analyzer=ngram_analyzer,
                markov_analyzer=markov_analyzer,
                simplex_analyzer=simplex_analyzer,
                val_process=val_process,
                return_per_position=return_per_position,
                minimum_cross_entropy=minimum_cross_entropy,
                do_eval=getattr(config,'do_eval',True)
            )
            t1 = time.time()
            #print(f"[TIMING] Full evaluation step took {t1 - t0:.3f} seconds")
            model.train()
    # Final evaluation
    model.eval()
    #build final ngram table for remaining sequences
    if ngram_analyzer is not None and len(train_sequences_since_last_action) > 0:
        print(f"[Training] Building final N-Gram table...")
        new_train_tensor = torch.cat(train_sequences_since_last_action, dim=0)
        temp_analyzer=NGramAnalyzer(vocab_size=model.cfg.d_vocab,n_grams=ngram_analyzer.n_grams)
        temp_analyzer.build_from_sequences(new_train_tensor)#on same device as new_train_tensor
        if last_action_batch_tokens==0:
            ngram_analyzer.count_tables=temp_analyzer.count_tables
        else:
            ngram_analyzer.merge_ngram_tables(temp_analyzer.count_tables)   
        
        persister.save_ngram_data(ngram_analyzer, tokens_trained=tokens_trained_so_far)
    
    _evaluate_log_and_persist(
        persister=persister,
        model=model,
        optimizer=optimizer,
        log=log,
        verbose=config.verbose,
        config=config,
        device=device,
        tokens_trained=tokens_trained_so_far,
        dataset_config=config.dataset,
        ngram_analyzer=ngram_analyzer,
        markov_analyzer=markov_analyzer,
        simplex_analyzer=simplex_analyzer,
        val_process=val_process,
        return_per_position=return_per_position,
        minimum_cross_entropy=minimum_cross_entropy
    )

    suffix_loader = config.dataset.to_dataloader(
        sequence_length=model.cfg.n_ctx,
        train=False,
        suffix_eval=True,
    )

    suffix_ce = evaluate_suffix_only(model, suffix_loader, device,suffix_eval=False)
    
    if suffix_ce is not None:
        print(f"Suffix Only Cross-Entropy: {suffix_ce:.4f}") 
        wandb.log({"test/suffix_only_ce": suffix_ce})

    
    # Close logger
    config.logging.close()
    
    return model, log


def _main(config_path: pathlib.Path):
    """Main entry point."""
    config = TrainConfig.from_yaml(config_path)
    train_model(config)


if __name__ == "__main__":
    fire.Fire(_main)

