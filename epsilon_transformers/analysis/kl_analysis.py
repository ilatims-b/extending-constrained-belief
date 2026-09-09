import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Union
import numpy as np

class MarkovKLAnalyzer:
    """Analyzes KL divergence against ground truth Markov process distributions."""
    
    def __init__(self, vocab_size: int, seq_len: Optional[int] = None):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
    
    def _get_tensor(self, data, device):
        if not isinstance(data, torch.Tensor):
            return torch.tensor(data, dtype=torch.float32, device=device)
        return data.to(device=device, dtype=torch.float32)

    def _get_process_tensors(self, process, device):
        """Helper to get transition matrices for a single process."""
        T_emit = self._get_tensor(process.transition_matrix, device)
        if hasattr(process, 'norm_transition_matrix'):
            T_next = self._get_tensor(process.norm_transition_matrix, device)
        else:
            T_next = T_emit
        return T_emit, T_next

    def compute_ground_truth_distributions(self, 
                                          sequences: torch.Tensor, 
                                          process, 
                                          start_state_idx: Optional[int] = None) -> torch.Tensor:
        """
        Computes the Ground Truth probability distributions P(x_{t+1} | x_0...x_t).
        Output[t] is the GT distribution for token t+1.
        """
        return self._compute_standard_gt_distributions(sequences, process, start_state_idx)

    def compute_kl_divergence_batch(self,
                                   model_logits: torch.Tensor,
                                   sequences: torch.Tensor,
                                   process,
                                   start_state_idx: Optional[int]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        gt_dists = self.compute_ground_truth_distributions(sequences, process, start_state_idx)
        
        batch_size, seq_len, vocab_size = model_logits.shape
        device = model_logits.device
        
        kl_per_position = torch.zeros(seq_len, device=device)
        kl_all_values_list = []

        for pos in range(seq_len):
            # Model Logits at 'pos' predict 'pos+1'
            logit_batch = model_logits[:, pos, :]
            model_log_probs = F.log_softmax(logit_batch, dim=-1)
            
            # GT Dist at 'pos' is also prediction for 'pos+1'
            gt_dist = gt_dists[:, pos, :]
            
            gt_log_probs = torch.log(gt_dist + 1e-10)
            kl_batch = torch.sum(gt_dist * (gt_log_probs - model_log_probs), dim=-1)
            
            kl_per_position[pos] = kl_batch.mean()
            kl_all_values_list.append(kl_batch)

        if len(kl_all_values_list) > 0:
            kl_all_values = torch.stack(kl_all_values_list, dim=0)
        else:
            kl_all_values = torch.tensor([], device=device)

        return kl_per_position, kl_all_values

    # =========================================================================
    # Standard Process Logic
    # =========================================================================

    def _compute_standard_gt_distributions(self, sequences, process, start_state_idx):
        batch_size, seq_len = sequences.shape
        device = sequences.device
        T_emit, T_next = self._get_process_tensors(process, device)
        
        if start_state_idx is not None:
            vec = torch.zeros(process.num_states, device=device, dtype=torch.float32)
            vec[start_state_idx] = 1.0
        else:
            vec = process.steady_state_vector
            if isinstance(vec, np.ndarray):
                vec = torch.from_numpy(vec).float().to(device)
        current_states = vec.unsqueeze(0).expand(batch_size, -1)
        
        T_emit_marginal = T_emit.sum(dim=2).t()
        
        all_gt_dists = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)

        lookup = None
        if process.vocab_map is not None:
            lookup = torch.full((self.vocab_size,), -1, device=device, dtype=torch.long)
            if isinstance(process.vocab_map, dict):
                for k, v in process.vocab_map.items(): lookup[v] = k
            else:
                for k, v in enumerate(process.vocab_map): lookup[v] = k

        for pos in range(seq_len):
            global_emissions = sequences[:, pos]
            local_emissions = global_emissions.clone()
            
            if lookup is not None:
                local_emissions = lookup[global_emissions]

            T_selected = T_next[local_emissions]
            next_states = torch.einsum("bs, bsd -> bd", current_states, T_selected)
            current_states = next_states / (next_states.sum(dim=1, keepdim=True) + 1e-12)

            gt_dist = torch.matmul(current_states, T_emit_marginal)
            
            if process.vocab_map is not None:
                gt_dist_mapped = torch.zeros(batch_size, self.vocab_size, device=device)
                if isinstance(process.vocab_map, dict):
                    vals = list(process.vocab_map.values())
                    keys = list(process.vocab_map.keys())
                    indices = torch.tensor(vals, device=device, dtype=torch.long)
                    gt_dist_mapped.index_add_(1, indices, gt_dist[:, keys])
                elif isinstance(process.vocab_map, list):
                    indices = torch.tensor(process.vocab_map, device=device, dtype=torch.long)
                    gt_dist_mapped.index_add_(1, indices, gt_dist)
                gt_dist = gt_dist_mapped

            gt_dist = gt_dist / (gt_dist.sum(dim=1, keepdim=True) + 1e-12)
            all_gt_dists[:, pos, :] = gt_dist
            #all_gt_dists shape is [batch_size, seq_len, vocab_size]

        return all_gt_dists

def compute_markov_kl_divergence(model_logits, sequences, process, analyzer=None, return_per_position=True, start_state_idx: Optional[int]=None, batch_size:Optional[int]=None) -> Dict[str, float]:
    if analyzer is None:
        analyzer = MarkovKLAnalyzer(model_logits.shape[-1], model_logits.shape[1])
    total_samples = model_logits.shape[0]
    if batch_size is None or batch_size >= total_samples:
        kl_per_position, kl_all_values = analyzer.compute_kl_divergence_batch(
            model_logits, sequences, process, start_state_idx=start_state_idx
        )
    else:
        # Process in chunks to save memory
        all_kl_per_pos = []
        all_kl_vals = []
        chunk_sizes = []
        for i in range(0, total_samples, batch_size):
            batch_logits = model_logits[i : i + batch_size]
            batch_seqs = sequences[i : i + batch_size]
            k_pos, k_vals = analyzer.compute_kl_divergence_batch(
                batch_logits, batch_seqs, process, start_state_idx=start_state_idx
            )
            all_kl_per_pos.append(k_pos)
            all_kl_vals.append(k_vals)
            chunk_sizes.append(batch_logits.shape[0])
            print(f"Processed batch {i // batch_size + 1}")

        # k_pos is (seq_len,) per chunk -> weight by chunk size, not a plain mean,
        # since the last chunk is usually smaller than the rest.
        weights = torch.tensor(chunk_sizes, dtype=torch.float32, device=all_kl_per_pos[0].device)
        weights = weights / weights.sum()
        kl_per_position = (torch.stack(all_kl_per_pos) * weights.unsqueeze(1)).sum(dim=0)
        # k_vals is (seq_len, chunk_batch) per chunk -> concat along the batch dim (1).
        kl_all_values = torch.cat(all_kl_vals, dim=1)
    results = {
        "kl_div_markov": float(kl_all_values.mean().item()),
        # "individual_kls": kl_all_values.detach()
    }
    
    if return_per_position:
        kl_per_pos_cpu = kl_per_position.cpu().tolist()
        for pos_idx, val in enumerate(kl_per_pos_cpu):
            results[f"kl_div_markov_pos_{pos_idx + 1}"] = val
    
    return results