import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

class NGramAnalyzer:
    """
    Highly optimized N-Gram Analyzer using dense PyTorch tensors.
    Avoids dictionaries and Python loops for maximum GPU throughput.
    """
    
    def __init__(self, vocab_size: int, n_grams: List[int] = None):
        self.vocab_size = vocab_size
        self.n_grams = n_grams if n_grams is not None else [1, 2, 3]
        
        # Store probability tables directly on the device as tensors
        # Key: n -> Value: Tensor of shape [V, V, ..., V] (n times)
        self.prob_tables: Dict[int, torch.Tensor] = {}
        self.count_tables: Dict[int, torch.Tensor]={}
        self.device = torch.device('cpu') 

    def build_from_sequences(self, sequences: torch.Tensor) -> None:
        """
        Vectorized build of N-Gram counts.
        Args:
            sequences: LongTensor of shape [Batch, Length]
        """
        self.device = sequences.device
        sequences = sequences.long()
        
        # Safety check for vocab overflow
        if sequences.max() >= self.vocab_size:
            raise ValueError(f"Sequence contains token {sequences.max()} >= vocab_size {self.vocab_size}. Check config.")

        print(f"[NGramAnalyzer] Building statistics on {self.device} for N={self.n_grams}...")

        for n in self.n_grams:
            if sequences.size(1) < n:
                continue

            # 1. Extract all windows of size N
            # unfold -> [Batch, Num_Windows, n]
            windows = sequences.unfold(1, n, 1)
            
            # 2. Flatten -> [Total_Samples, n]
            windows_flat = windows.reshape(-1, n)
            
            # 3. Linearize indices: Index = w_0 * V^(n-1) + ... + w_n-1 * V^0
            # This dense mapping requires vocab_size^n memory. Fine for V<100, N<4.
            powers = torch.tensor(
                [self.vocab_size ** i for i in range(n - 1, -1, -1)], 
                device=self.device, 
                dtype=torch.long
            )
            
            flat_indices = (windows_flat * powers).sum(dim=1)
            
            # 4. Count (Histogram)
            total_bins = self.vocab_size ** n
            counts = torch.bincount(flat_indices, minlength=total_bins).float()
            
            # 5. Reshape back to [V, V, ..., V]
            dense_counts = counts.view([self.vocab_size] * n)
            self.count_tables[n] = dense_counts
            
        self.build_prob_tables_from_counts()
        print(f"[NGramAnalyzer] Build complete.")

    def build_prob_tables_from_counts(self):
        self.prob_tables={}
        for n in self.count_tables:
            if n not in self.count_tables:
                continue
            dense_counts = self.count_tables[n].clone()
            
            # Smoothing prevents log(0) for transitions that are unseen in the data
            dense_counts += 1e-10 
            
            sums = dense_counts.sum(dim=-1, keepdim=True)
            probs = dense_counts / sums
            self.prob_tables[n] = probs

    def merge_ngram_tables(self, other_count_tables: Dict[int, torch.Tensor]):
        print("merging ngram tables")
        for n in self.n_grams:
            if n in other_count_tables:
                if n in self.count_tables:
                    self.count_tables[n] = self.count_tables[n] + other_count_tables[n]
                else:
                    self.count_tables[n] = other_count_tables[n].clone()
        self.build_prob_tables_from_counts()
        print("ngram tables merged")

    def compute_kl_divergence_batch(self, model_logits: torch.Tensor, sequences: torch.Tensor, n: int):
        """
        Vectorized KL computation for a batch.
        Aligns prediction indices to match Transformer autoregressive task.
        """
        # Ensure inputs are on same device
        if sequences.device != self.device: sequences = sequences.to(self.device)
        if model_logits.device != self.device: model_logits = model_logits.to(self.device)

        batch_size, seq_len, vocab_size = model_logits.shape
        if n not in self.prob_tables:
            return torch.tensor([]), torch.tensor([])
        
        prob_table = self.prob_tables[n]

        # --- ALIGNMENT LOGIC ---
        # An N-gram model needs (n-1) tokens of context to predict the Nth token.
        # Transformer `logits[i]` predicts `x_{i+1}` using context `x_0...x_i` (length i+1).
        # To make a valid N-gram prediction, we need `Context_Len >= n-1`.
        # Therefore: `i+1 >= n-1`  =>  `i >= n-2`.
        
        start_idx = max(0, n - 2)
        valid_logits = model_logits[:, start_idx:, :]
        
        if valid_logits.shape[1] == 0:
             return torch.tensor([]), torch.tensor([])

        # --- PREPARE GROUND TRUTH ---
        if n == 1:
            # Unigram: Context independent. GT is constant across positions.
            gt_dist = prob_table.view(1, 1, -1).expand(batch_size, valid_logits.shape[1], vocab_size)
        else:
            # Extract context windows from INPUT sequences corresponding to the valid logits.
            # We need to predict x_{start_idx+1} ... x_L.
            # The context for x_{k} is x_{k-(n-1)} ... x_{k-1}.
            # The first target is x_{start_idx+1}.
            # Context indices: start_idx+1 - (n-1) to start_idx.
            # Since start_idx = n-2, this simplifies to indices: 0 to n-2.
            # This corresponds exactly to the start of the sequence.
            
            # We unfold the sequence into windows of size (n-1).
            context_windows = sequences.unfold(1, n - 1, 1)
            
            # We need the first `valid_logits.shape[1]` windows to match the predictions.
            num_preds = valid_logits.shape[1]
            
            if context_windows.size(1) < num_preds:
                 # Should not happen if start_idx logic is correct, but safety check
                 return torch.tensor([]), torch.tensor([])
                 
            context_windows = context_windows[:, :num_preds, :]

            # Flatten and lookup
            ctx_flat = context_windows.reshape(-1, n - 1) 
            indices = ctx_flat.unbind(dim=1)
            gt_dist_flat = prob_table[indices]
            
            gt_dist = gt_dist_flat.view(batch_size, -1, vocab_size)

        # --- COMPUTE KL ---
        model_log_probs = F.log_softmax(valid_logits, dim=-1)
        
        gt_dist = gt_dist + 1e-12
        gt_log_probs = torch.log(gt_dist)
        
        kl_elementwise = gt_dist * (gt_log_probs - model_log_probs)
        kl_per_token = kl_elementwise.sum(dim=-1) # [Batch, Valid_Len]
        
        kl_per_position = kl_per_token.mean(dim=0)
        kl_all_values = kl_per_token.view(-1)
        
        return kl_per_position, kl_all_values


def compute_ngram_kl_divergence(model_logits, sequences, ngram_analyzer, n_values=None, return_per_position=True, batch_size: Optional[int]=None) -> Dict[str, float]:
    if n_values is None:
        n_values = ngram_analyzer.n_grams
    
    results = {}
    # for n in n_values:
    #     kl_per_position, kl_all_values = ngram_analyzer.compute_kl_divergence_batch(model_logits, sequences, n)
    total_samples = model_logits.shape[0]
    
    for n in n_values:
        if batch_size is None or batch_size >= total_samples:
            kl_per_position, kl_all_values = ngram_analyzer.compute_kl_divergence_batch(model_logits, sequences, n)
        else:
            all_kl_per_pos = []
            all_kl_vals = []
            for i in range(0, total_samples, batch_size):
                k_pos, k_vals = ngram_analyzer.compute_kl_divergence_batch(
                    model_logits[i : i + batch_size], 
                    sequences[i : i + batch_size], 
                    n
                )
                if k_vals.numel() > 0:
                    all_kl_per_pos.append(k_pos)
                    all_kl_vals.append(k_vals)
            
            kl_per_position = torch.stack(all_kl_per_pos).mean(dim=0) if all_kl_per_pos else torch.tensor([])
            kl_all_values = torch.cat(all_kl_vals) if all_kl_vals else torch.tensor([])    
        if kl_all_values.numel() > 0:
            results[f"kl_div_ngram_{n}"] = float(kl_all_values.mean().item())
        else:
            results[f"kl_div_ngram_{n}"] = 0.0
        
        if return_per_position and kl_per_position.numel() > 0:
            kl_pos_list = kl_per_position.tolist()
            for i, val in enumerate(kl_pos_list):
                # Calculate real position index (0-indexed)
                # i=0 corresponds to logits[start_idx].
                # logits[k] predicts pos k+1.
                # So real_pos_idx is the index in the LOGITS array.
                
                start_idx = max(0, n - 2)
                real_pos_idx = start_idx + i 
                
                results[f"kl_div_ngram_{n}_pos_{real_pos_idx}"] = val
    
    return results