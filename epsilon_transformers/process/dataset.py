import torch
import numpy as np
from typing import Iterator, Optional, Union, List
from jaxtyping import Float
from torch.utils.data import IterableDataset
import random

from epsilon_transformers.process.Process import Process
from epsilon_transformers.process.processes import PROCESS_REGISTRY

# TODO: Create a custom dataloader so you don't have to import the collate_function everywehre
# TODO: Assert they are in the correct vocabulary
# TODO: Make the dataset parallel distributed (??)
# TODO: Figure out the device allocation for batching
# TODO: Test the ProcessDataset __iter__ for robustness against StopIter


class ProcessDataset(IterableDataset):
    samples: Iterator[int]
    process_params: dict[str, float]
    sequence_length: int
    num_samples: int
    device:torch.device
    chunk_size: int
    start_state_idx:Optional[int]=None
    steady_state:Optional[list[float]]=None
    process: Process
    

    samples:Optional[Iterator[int]]=None #for cpu fallback

    def __init__(
        self,
        process_name: str,
        process_params: dict[str, float],
        sequence_length: int,
        num_samples: int,
        device: Optional[torch.device] = None,
        chunk_size: int = 2048,
        start_state_idx: Optional[int] = None,
        steady_state: Optional[list[float]] = None,
        vocab_map:Optional[Union[list[int],dict[int,int]]]=None,

    ):
        super().__init__()
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device

        process_class = PROCESS_REGISTRY.get(process_name, None)
        if process_class is None:
            raise ValueError(
                f"{process_name} is not a recognized process. It must be one of the following {PROCESS_REGISTRY.keys()}"
            )
        self.process: Process = process_class(**process_params, vocab_map=vocab_map)


        if steady_state is not None:
            if len(steady_state) != self.process.num_states:
                raise ValueError(
                    f"Provided steady_state length ({len(steady_state)}) does not match process num_states ({self.process.num_states})"
                )
            # Update the instance variable on the process
            self.process.steady_state_vector = np.array(steady_state, dtype=float)
            # Invalidate GPU cache so the new vector is used
            self.process._gpu_steady_state = None

        # self.samples = self.process.yield_emissions(
        #     sequence_len=num_samples * (sequence_length + 1),current_state_idx=start_state_idx
        # )
        # self.process: Process = process_class(**process_params)
        self.sequence_length = sequence_length
        self.num_samples = num_samples
        self.chunk_size = chunk_size
        self.start_state_idx = start_state_idx
        self.steady_state = steady_state
        self.enable_truncation = False
        self.truncation_choices = None
               

    def __len__(self):
        return self.num_samples

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over samples, generating in GPU batches for efficiency.
        
        Yields:
            Tuple of (input_sequence, target_sequence), each of shape (sequence_length,)
        """
        samples_yielded = 0
        
        while samples_yielded < self.num_samples:
            # Calculate chunk size for this iteration
            current_chunk_size = min(self.chunk_size, self.num_samples - samples_yielded)
            
            # Generate batch on GPU: (chunk_size, sequence_length + 1)
            batch_data = self.process.generate_batch_gpu(
                batch_size=current_chunk_size,
                seq_len=self.sequence_length + 1,
                device=self.device,
                start_state_idx=self.start_state_idx,
                
            )
            
            # Yield samples one by one
            # The DataLoader will regroup these into training batch_size
            for i in range(current_chunk_size):
                seq = batch_data[i]
                # Input: 0 to N-1, Target: 1 to N
                input_seq = seq[:-1]
                target_seq =  seq[1:]

                if not self.enable_truncation:
                    prefix_mask = torch.ones(self.sequence_length, dtype=torch.bool, device=self.device)
                    suffix_mask = torch.zeros(self.sequence_length, dtype=torch.bool, device=self.device)

                    yield input_seq, target_seq, prefix_mask, suffix_mask
                else:
                    keep_len = random.choice(self.truncation_choices)
                    prefix_mask = torch.zeros(self.sequence_length, dtype=torch.bool, device=self.device)
                    prefix_mask[:keep_len] = True

                    suffix_mask = ~prefix_mask

                    # PAD_TOKEN = 0  
                    # truncated_input = input_seq.clone()
                    # if keep_len < self.sequence_length:
                    #     truncated_input[keep_len:] = PAD_TOKEN
                    yield input_seq,target_seq,prefix_mask, suffix_mask      
            samples_yielded += current_chunk_size

class ProcessDatasetCPU(IterableDataset):
    """
    Legacy CPU-only dataset (for backward compatibility).
    """
    samples: Iterator[int]
    sequence_length: int
    num_samples: int
    start_state_idx:Optional[int]=None
    steady_state: Optional[list[float]]=None

    def __init__(
        self,
        process_name: str,
        process_params: dict[str, float],
        sequence_length: int,
        num_samples: int,
        start_state_idx: Optional[int] = None,
        steady_state: Optional[list[float]] = None,
    ):
        super().__init__()
        process_class = PROCESS_REGISTRY.get(process_name, None)
        if process_class is None:
            raise ValueError(
                f"{process_name} is not a recognized process. It must be one of the following {PROCESS_REGISTRY.keys()}"
            )
        process: Process = process_class(**process_params)
        if steady_state is not None:
            self.process.steady_state_vector = np.array(steady_state, dtype=float)
        self.samples = process.yield_emissions(
            sequence_len=num_samples * (sequence_length + 1),current_state_idx=start_state_idx
        )
        self.sequence_length = sequence_length
        self.num_samples = num_samples
        self.start_state_idx = start_state_idx
        self.steady_state = steady_state

    def __len__(self):
        return self.num_samples

    def __iter__(self) -> Iterator[tuple[list[int], list[int]]]:
        for _ in range(self.num_samples):
            process_history = [
                next(self.samples) for _ in range(self.sequence_length + 1)
            ]
            yield (process_history[:-1], process_history[1:])



def process_dataset_collate_fn(
    batch: list[tuple[list[int], list[int]]],
) -> tuple[
    Float[torch.Tensor, "batch_size sequence_length"],
    Float[torch.Tensor, "batch_size sequence_length"],
]:
    
    data = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    prefix_mask = [x[2] for x in batch]
    suffix_mask = [x[3] for x in batch]


    if isinstance(data[0], torch.Tensor):
        return (
            torch.stack(data),
            torch.stack(labels),
            torch.stack(prefix_mask),
            torch.stack(suffix_mask),
        )
    else:
        return (
            torch.tensor(data, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(prefix_mask, dtype=torch.bool),
            torch.tensor(suffix_mask, dtype=torch.bool),
        )


