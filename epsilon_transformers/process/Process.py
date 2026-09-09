import numpy as np
from typing import Iterator, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from jaxtyping import Float
from collections import deque
import torch

from epsilon_transformers.process.MixedStateTree import (
    MixedStateTree,
    MixedStateTreeNode,
)

# TODO: Test yield_emission_histories for different emissions in the emission history
# TODO: Rename _create_hmm
# TODO: Delete generate_process_history (??)


@dataclass
class ProcessHistory:
    symbols: list[int]
    states: list[str]

    def __post_init__(self):
        pass
        # assert len(self.symbols) == len(
        #     self.states
        # ), "length of symbols & states must be the same"

    def __len__(self):
        return len(self.states)

    
class Process(ABC):
    name: str
    transition_matrix: Float[np.ndarray, "vocab_len num_states num_states"]
    state_names_dict: dict[str, int]
    vocab_len: int
    num_states: int
    steady_state_vector: Float[np.ndarray, "num_states"]
    vocab_map: Optional[Union[list[int], dict[int, int]]]
    
    # GPU tensors (lazily initialized)
    _gpu_transition_matrix: Optional[torch.Tensor] = None
    _gpu_steady_state: Optional[torch.Tensor] = None
    _gpu_vocab_map: Optional[torch.Tensor] = None
    _gpu_device: Optional[torch.device] = None

    def __init__(self, vocab_map: Optional[Union[list[int], dict[int, int]]] = None):
        self.transition_matrix, self.state_names_dict = self._create_hmm()
        
        if (
            len(self.transition_matrix.shape) != 3
            or self.transition_matrix.shape[1] != self.transition_matrix.shape[2]
        ):
            raise ValueError(
                "Transition matrix should have 3 axes and the final two dims shoulds be square"
            )

        if self.transition_matrix.shape[1] != self.transition_matrix.shape[2]:
            raise ValueError("Transition matrix should be square")

        transition = self.transition_matrix.sum(axis=0)
        if not np.allclose(transition.sum(axis=1), 1.0):
            raise ValueError("Transition matrix should be stochastic and sum to 1")

        self.vocab_len = self.transition_matrix.shape[0]
        self.num_states = self.transition_matrix.shape[1]
        self.steady_state_vector = self._compute_steady_state()
        
        # Vocabulary Mapping Setup
        self.vocab_map = vocab_map
        if self.vocab_map is not None:
            if len(self.vocab_map) != self.vocab_len:
                raise ValueError(f"vocab_map length ({len(self.vocab_map)}) must match process vocab_len ({self.vocab_len})")
        
        # Reset GPU cache
        self._gpu_transition_matrix = None
        self._gpu_steady_state = None
        self._gpu_vocab_map = None
        self._gpu_device = None

    def _compute_steady_state(self) -> Float[np.ndarray, "num_states"]:
        """Calculates steady state vector from transition matrix."""
        state_transition_matrix = np.sum(self.transition_matrix, axis=0)
        eigenvalues, eigenvectors = np.linalg.eig(state_transition_matrix.T)
        steady_state_vector = eigenvectors[:, np.isclose(eigenvalues, 1)].real
        
        if steady_state_vector.shape[1] > 0:
            steady_state_vector = steady_state_vector[:, 0]
            
        normalized_steady_state_vector = steady_state_vector / steady_state_vector.sum()
        return normalized_steady_state_vector

    @property
    def is_unifilar(self) -> bool:
        for i in range(self.num_states):
            for j in range(self.vocab_len):
                if np.count_nonzero(self.transition_matrix[j, i, :]) > 1:
                    return False
        return True

    @abstractmethod
    def _create_hmm(
        self,
    ) -> tuple[Float[np.ndarray, "vocab_len num_states num_states"], dict[str, int]]:
        ...

    def _apply_vocab_map(self, emission_idx: int) -> int:
        if self.vocab_map is None:
            return emission_idx
        if isinstance(self.vocab_map, dict):
            return self.vocab_map[emission_idx]
        return self.vocab_map[emission_idx]

    def _ensure_gpu_tensors(self, device: torch.device):
        """Lazily initialize GPU tensors for batch generation."""
        if self._gpu_device != device or self._gpu_transition_matrix is None:
            self._gpu_transition_matrix = torch.tensor(
                self.transition_matrix, dtype=torch.float32, device=device
            )
            if self.vocab_map is not None:
                # Convert dict to list for tensor creation if necessary
                v_map = self.vocab_map
                if isinstance(v_map, dict):
                    v_map = [v_map[i] for i in range(self.vocab_len)]
                self._gpu_vocab_map = torch.tensor(v_map, dtype=torch.long, device=device)
            
            self._gpu_device = device

    def _get_gpu_steady_state(self, device: torch.device) -> torch.Tensor:
        if self._gpu_steady_state is None or self._gpu_device != device:
            self._gpu_steady_state = torch.tensor(
                self.steady_state_vector, dtype=torch.float32, device=device
            )
        return self._gpu_steady_state

    def generate_batch_gpu(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        start_state_idx: Optional[int] = None,
    ) -> torch.Tensor:
        self._ensure_gpu_tensors(device)
        T = self._gpu_transition_matrix 
        
        if start_state_idx is not None:
            current_states = torch.full((batch_size,), start_state_idx, dtype=torch.long, device=device)
        else:
            steady_state = self._get_gpu_steady_state(device)
            probs = steady_state.unsqueeze(0).expand(batch_size, -1) if steady_state.ndim==1 else steady_state
            current_states = torch.multinomial(probs, num_samples=1).squeeze(-1)

        emissions = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
        
        for t in range(seq_len):
            trans_probs = T[:, current_states, :].permute(1, 0, 2)
            flat_probs = trans_probs.reshape(batch_size, -1)
            joint_idx = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
            
            emission = joint_idx // self.num_states
            next_state = joint_idx % self.num_states
            
            emissions[:, t] = emission
            current_states = next_state
        
        # Apply vocab mapping if it exists
        if self._gpu_vocab_map is not None:
            emissions = self._gpu_vocab_map[emissions]
            
        return emissions
        
    def generate_batch_gpu_with_beliefs(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        start_state_idx: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            emissions: (batch_size, seq_len) - Discrete observations
            true_states: (batch_size, seq_len) - Ground truth state indices
            beliefs: (batch_size, seq_len, num_states) - Observer's belief distributions
        """
        self._ensure_gpu_tensors(device)
        T = self._gpu_transition_matrix  # (vocab_len, num_states, num_states)
        
        # 1. Initialize Ground Truth States
        if start_state_idx is not None:
            current_states = torch.full((batch_size,), start_state_idx, dtype=torch.long, device=device)
            # Initial belief is certain
            current_belief = torch.zeros((batch_size, self.num_states), device=device)
            current_belief[:, start_state_idx] = 1.0
        else:
            steady_state = self._get_gpu_steady_state(device)
            current_states = torch.multinomial(
                steady_state.unsqueeze(0).expand(batch_size, -1), 
                num_samples=1
            ).squeeze(-1)
            # Initial belief starts at steady state
            current_belief = steady_state.unsqueeze(0).expand(batch_size, -1)

        emissions = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
        true_states = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
        beliefs = torch.empty(batch_size, seq_len, self.num_states, dtype=torch.float32, device=device)

        for t in range(seq_len):
            # --- A. Generate Next Step (Ground Truth) ---
            trans_probs = T[:, current_states, :].permute(1, 0, 2) # (batch_size, vocab_len, num_states)
            flat_probs = trans_probs.reshape(batch_size, -1)
            
            joint_idx = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
            
            emission = joint_idx // self.num_states
            next_state = joint_idx % self.num_states
            
            # --- B. Update Observer's Belief (Filtering) ---
            # The observer sees 'emission' and updates current_belief
            # T_emit shape: (batch_size, num_states, num_states)
            T_emit = T[emission] 
            
            # Belief Update: b_{t} = (b_{t-1} @ T_emit) / Normalization
            # [B, 1, S] @ [B, S, S] -> [B, 1, S] -> [B, S]
            next_belief = torch.bmm(current_belief.unsqueeze(1), T_emit).squeeze(1)
            
            # Normalize
            denom = next_belief.sum(dim=1, keepdim=True)
            current_belief = next_belief / torch.where(denom < 1e-12, torch.ones_like(denom), denom)

            if self._gpu_vocab_map is not None:
                stored_emission=self._gpu_vocab_map[emission]
            else:
                stored_emission=emission    
            # --- C. Store results ---
            emissions[:, t] = stored_emission
            true_states[:, t] = next_state
            beliefs[:, t] = current_belief
            
            current_states = next_state
        
        return emissions, true_states, beliefs

    def _sample_emission(self, current_state_idx: Optional[int] = None) -> int:
        if current_state_idx is None:
            current_state_idx = np.random.choice(
                self.num_states, p=self.steady_state_vector
            )

        assert (
            0 <= current_state_idx < self.num_states
        ), "current_state_index must be positive & less than num_states"

        p = self.transition_matrix[:, current_state_idx, :].sum(axis=1)
        emission = np.random.choice(self.vocab_len, p=p)
        if self.vocab_map is not None:
            emission = self._apply_vocab_map(emission)
        return emission    

    def yield_emissions(
        self, sequence_len: int, current_state_idx: int | None = None
    ) -> Iterator[int]:
        if current_state_idx is None:
            current_state_idx = np.random.choice(
                self.num_states, p=self.steady_state_vector
            )
        assert (
            0 <= current_state_idx < self.num_states
        ), "current_state_index must be positive & less than num_states"
        for _ in range(sequence_len):
            emission, next_state_idx = self._sample_emission_and_next_state(
                current_state_idx
            )
            yield emission
            current_state_idx = next_state_idx


    def _sample_emission_and_next_state(
        self, current_state_idx: int
    ) -> tuple[int, int]:
        transition_probs = self.transition_matrix[:, current_state_idx, :] 
        emission_next_state_idx = np.random.choice(
            transition_probs.size, p=transition_probs.ravel()
        )
        emission = emission_next_state_idx // self.num_states
        next_state_idx = emission_next_state_idx % self.num_states
        if self.vocab_map is not None:
            emission = self._apply_vocab_map(emission)
        return emission, next_state_idx
    
    def yield_emission_histories(
        self, sequence_len: int, num_sequences: int, start_state_idx: Optional[int]=None
    ) -> Iterator[list[int]]:
        for _ in range(num_sequences):
            yield [x for x in self.yield_emissions(sequence_len=sequence_len, current_state_idx=start_state_idx)]

    def generate_process_history(
        self, total_length: int, current_state_idx: int | None = None
    ) -> ProcessHistory:
        if current_state_idx is None:
            current_state_idx = np.random.choice(
                self.num_states, p=self.steady_state_vector
            )
        assert (
            0 <= current_state_idx < self.num_states
        ), "current_state_index must be positive & less than num_states"

        index_to_state_names_dict = {v: k for k, v in self.state_names_dict.items()}

        symbols = []
        states = []

        for _ in range(total_length):
            states.append(index_to_state_names_dict[current_state_idx])
            emission, next_state_idx = self._sample_emission_and_next_state(
                current_state_idx
            )
            symbols.append(emission)
            current_state_idx = next_state_idx

        return ProcessHistory(symbols=symbols, states=states)

    # TODO: You can get rid of the stack, and just iterate through the nodes & the depth as tuples
    #added
    def derive_mixed_state_presentation(self, depth: int,  start_state_idx: Optional[int] = None) -> MixedStateTree:
        if start_state_idx is not None:
            assert 0 <= start_state_idx < self.num_states, "start_state_idx out of bounds"
            initial_dist = np.zeros(self.num_states)
            initial_dist[start_state_idx] = 1.0
        else:
            initial_dist = self.steady_state_vector

        tree_root = MixedStateTreeNode(
            state_prob_vector = initial_dist,
            children=set(),
            path=[],
            emission_prob=0,
        )
        nodes = set([tree_root])

        stack: deque[
            tuple[MixedStateTreeNode, Float[np.ndarray, "num_states"], list[int], int]
        ] = deque([(tree_root, initial_dist, [], 0)])
        while stack:
            current_node, state_prob_vector, current_path, current_depth = stack.pop()
            if current_depth < depth:
                emission_probs = _compute_emission_probabilities(
                    self, state_prob_vector
                )
                for emission in range(self.vocab_len):
                    if emission_probs[emission] > 0:
                        next_state_prob_vector = _compute_next_distribution(
                            self.transition_matrix, state_prob_vector, emission
                        )
                        child_path = current_path + [emission]
                        child_node = MixedStateTreeNode(
                            state_prob_vector=next_state_prob_vector,
                            path=child_path,
                            children=set(),
                            emission_prob=emission_probs[emission],
                        )
                        current_node.add_child(child_node)

                        stack.append(
                            (
                                child_node,
                                next_state_prob_vector,
                                child_path,
                                current_depth + 1,
                            )
                        )
            nodes.add(current_node)

        return MixedStateTree(
            root_node=tree_root, process=self.name, nodes=nodes, depth=depth
        )

def _compute_emission_probabilities(
    hmm: Process, state_prob_vector: Float[np.ndarray, "num_states"]
) -> Float[np.ndarray, "vocab_len"]:
    """
    Compute the probabilities associated with each emission given the current mixed state.
    """
    T = hmm.transition_matrix
    emission_probs = np.einsum("s,esd->ed", state_prob_vector, T).sum(axis=1)
    emission_probs /= emission_probs.sum()
    return emission_probs

def _compute_next_distribution(
    epsilon_machine: Float[np.ndarray, "vocab_len num_states num_states"],
    current_state_prob_vector: Float[np.ndarray, "num_states"],
    current_emission: int,
) -> Float[np.ndarray, "num_states"]:
    """
    Compute the next mixed state distribution for a given output.
    """
    X_next = np.einsum(
        "sd, s -> d", epsilon_machine[current_emission], current_state_prob_vector
    )
    return X_next / np.sum(X_next) if np.sum(X_next) != 0 else X_next

class NormTransitionMixin:
    # GPU tensors for the norm matrix
    _gpu_norm_transition_matrix: Optional[torch.Tensor] = None
    def __init__(self, **kwargs):
        # Call Process.__init__(), forwarding any kwargs (e.g. vocab_map)
        super().__init__(**kwargs)
        # Add extra matrix needed only in this subclass
        self.norm_transition_matrix = self._create_norm_matrix()
        if (
            len(self.norm_transition_matrix.shape) != 3
            or self.norm_transition_matrix.shape[1] != self.norm_transition_matrix.shape[2]
        ):
            raise ValueError(
                "Transition matrix should have 3 axes and the final two dims shoulds be square"
            )

        if self.norm_transition_matrix.shape[1] != self.norm_transition_matrix.shape[2]:
            raise ValueError("Transition matrix should be square")

        transition = self.norm_transition_matrix.sum(axis=0)
        # if not np.allclose(transition.sum(axis=1), 1.0):
        #     raise ValueError("Transition matrix should be stochastic and sum to 1")
        # Reset GPU cache for norm matrix
        self._gpu_norm_transition_matrix = None
        

    @abstractmethod
    def _create_norm_matrix(
        self,
    ) -> Float[np.ndarray, "vocab_len num_states num_states"] :
        """
        Create the HMM which defines the process.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices.
        """
        ...
    def _ensure_gpu_tensors(self, device: torch.device):
        """Override to also initialize norm transition matrix on GPU."""
        # Call parent's _ensure_gpu_tensors
        super()._ensure_gpu_tensors(device)
        # Also initialize norm transition matrix
        if self._gpu_norm_transition_matrix is None or self._gpu_device != device:
            self._gpu_norm_transition_matrix = torch.tensor(
                self.norm_transition_matrix, dtype=torch.float32, device=device
            )
    def generate_batch_gpu(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        start_state_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a batch of sequences on GPU for NormTransitionMixin processes.
        
        For these processes:
        - Emission is sampled from transition_matrix (marginal over next states)
        - Next state is sampled from norm_transition_matrix given the emission
        """
        self._ensure_gpu_tensors(device)
        
        T = self._gpu_transition_matrix  # (vocab_len, num_states, num_states)
        T_norm = self._gpu_norm_transition_matrix  # (vocab_len, num_states, num_states)
        
        # Sample initial states for all sequences: (batch_size,)
        if start_state_idx is not None:
            assert 0 <= start_state_idx < self.num_states, f"Invalid start_state_idx: {start_state_idx}"
            current_states = torch.full(
                (batch_size,), 
                start_state_idx, 
                dtype=torch.long, 
                device=device
            )
        else:
            steady_state = self._get_gpu_steady_state(device)  # (num_states,)
            current_states = torch.multinomial(
                steady_state.unsqueeze(0).expand(batch_size, -1), 
                num_samples=1
            ).squeeze(-1)  # (batch_size,)
        
        
        
        emissions = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
        
        for t in range(seq_len):
            # Get transition probs for current states
            # T[:, current_states, :] -> (vocab_len, batch_size, num_states)
            trans_probs = T[:, current_states, :]  # (vocab_len, batch_size, num_states)
            
            # Marginalize over next states to get emission probs: (batch_size, vocab_len)
            emission_probs = trans_probs.sum(dim=2).permute(1, 0)  # (batch_size, vocab_len)
            
            # Sample emissions
            emission = torch.multinomial(emission_probs, num_samples=1).squeeze(-1)  # (batch_size,)
            if self._gpu_vocab_map is not None:
                emissions[:, t] = self._gpu_vocab_map[emission]
            else:
                emissions[:, t] = emission


            # Now sample next state from norm_transition_matrix given emission
            # T_norm[emission, current_states, :] -> need to gather properly
            # For each sample i, we need T_norm[emission[i], current_states[i], :]
            
            # Efficient gather: (vocab_len, num_states, num_states) -> index by emission and current_states
            # T_norm[emission] gives (batch_size, num_states, num_states)
            next_state_probs = T_norm[emission, current_states, :]  # (batch_size, num_states)
            
            # Sample next states
            next_state = torch.multinomial(next_state_probs, num_samples=1).squeeze(-1)  # (batch_size,)
            current_states = next_state
        
        return emissions

    def _sample_emission_and_next_state(self, current_state_idx: int):
        """Override the Process version with the linear-process version."""
        transition_probs = self.transition_matrix[:, current_state_idx, :]

        emission_next_state_idx = np.random.choice(
            transition_probs.size, p=transition_probs.ravel()
        )

        emission = emission_next_state_idx // self.num_states


        next_state_idx = np.random.choice(
            self.num_states,
            p=self.norm_transition_matrix[emission, current_state_idx, :]
        )

        return emission, next_state_idx
    
    def derive_mixed_state_presentation(self, depth: int, start_state_idx: Optional[int] = None) -> MixedStateTree:
        if start_state_idx is not None:
            assert 0 <= start_state_idx < self.num_states, "start_state_idx out of bounds"
            initial_dist = np.zeros(self.num_states)
            initial_dist[start_state_idx] = 1.0
        else:
            initial_dist = self.steady_state_vector
        tree_root = MixedStateTreeNode(
            state_prob_vector= initial_dist,
            children=set(),
            path=[],
            emission_prob=0,
        )
        nodes = set([tree_root])

        stack: deque[
            tuple[MixedStateTreeNode, Float[np.ndarray, "num_states"], list[int], int]
        ] = deque([(tree_root, initial_dist, [], 0)])
        while stack:
            current_node, state_prob_vector, current_path, current_depth = stack.pop()
            if current_depth < depth:
                emission_probs = self._compute_emission_probabilities(
                    self, state_prob_vector
                )
                for emission in range(self.vocab_len):
                    if emission_probs[emission] > 0:
                        next_state_prob_vector = self._compute_next_distribution(
                            self.norm_transition_matrix, state_prob_vector, emission
                        )
                        child_path = current_path + [emission]
                        child_node = MixedStateTreeNode(
                            state_prob_vector=next_state_prob_vector,
                            path=child_path,
                            children=set(),
                            emission_prob=emission_probs[emission],
                        )
                        current_node.add_child(child_node)

                        stack.append(
                            (
                                child_node,
                                next_state_prob_vector,
                                child_path,
                                current_depth + 1,
                            )
                        )
            nodes.add(current_node)

        return MixedStateTree(
            root_node=tree_root, process=self.name, nodes=nodes, depth=depth
        )
    
    def _compute_emission_probabilities(self,
    hmm: Process, state_prob_vector: Float[np.ndarray, "num_states"]
) -> Float[np.ndarray, "vocab_len"]:
        """
        Compute the probabilities associated with each emission given the current mixed state.
        """
        T = hmm.transition_matrix
        emission_probs = np.einsum("s,esd->ed", state_prob_vector, T).sum(axis=1)
        emission_probs /= emission_probs.sum()
        return emission_probs


    def _compute_next_distribution(self,
        epsilon_machine: Float[np.ndarray, "vocab_len num_states num_states"],
        current_state_prob_vector: Float[np.ndarray, "num_states"],
        current_emission: int,
    ) -> Float[np.ndarray, "num_states"]:
            """
            Compute the next mixed state distribution for a given output.
            """
            X_next = np.einsum(
                "sd, s -> d", epsilon_machine[current_emission], current_state_prob_vector
            )
            return X_next 



