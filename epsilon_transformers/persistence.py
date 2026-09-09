import torch
import pathlib
from typing import Optional, Dict, Any, List, Tuple
import csv
import pandas as pd  
import re
import json
from collections import OrderedDict
from epsilon_transformers.training.configs.model_configs import RawModelConfig
from epsilon_transformers.analysis.ngram_analysis import NGramAnalyzer

class Persister:
    """Handles model persistence and checkpoint management."""
    
    def __init__(self, save_dir: str = "./checkpoints"):
        """
        Initialize persister.
        
        Args:
            save_dir: Directory to save checkpoints
        """
        self.save_dir = pathlib.Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        existing = self.get_model_checkpoints()
        self.checkpoint_count = len(existing)

    def save_config(self, config_dict: Dict[str, Any]):
        """Saves the training configuration to JSON."""
        with open(self.save_dir / 'train_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)    

    def save_model(self, model: Any, tokens_trained: int, metadata: Optional[Dict] = None):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            tokens_trained: Number of tokens trained so far
            metadata: Optional metadata to save with checkpoint
        """
        checkpoint_num = self.checkpoint_count
        checkpoint_path = self.save_dir / f"checkpoint_{checkpoint_num}_tokens_{tokens_trained}.pt"
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'tokens_trained': tokens_trained,
            'checkpoint_number': checkpoint_num,
            'metadata': metadata or {}
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_count += 1
        
        print(f"[Persister] Saved checkpoint {checkpoint_num} at {checkpoint_path}")

    
    def load_training_config(self) -> Dict[str, Any]:
        """Loads the training configuration from JSON."""
        config_path = self.save_dir / 'train_config.json'
        if not config_path.exists():
            checkpoints=self.get_model_checkpoints()
            if not checkpoints:
                print("no checkpts to infer cfg")
                return None
            latest=checkpoints[-1]
            checkpoint_data=torch.load(latest, map_location='cpu')
            if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
                state_dict=checkpoint_data['model_state_dict']
            else:    
                state_dict=checkpoint_data

            config_obj=_state_dict_to_model_config(state_dict=state_dict)    

            # Convert Pydantic object to Dictionary for compatibility
            if hasattr(config_obj, 'model_dump'):
                return config_obj.model_dump()
            
            if hasattr(config_obj,'dict'):
                return config_obj.dict()

            return config_obj.__dict__    
                
        
        with open(config_path, 'r') as f:
            return json.load(f)

    def get_model_checkpoints(self) -> List[pathlib.Path]:
        """
        Retrieve sorted list of model checkpoints.
        Supports both formats:
        1. checkpoint_X_tokens_Y.pt
        2. Y.pt (e.g. 6400.pt)
        """
        all_files = list(self.save_dir.glob("*.pt"))
        valid_checkpoints = []

        def parse_tokens(path: pathlib.Path) -> int:
            # Format 1: Pure number "6400.pt"
            short_match = re.match(r"^(\d+)\.pt$", path.name)
            if short_match:
                return int(short_match.group(1))
            
            # Format 2: "checkpoint_X_tokens_6400.pt"
            long_match = re.search(r"tokens_(\d+)", path.name)
            if long_match:
                return int(long_match.group(1))
            
            return -1

        for p in all_files:
            # Filter out auxiliary files like ngram counts or configs
            if "ngram_counts" in p.name or "train_config" in p.name:
                continue
            
            if parse_tokens(p) != -1:
                valid_checkpoints.append(p)

        print(f"[Persister] Found {len(valid_checkpoints)} checkpoints in {self.save_dir}")
        return sorted(valid_checkpoints, key=parse_tokens)      

    def load_model(self, checkpoint_path: pathlib.Path|str, device: str='cpu') -> Any:
        """
        Load model from checkpoint.
        
        Args:
            model: Model to load weights into
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint dictionary with metadata
        """
        checkpoint_path=pathlib.Path(checkpoint_path)
        if not checkpoint_path.exists():
            checkpoint_path=self.save_dir / checkpoint_path
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"checkpt not found at {checkpoint_path}")
        checkpoint_data=torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
            state_dict=checkpoint_data['model_state_dict']
        else:
            state_dict=checkpoint_data
        
        train_config = self.load_training_config()
        config=None
        if train_config is not None:
            raw_dict=train_config.get('model',train_config).copy()
            if 'n_heads' in raw_dict and 'n_head' not in raw_dict:
                raw_dict['n_head']=raw_dict.pop('n_heads')
            if 'd_mlp' not in raw_dict and 'd_model' in raw_dict:
                raw_dict['d_mlp']=4*raw_dict['d_model']
                print("assuming d_mlp=4*d_model")  
            try:  
                valid_keys=set(RawModelConfig.model_fields.keys())
            except AttributeError:
                valid_keys=set(RawModelConfig.__fields__.keys())

            filtered_dict={k: v for k, v in raw_dict.items() if k in valid_keys} 
            try:
                config=RawModelConfig(**filtered_dict)
            except Exception as e:
                print(f"{e} error constructing rawmodelconfig from json config, fallback to state dict")

                config=_state_dict_to_model_config(state_dict=state_dict)       

        if config is None:
            print("No train_config.json found, inferring from state_dict")
            config=_state_dict_to_model_config(state_dict=state_dict)

        try:
            model= config.to_hooked_transformer(device=device)
        except Exception as e:
            raise ValueError(f"failed to initialize hookedtransformer: {e}")    
        model.load_state_dict(state_dict=state_dict)
        return model    

    def load_final_model(self, device: str='cpu'):
        checkpoints=self.get_model_checkpoints()
        latest=checkpoints[-1]
        return self.load_model(checkpoint_path=latest, device=device)
    
    def load_latest_checkpoint(self, device: str = "cpu"):
        checkpoints = self.get_model_checkpoints()
        if not checkpoints:
            print("[Persister] No checkpoints found")
            return None

        latest = checkpoints[-1]
        print(f"[Persister] Loading checkpoint: {latest.name}")
        return torch.load(latest, map_location=device)

    def save_metrics_to_csv(self, split: str, metrics: Dict[str, float], step: int):
        """
        Append metrics to train_logs.csv or test_logs.csv.

        Args:
            split: 'train' or 'test'
            metrics: dictionary of metric_name -> value
            step: current training step (e.g., tokens_trained)
        """
        assert split in ["train", "test"], "split must be 'train' or 'test'"

        filename = self.save_dir / f"{split}_logs.csv"
        fieldnames = ["step", "metric_name", "value"]

        # Create file with header if it doesn't exist
        file_exists = filename.exists()
        with open(filename, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            for metric_name, val in metrics.items():
                writer.writerow({
                    "step": step,
                    "metric_name": metric_name,
                    "value": val
                })

        print(f"[Persister] Logged {len(metrics)} {split} metrics to {filename.name}")

    def load_metrics_csv(self, split: str) -> Optional[pd.DataFrame]:
        """
        Load persisted CSV metrics as a DataFrame.
        Returns None if file doesn't exist.
        """
        filename = self.save_dir / f"{split}_logs.csv"
        if not filename.exists():
            print(f"[Persister] No {split}_logs.csv found at {self.save_dir}")
            return None
        df = pd.read_csv(filename)
        print(f"[Persister] Loaded {len(df)} entries from {filename.name}")
        return df

    def load_val_log(self) -> pd.DataFrame:
        return self.load_metrics_csv('test')

    def load_train_log(self) -> pd.DataFrame:
        return self.load_metrics_csv('train')
    

    def save_ngram_data(self, analyzer: 'NGramAnalyzer', tokens_trained: int):
        if analyzer is None:
            return
        
        filename = self.save_dir / f"ngram_counts_tokens_{tokens_trained}.pt"
        data = {
            #'prob_tables': analyzer.prob_tables,
            'count_tables': analyzer.count_tables,
            'n_grams': analyzer.n_grams,
            'vocab_size': analyzer.vocab_size
        }
        torch.save(data, filename)
        print(f"[Persister] Saved N-Gram count tables to {filename}")

    def load_ngram_data(self, tokens_trained: int, device: str = 'cpu') -> Optional[Dict]:
        filename = self.save_dir / f"ngram_counts_tokens_{tokens_trained}.pt"
        
        if not filename.exists():
            files = list(self.save_dir.glob("ngram_counts_tokens_*.pt"))
            if not files:
                print("[Persister] No saved N-Gram counts found.")
                return None
            def parse_tokens(p):
                match=re.search(r"tokens_(\d+)",p.name)
                return int(match.group(1)) if match else -1    
            files.sort(key=parse_tokens)
            valid_files=[f for f in files if parse_tokens(f)< tokens_trained]
            if not valid_files:
                print("[Persister] No saved N-Gram counts found for this tokens_trained.")
                return None
        try:
            data=torch.load(filename,map_location=device) 
            return data
        except Exception as e:
            print(f"[persister]failed to load ngram counts:{e}")   
            return None        

    
def _state_dict_to_model_config(state_dict: OrderedDict, n_ctx: int = 10) -> RawModelConfig:
        _HOOKED_TRANSFORMER_MODULE_REGEXES_REGISTRY: Dict[str, List[Tuple[str, int]]] = {
            r"embed\.W_E": [('d_vocab', 0), ('d_model', 1)],
            r"pos_embed\.W_pos": [],
            r"blocks\.\d+\.ln\d+\.(w|b)": [],
            r"blocks\.\d+\.attn\.W_Q": [('n_head', 0), ('d_head', 2)],
            r"blocks\.\d+\.attn\.b_Q": [],
            r"blocks\.\d+\.attn\.W_K": [],
            r"blocks\.\d+\.attn\.b_K": [],
            r"blocks\.\d+\.attn\.W_O": [],
            r"blocks\.\d+\.attn\.b_O": [],
            r"blocks\.\d+\.attn\.W_V": [],
            r"blocks\.\d+\.attn\.b_V": [],
            r"blocks\.\d+\.attn\.mask": [],
            r"blocks\.\d+\.attn\.IGNORE": [],
            r"blocks\.\d+\.mlp\.W_in": [('d_mlp', 1)],
            r"blocks\.\d+\.mlp\.b_in": [],
            r"blocks\.\d+\.mlp\.W_out": [],
            r"blocks\.\d+\.mlp\.b_out": [],
            r"ln_final\.(w|b)": [],
            r"unembed\.(W_U|b_U)": []
        }
        def _extract_true_key(dictionary: Dict[str, bool]) -> str:
            out = []
            for key, value in dictionary.items():
                if value:
                    out.append(key)
            assert len(out) == 1,  f"{out} does not fit one of the expected module regexs: {_HOOKED_TRANSFORMER_MODULE_REGEXES_REGISTRY}"
            return out[0]
        def _extract_n_layers(state_dict: OrderedDict) -> int:
            highest_block_idx = None
            for key in state_dict.keys():
                if not bool(re.match(r"blocks\.\d+\.", key)):
                    continue
                local_block_idx = int(re.search(r'\d+', key).group())
                if highest_block_idx is None:
                    highest_block_idx = local_block_idx
                elif local_block_idx > highest_block_idx:
                    highest_block_idx = local_block_idx
            return highest_block_idx + 1

        param_dict = dict(d_vocab=None, d_model=None, n_ctx=n_ctx, d_head=None, n_head=None, d_mlp=None, n_layers=_extract_n_layers(state_dict=state_dict))
        for module_name, module in state_dict.items():
            regex_dict = {pattern: bool(re.match(pattern, module_name)) for pattern in _HOOKED_TRANSFORMER_MODULE_REGEXES_REGISTRY.keys()}
            pattern = _extract_true_key(regex_dict)
            for key, dim in _HOOKED_TRANSFORMER_MODULE_REGEXES_REGISTRY[pattern]:
                if param_dict[key] is None:
                    param_dict[key] = module.size()[dim]
        assert all([value is not None for value in param_dict.values()])
        return RawModelConfig(**param_dict)


              