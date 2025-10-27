import os
from dotenv import load_dotenv

from nam.data import Dataset, AbstractDataset, _DEFAULT_REQUIRE_INPUT_PRE_SILENCE, DataError
import torch as _torch
from huggingface_hub import login
from sentence_transformers import SentenceTransformer, models, losses, InputExample

from copy import deepcopy as _deepcopy
from pathlib import Path as _Path
from typing import (
    Any as _Any,
    Callable as _Callable,
    Optional as _Optional,
    Sequence as _Sequence,
    Tuple as _Tuple,
    Union as _Union,
)

load_dotenv()

login(token=os.getenv("HF_TOKEN"))
gemma_model = SentenceTransformer("google/embeddinggemma-300m")

class PromptDataset(Dataset):
    
    def __init__(self, 
        x: _torch.Tensor,
        y: _torch.Tensor,
        prompt: str,
        embedding_size: int,
        nx: int,
        ny: _Optional[int],
        start: _Optional[int] = None,
        stop: _Optional[int] = None,
        start_samples: _Optional[int] = None,
        stop_samples: _Optional[int] = None,
        start_seconds: _Optional[_Union[int, float]] = None,
        stop_seconds: _Optional[_Union[int, float]] = None,
        delay: _Optional[_Union[int, float]] = None,
        y_scale: float = 1.0,
        x_path: _Optional[_Union[str, _Path]] = None,
        y_path: _Optional[_Union[str, _Path]] = None,
        input_gain: float = 0.0,
        sample_rate: _Optional[float] = None,
        require_input_pre_silence: _Optional[
            float
        ] = _DEFAULT_REQUIRE_INPUT_PRE_SILENCE,
    ):
        super().__init__(x, y, nx, ny, start, stop, 
            start_samples, stop_samples, start_seconds, 
            stop_seconds, delay, y_scale, x_path, y_path, 
            input_gain, sample_rate, require_input_pre_silence
        )
        
        self._prompt = prompt
        self._prompt_emb = gemma_model.encode(self.prompt)
        if self.prompt_emb.shape[0] != embedding_size:
            raise DataError("Embedding size mismatch!")
    
        print(f"x shape: {x.shape}, mean: {x.mean().item()}")
        print(f"y shape: {y.shape}, mean: {y.mean().item()}")
        print(f"Prompt embedding shape: {self._prompt_emb.shape}")

    @property
    def prompt(self):
        return self._prompt 
    
    @prompt.setter
    def prompt(self, value: str):
        self._prompt = value

    @property
    def prompt_emb(self):
        return self._prompt_emb
    
    @prompt_emb.setter
    def prompt_emb(self, value):
        self._prompt_emb = value
        
    def __getitem__(self, idx: int) -> _Tuple:
        x_wind, y_wind = super().__getitem__(idx)
        emb = self._prompt_emb[:, None] 
        return x_wind, emb, y_wind

    @classmethod
    def parse_config(cls, config):
        """ 
        :param config: 
            Must contain:
                x_path 
                y_path
                prompt
                embedding_size
        """
        parse_conf = super().parse_config(config)

        if "prompt" not in config:
            raise DataError("Missing prompt value!")
        if "embedding_size" not in config:
            raise DataError("Missing embedding_size!")

        parse_conf["prompt"] = config.pop("prompt")
        parse_conf["embedding_size"] = config.pop("embedding_size")

        return parse_conf
        
    @classmethod
    def init_from_config(cls, config):
        parsed_conf = cls.parse_config(config)
        return cls(**parsed_conf)
    
# register_dataset_initializer("prompt_dataset", PromptDataset.init_from_config)

