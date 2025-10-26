
import torch as _torch
import torch.nn as nn 

from .base import BaseNet 
from .factory import register
from .wavenet import _WaveNet 

from typing import Optional as _Optional

class PromptWaveNet(BaseNet):
    def __init__(
        self,
        *,
        net_config: dict, 
        embedding_dim: int,
        condition_size: int,
        sample_rate: _Optional[float] = None, 
    ):
        super().__init(sample_rate=sample_rate)
        self._prompt_proj = nn.Linear(embedding_dim, condition_size)
        self._wavenet = _WaveNet(**net_config)

    def _forward(self, x: _torch.Tensor, prompt_emb: _torch.Tensor) -> _torch.Tensor:
        if x.ndim == 2:
            x = x[:, None, :] # B, 1, L where B is batch size and L is length of stream
        if prompt_emb.ndim == 2:
            prompt_emb = prompt_emb[:, :, None] # B, emb_dim, 1
        spread_prompt_emb = prompt_emb.expand(-1, -1, x.shape[-1]) # B, emb_dim, L so we copy da hoe L times
        cond = self._prompt_proj(spread_prompt_emb.transpose(1,2)).transpose(1, 2) # B, cond_size, L 
        y_hat = self._wavenet(x, cond)
        return y_hat[:, 0, :] # B, L_out 
    
    @property
    def pad_start_default(self) -> bool:
        return True 
    
    @property
    def receptive_field(self) -> int:
        return self._wavenet.receptive_field
        

register("PromptWaveNet", PromptWaveNet.init_from_config)