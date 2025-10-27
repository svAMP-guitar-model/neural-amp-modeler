
import torch as _torch
import torch.nn as nn 

from .base import BaseNet 
# from .factory import register
from .wavenet import _WaveNet 

from typing import Optional as _Optional

class PromptWaveNet(BaseNet):
    def __init__(
        self,
        embedding_size: int,
        condition_size: int,
        sample_rate: _Optional[float] = None, 
        **net_config: dict, 
    ):
        super().__init__(sample_rate=sample_rate)
        self._prompt_proj = nn.Linear(embedding_size, condition_size)
        self._wavenet = _WaveNet(**net_config)

    def forward(self, x: _torch.Tensor, prompt_emb: _torch.Tensor, pad_start: bool | None = None, **kwargs):
        pad_start = self.pad_start_default if pad_start is None else pad_start
        
        if x.ndim == 1:
            x = x[None]
        if pad_start:
            x = _torch.cat(
               (_torch.zeros((len(x), self.receptive_field - 1)).to(x.device), x),
                dim=1, 
            )
        if isinstance(prompt_emb, (list, tuple)):
            prompt_emb = _torch.tensor(prompt_emb)
        elif hasattr(prompt_emb, "__class__") and prompt_emb.__class__.__name__ == "ndarray":
            prompt_emb = _torch.from_numpy(prompt_emb)

        return self._forward_mps_safe(x, prompt_emb=prompt_emb)

    def _forward(self, x: _torch.Tensor, prompt_emb: _torch.Tensor) -> _torch.Tensor:
        if x.ndim == 2:
            x = x[:, None, :] # B, 1, L where B is batch size and L is length of stream
        if prompt_emb.ndim == 2:
            prompt_emb = prompt_emb[:, :, None] # B, emb_dim, 1
        spread_prompt_emb = prompt_emb.expand(-1, -1, x.shape[-1]) # B, emb_dim, L so we copy da ho L times
        cond = self._prompt_proj(spread_prompt_emb.transpose(1,2)).transpose(1, 2) # B, cond_size, L 
        y_hat = self._wavenet(x, cond)
        return y_hat[:, 0, :] # B, L_out 
    
    @property
    def pad_start_default(self) -> bool:
        return True 
    
    @property
    def receptive_field(self) -> int:
        return self._wavenet.receptive_field
        
    def _export_config(self):
        return self._wavenet.export_config()
    
    def _export_weights(self):
        return self._wavenet.export_weights()

# register("PromptWaveNet", PromptWaveNet.init_from_config)