
import torch
from torch import nn

from llm.core.rms_norm import RMSNorm
from llm.core.swi_glu import SwiGLU
from llm.core.rope import RoPE
from llm.core.group_query_attention import GroupedQueryAttention

class MistralDecoder(nn.Module):
    def __init__(self, 
        num_q_heads: int,
        num_kv_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        window_size: int,
        rope: RoPE,
        dropout: float = 0.1
    ):
        super().__init__()
        self._heads = GroupedQueryAttention(
            num_q_heads=num_q_heads, 
            num_kv_heads=num_kv_heads,
            emb_size=emb_size, 
            head_size=head_size, 
            max_seq_len=max_seq_len,
            window_size=window_size,
            rope=rope,
            dropout=dropout
        )
        self._ff = SwiGLU(emb_size=emb_size, dropout=dropout)
        self._norm1 = RMSNorm(emb_size)
        self._norm2 = RMSNorm(emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, use_cache: bool = True, cache: list = None) -> torch.Tensor:
        norm1_out = self._norm1(x)
        attention, kv_caches = self._heads(norm1_out, mask, use_cache=use_cache, cache=cache)
        out = attention + x
        
        norm2_out = self._norm2(out)
        ffn_out = self._ff(norm2_out)

        if use_cache is True:
            return (ffn_out + out, kv_caches)
        else:
            return (ffn_out + out, None)