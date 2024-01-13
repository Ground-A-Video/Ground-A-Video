from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention import FeedForward, CrossAttention, AdaLayerNorm
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from .cross_attention_old import CrossAttention # use locally defined CrossAttention
if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


@dataclass
class SpatioTemporalTransformerModelOutput(BaseOutput):
    """torch.FloatTensor of shape [batch x channel x frames x height x width]"""

    sample: torch.FloatTensor


class SpatioTemporalTransformerModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        **transformer_kwargs,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                SpatioTemporalTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    **transformer_kwargs,
                )
                for d in range(num_layers)
            ]
        )

        # Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(
        self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True
    ):
        # 1. Input
        clip_length = None
        is_video = hidden_states.ndim == 5
        if is_video:
            clip_length = hidden_states.shape[2]
            hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
            # this enables applying different context vectros on different frames (differently optimized null-embeddings)
            if encoder_hidden_states.shape[0] != clip_length:
                encoder_hidden_states = encoder_hidden_states.repeat_interleave(clip_length, 0)

        *_, h, w = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            hidden_states = rearrange(hidden_states, "b c h w -> b (h w) c")
        else:
            hidden_states = rearrange(hidden_states, "b c h w -> b (h w) c")
            hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                clip_length=clip_length,
            )

        # 3. Output
        if not self.use_linear_projection:
            hidden_states = rearrange(hidden_states, "b (h w) c -> b c h w", h=h, w=w).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = rearrange(hidden_states, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        output = hidden_states + residual
        if is_video:
            output = rearrange(output, "(b f) c h w -> b c f h w", f=clip_length)

        if not return_dict:
            return (output,)

        return SpatioTemporalTransformerModelOutput(sample=output)


class SpatioTemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        use_sparse_causal_attention: bool = True,   # TODO: deprecate
        use_spatial_temporal_attention: bool = True,
        temporal_attention_position: str = "after_feedforward",
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.use_sparse_causal_attention = use_sparse_causal_attention
        self.use_spatial_temporal_attention = use_spatial_temporal_attention

        self.temporal_attention_position = temporal_attention_position
        temporal_attention_positions = ["after_spatial", "after_cross", "after_feedforward"]
        if temporal_attention_position not in temporal_attention_positions:
            raise ValueError(
                f"`temporal_attention_position` must be one of {temporal_attention_positions}"
            )

        # 1. Spatial-Attn (Spatial-Temporal Self-Attention)
        spatial_attention = SpatialTemporalAttnetion if use_spatial_temporal_attention else CrossAttention    ## ST -> ST-Full
        self.attn1 = spatial_attention(
            query_dim=dim,
            key_dim=dim,
            value_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
        )  
        self.norm1 = (
            AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        )

        # 2. Cross-Attn (Modulated Cross-Attention)
        if cross_attention_dim is not None:
            self.attn2 = ModulatedCrossAttention(
                query_dim=dim,
                key_dim=cross_attention_dim,
                value_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
            )
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
            )
        else:
            self.attn2 = None
            self.norm2 = None

        # 4. Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)
    

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool, valid:bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        attention_mask=None,
        clip_length=None,
    ):
        # 1. Self-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        kwargs = dict(
            hidden_states=norm_hidden_states,
            attention_mask=attention_mask,
        )
        if self.only_cross_attention:
            kwargs.update(encoder_hidden_states=encoder_hidden_states)
        if self.use_spatial_temporal_attention:
            kwargs.update(clip_length=clip_length)
        hidden_states = hidden_states + self.attn1(hidden_states=kwargs["hidden_states"], attention_mask=kwargs["attention_mask"], clip_length=clip_length)

        if self.attn2 is not None:
            # 2. Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states,
                    mask=attention_mask,
                )
                + hidden_states
            )

        # 3. Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states

    def apply_temporal_attention(self, hidden_states, timestep, clip_length):
        d = hidden_states.shape[1]
        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=clip_length)
        norm_hidden_states = (
            self.norm_temporal(hidden_states, timestep)
            if self.use_ada_layer_norm
            else self.norm_temporal(hidden_states)
        )
        hidden_states = self.attn_temporal(norm_hidden_states) + hidden_states
        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
        return hidden_states


class CA(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64, dropout=0):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads


        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)


        self.to_out = nn.Sequential( nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )


    def fill_inf_from_mask(self, sim, mask):
        if mask is not None:
            B,M = mask.shape
            mask = mask.unsqueeze(1).repeat(1,self.heads,1).reshape(B*self.heads,1,-1)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)
        return sim 


    def forward(self, x, key, value, mask=None):

        q = self.to_q(x)     
        k = self.to_k(key)   
        v = self.to_v(value) 
   
        B, N, HC = q.shape 
        _, M, _ = key.shape
        H = self.heads
        C = HC // H 

        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) 
        k = k.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) 
        v = v.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) 
        
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=mask, op=None
        )
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) 

        return self.to_out(out)


class SpatialTemporalAttnetion(CA):
    def forward(self, hidden_states, attention_mask=None, clip_length=None):
        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        key = rearrange(key, "(b f) d c -> b f d c", f=clip_length)
        key = torch.cat( [ key[:, [iii]*clip_length] for iii in range(clip_length) ], dim=2 )
        key = rearrange(key, "b f d c -> (b f) d c", f=clip_length)
        
        value = rearrange(value, "(b f) d c -> b f d c", f=clip_length)
        value = torch.cat( [ value[:, [iii]*clip_length] for iii in range(clip_length) ], dim=2 )
        value = rearrange(value, "b f d c -> (b f) d c", f=clip_length)

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)
        
        out = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=None
        )

        out = self.reshape_batch_dim_to_heads(out)
        return self.to_out(out)


    def reshape_heads_to_batch_dim(self,tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor


    def reshape_batch_dim_to_heads(self,tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor


class ModulatedCrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64, dropout=0):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential( nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )
        
        # NOTE: when forwarding with unet3d, follow order `conditional prediction -> unconditional prediction`
        self.is_conditional_branch = True

    def fill_inf_from_mask(self, sim, mask):
        if mask is not None:
            B,M = mask.shape
            mask = mask.unsqueeze(1).repeat(1,self.heads,1).reshape(B*self.heads,1,-1)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)
        return sim 

    def forward(self, x, key, value, mask=None):
        # B*N*(H*C)
        q = self.to_q(x)     
        k = self.to_k(key)   
        v = self.to_v(value) 
   
        B, N, HC = q.shape 
        _, M, _ = key.shape
        H = self.heads
        C = HC // H 

        q = self.reshape_heads_to_batch_dim(q)
        clip_length = key.shape[0]
        
        if not self.is_conditional_branch:
            # concat k
            k = rearrange(k, "(b f) d c -> b f d c", f=clip_length)
            k = torch.cat( [ k[:, [iii]*clip_length] for iii in range(clip_length) ], dim=2 )
            k = rearrange(k, "b f d c -> (b f) d c", f=clip_length)

            # concat v
            v = rearrange(v, "(b f) d c -> b f d c", f=clip_length)
            v = torch.cat( [ v[:, [iii]*clip_length] for iii in range(clip_length) ], dim=2 )
            v = rearrange(v, "b f d c -> (b f) d c", f=clip_length)        
    
        self.is_conditional_branch = not self.is_conditional_branch

        k = self.reshape_heads_to_batch_dim(k)
        v = self.reshape_heads_to_batch_dim(v)

        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=mask, op=None
        )
        out = self.reshape_batch_dim_to_heads(out)

        return self.to_out(out)
    
    def reshape_heads_to_batch_dim(self,tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self,tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

