from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils import checkpoint
import xformers
from typing import Optional, List
from einops import rearrange, repeat


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class CrossAttention(nn.Module):
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
        # B*N*(H*C)
        q = self.to_q(x)
        k = self.to_k(key)
        v = self.to_v(value)
   
        B, N, HC = q.shape 
        _, M, _ = key.shape
        H = self.heads
        C = HC // H 

        # (B*H)*N*C
        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C)
        k = k.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C)
        v = v.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C)
        
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=mask, op=None
        )
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C))

        return self.to_out(out)


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


class SelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )

    def forward(self, x):
        # B*N*(H*C)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        B, N, HC = q.shape 
        H = self.heads
        C = HC // H 

        # (B*H)*N*C
        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) 
        k = k.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C)
        v = v.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C)
        
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=None
        )
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C))

        return self.to_out(out)


class SpatialTemporalAttention(CrossAttention):
    def forward(self, hidden_states, attention_mask=None, clip_length=None):
        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # concatenate across all frames
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


class GatedSelfAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim,  n_heads, d_head):
        super().__init__()
        
        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  


    def forward(self, x, objs, clip_length=None):

        N_visual = x.shape[1]
        objs = self.linear(objs)
        x = x + self.scale*torch.tanh(self.alpha_attn) * self.attn(  self.norm1(torch.cat([x,objs],dim=1))  )[:,0:N_visual,:]
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) )  
        
        return x 


class CrossFrameGatedAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim,  n_heads, d_head):
        super().__init__()
        
        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SpatialTemporalAttention(
                    query_dim=query_dim, 
                    key_dim=query_dim,
                    value_dim=query_dim,
                    heads=n_heads, 
                    dim_head=d_head,
                    dropout=0,
                    )
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  


    def forward(self, x, objs, clip_length):
        N_visual = x.shape[1]
        objs = self.linear(objs)
        x = x + self.scale*torch.tanh(self.alpha_attn) * self.attn(  hidden_states=self.norm1(torch.cat([x,objs],dim=1)), clip_length=clip_length  )[:,0:N_visual,:]
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) )  
        
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head, fuser_type, use_checkpoint=True):
        super().__init__()
        self.attn1 = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)  
        self.ff = FeedForward(query_dim, glu=True)
        self.attn2 = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads, dim_head=d_head)  
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.use_checkpoint = use_checkpoint
        self.fuser = GatedSelfAttentionDense(query_dim, key_dim, n_heads, d_head) # note key_dim here actually is context_dim


    def forward(self, x, context, objs):
        if self.use_checkpoint and x.requires_grad:
            return checkpoint.checkpoint(self._forward, x, context, objs)
        else:
            return self._forward(x, context, objs)

    def _forward(self, x, context, objs): 
        x = self.attn1( self.norm1(x) ) + x 
        x = self.fuser(x, objs) # identity mapping in the beginning 
        x = self.attn2(self.norm2(x), context, context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class BasicTransformerBlock3D(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head, fuser_type, 
                    use_checkpoint=True, temporal_attention_position="after_feedforward"):
        super().__init__()

        # 1. Spatial-Attn (Spatial-Temporal Self-Attention)
        dropout = 0
        only_cross_attention = False
        self.attn1 = SpatialTemporalAttention(
            query_dim=query_dim,
            key_dim=query_dim,
            value_dim=query_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        # 4. Feed Forward
        self.ff = FeedForward(query_dim, glu=True)
        # 3. Cross-Attn (Modulated Cross-Attention)
        self.attn2 = ModulatedCrossAttention(
            query_dim=query_dim, 
            key_dim=key_dim, 
            value_dim=value_dim, 
            heads=n_heads, 
            dim_head=d_head)  
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.use_checkpoint = use_checkpoint

        # 2. Gated-Attn (Cross-Frame Gated-Attention)
        self.fuser = CrossFrameGatedAttentionDense(query_dim, key_dim, n_heads, d_head) # note key_dim here actually is context_dim

    def forward(self, x, context, objs, clip_length=None):
        if self.use_checkpoint and x.requires_grad:
            return checkpoint.checkpoint(self._forward, x, context, objs, clip_length)
        else:
            return self._forward(x, context, objs, clip_length)

    def _forward(self, x, context, objs, clip_length):
        '''
            Spatial-Temporal Self-Attn -> Cross-Frame Gated-Attn -> Modulated Cross-Attn -> FF
        '''
        attention_mask = None
        x = self.attn1(hidden_states=self.norm1(x), attention_mask=attention_mask, clip_length=clip_length) + x 
        x = self.fuser(x, objs, clip_length=clip_length)
        x = self.attn2(self.norm2(x), context, context) + x
        x = self.ff(self.norm3(x)) + x
        
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, key_dim, value_dim, n_heads, d_head, depth=1, fuser_type=None, use_checkpoint=True):
        super().__init__()
        self.in_channels = in_channels
        query_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        
        self.proj_in = nn.Conv2d(in_channels,
                                 query_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(query_dim, key_dim, value_dim, n_heads, d_head, fuser_type, use_checkpoint=use_checkpoint)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(query_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context, objs):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context, objs)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


class SpatialTransformer3D(nn.Module):
    def __init__(self, in_channels, key_dim, value_dim, n_heads, d_head, depth=1, fuser_type=None, use_checkpoint=True):
        super().__init__()
        self.in_channels = in_channels
        query_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        
        self.proj_in = nn.Conv2d(in_channels,
                                 query_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock3D(query_dim, key_dim, value_dim, n_heads, d_head, fuser_type, use_checkpoint=use_checkpoint)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(query_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context, objs):
        # 1. Input
        clip_length = None
        is_video = x.ndim == 5
        if is_video:
            clip_length = x.shape[2]
            x = rearrange(x, "b c f h w -> (b f) c h w")
            # this enables applying different context vectros on different frames (differently optimized null-embeddings)
            if context.shape[0] != clip_length:
                context = context.repeat_interleave(clip_length,0)

        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        # 2. Blocks
        for block in self.transformer_blocks:
            x = block(x, context, objs, clip_length=clip_length)
        
        # 3. Output
        # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)
        output = x + x_in

        if is_video:
            output = rearrange(output, "(b f) c h w -> b c f h w", f=clip_length)

        return output
