import math
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from lvdm.common import (
    checkpoint,
    exists,
    default,
)
from lvdm.basics import (
    zero_module,
)

# -----------------------------------------------------------
# Utils modules
# -----------------------------------------------------------

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


# -----------------------------------------------------------
# Transformer blocks
# -----------------------------------------------------------

class CrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., 
                 relative_position=False, temporal_length=None, img_cross_attention=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

        self.image_cross_attention_scale = 1.0
        self.text_context_len = 77
        self.img_cross_attention = img_cross_attention
        if self.img_cross_attention:
            self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.relative_position = relative_position
        if self.relative_position:
            assert(temporal_length is not None)
            self.relative_position_k = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        ## considering image token additionally
        if context is not None and self.img_cross_attention:
            context, context_img = context[:,:self.text_context_len,:], context[:,self.text_context_len:,:]
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_img)
            v_ip = self.to_v_ip(context_img)
        else:
            k = self.to_k(context)
            v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        if self.relative_position:
            len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = einsum('b t d, t s d -> b t s', q, k2) * self.scale # TODO check 
            sim += sim2
        del k

        if exists(mask):
            ## feasible for causal attention mask only
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            sim.masked_fill_(~(mask>0.5), max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        if self.relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            out2 = einsum('b t s, t s d -> b t d', sim, v2) # TODO check
            out += out2
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        ## considering image token additionally
        if context is not None and self.img_cross_attention:
            k_ip, v_ip = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_ip, v_ip))
            sim_ip =  torch.einsum('b i d, b j d -> b i j', q, k_ip) * self.scale
            del k_ip
            sim_ip = sim_ip.softmax(dim=-1)
            out_ip = torch.einsum('b i j, b j d -> b i d', sim_ip, v_ip)
            out_ip = rearrange(out_ip, '(b h) n d -> b n (h d)', h=h)
            out = out + self.image_cross_attention_scale * out_ip
        del q

        return self.to_out(out)


class BasicTransformerBlock(nn.Module):

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                disable_self_attn=False, attention_cls=None, img_cross_attention=False, c_aware=False):
        super().__init__()
        attn_cls = CrossAttention if attention_cls is None else attention_cls
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            img_cross_attention=img_cross_attention)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

        self.c_aware = c_aware
        if self.c_aware:
            self.attn2_out = None

    def forward(self, x, context=None, mask=None):
        ## implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
        input_tuple = (x,)      ## should not be (x), otherwise *input_tuple will decouple x into multiple arguments
        if context is not None:
            input_tuple = (x, context)
        if mask is not None:
            forward_mask = partial(self._forward, mask=mask)
            return checkpoint(forward_mask, (x,), self.parameters(), self.checkpoint)
        if context is not None and mask is not None:
            input_tuple = (x, context, mask)
        return checkpoint(self._forward, input_tuple, self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, mask=None):
        
        # Self-Attention or Temporal-Attention
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask) + x

        # Cross-Attention or Temporal-Attention
        if self.c_aware and context is not None:
            # Saved cross-attention output for Context-Aware Temporal Attention
            self.attn2_out = self.attn2(self.norm2(x), context=context, mask=mask)
            x = self.attn2_out + x
        else:
            x = self.attn2(self.norm2(x), context=context, mask=mask) + x
        
        # FFN
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data in spatial axis.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None,
                 use_checkpoint=True, disable_self_attn=False, use_linear=False, img_cross_attention=False, c_aware=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                img_cross_attention=img_cross_attention,
                disable_self_attn=disable_self_attn,
                checkpoint=use_checkpoint,
                c_aware=c_aware
            ) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

        self.c_aware = c_aware
        if self.c_aware:
            self.attn2_out = None

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x

        # proj in
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        # transformer (1block)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context)
            if self.c_aware:
                self.attn2_out = block.attn2_out # (bt, l, c)
        
        # proj out
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        
        return x + x_in
    
    
class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None,
                 use_checkpoint=True, use_linear=False, only_self_att=True, causal_attention=False,
                 relative_position=False, temporal_length=None, c_aware=False):
        super().__init__()

        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.causal_attention = causal_attention
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        if c_aware:
            attention_cls = ContextAwareTmpAttention
            self.attn2_out = None
        else:
            attention_cls = None

        if self.only_self_att: # True
            context_dim = None
        else:
            raise NotImplementedError("TemporalTransformer only supports self-attention now.")
        
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock( 
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim, # None
                attention_cls=attention_cls, # None or ContextAwareTmpAttention
                checkpoint=use_checkpoint) for d in range(depth)
        ])

        if not use_linear:
            self.proj_out = zero_module(nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

        self.c_aware = c_aware

    def forward(self, x, context=None):
        b, c, t, h, w = x.shape
        x_in = x

        # proj in
        x = self.norm(x)
        x = rearrange(x, 'b c t h w -> (b h w) c t').contiguous()
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'bhw c t -> bhw t c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        # prepare mask for context-aware temporal attention
        if self.c_aware:
            mask = self.attn2_out # set in openaimodel3d
            mask = rearrange(mask, '(b t) l c -> (b l) t c', b=b, t=t).contiguous()
        else:
            mask = None

        # transformer (1block)    
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, mask=mask) # no context
        x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
        
        # proj out
        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, 'b (h w) t c -> b c t h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = rearrange(x, 'b hw t c -> (b hw) c t').contiguous()
            x = self.proj_out(x)
            x = rearrange(x, '(b h w) c t -> b c t h w', b=b, h=h, w=w).contiguous()

        return x + x_in


class ContextAwareTmpAttention(nn.Module):
    """
    Context-Aware Temporal Attention
    same arguments as CrossAttention
    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., 
                relative_position=False, temporal_length=None, img_cross_attention=False, zero_diagonal=False):
        super().__init__()

        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim) # should be query_dim

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

        self.text_context_len = 77

        self.beta = nn.Parameter(torch.zeros(self.heads), requires_grad=True) # Context-Aware Bias
        self.mask_layernorm = nn.LayerNorm(query_dim)
        self.zero_diagonal = zero_diagonal

    def forward(self, x, context=None, mask=None):
        """
        x: (b*l, t, c) for temporal attention
        context: None (self-attention)
        mask: (b*l, t, c) attention output from cross-attention in spatial transformer
        if no mask is given, it degrades to normal temporal attention
        """
        
        h = self.heads

        q = self.to_q(x) # (b*l, t, c) -> (b*l, t, h*d)
        context = default(context, x) # x
        k = self.to_k(context) # (b*l, t, c) -> (b*l, t, h*d)
        v = self.to_v(context) # (b*l, t, c) -> (b*l, t, h*d)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale # (b*l*h, t, t)

        del k
        del q

        # Context-Aware Bias
        if exists(mask):
            
            # LayerNorm
            mask = self.mask_layernorm(mask) # (b*l, t, c)

            # compute symmetric attention matrix
            symmetric_mask = mask @ mask.transpose(-1, -2) / math.sqrt(mask.shape[-1]) # (b*l, t, c) @ (b*l, c, t) -> (b*l, t, t)
            
            # zero diagonal (False)
            if self.zero_diagonal: 
                symmetric_mask = symmetric_mask - torch.diag_embed(torch.diagonal(symmetric_mask, dim1=-2, dim2=-1)) 

            # repeat for heads
            symmetric_mask = repeat(symmetric_mask, 'b i j -> (b h) i j', h=h) # (bl*h, t, t)

            # prepare beta
            expanded_beta = self.beta.unsqueeze(1).unsqueeze(2) # (h, 1, 1)
            expanded_beta = repeat(expanded_beta, 'h i j -> (b h) i j', b=mask.shape[0]) # (h,1,1) -> (bl*h,1,1)

            # add Context-Aware Bias
            symmetric_mask = symmetric_mask * expanded_beta # (bl*h, t, t)
            sim = sim + symmetric_mask # zero-init

        # attention
        sim = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)

# -----------------------------------------------------------
# Contextualizer (4 layers of ContextualizerBlock)
# -----------------------------------------------------------

class Contextualizer(nn.Module):
    """
    Contextualizer: 4 layers of ContextualizerBlock
    """
    def __init__(self, dim, n_heads, d_head, improve=False):
        super().__init__()
        
        self.dim = dim
        self.n_heads = n_heads
        self.d_head = d_head

        self.block1 = ContextualizerBlock(dim, n_heads, d_head, improve=improve)
        self.block2 = ContextualizerBlock(dim, n_heads, d_head, improve=improve)
        self.block3 = ContextualizerBlock(dim, n_heads, d_head, improve=improve)
        self.block4 = ContextualizerBlock(dim, n_heads, d_head, last=True, improve=improve)

        self.improve = improve
        if self.improve:
            self.time_embed = nn.Sequential(
                nn.Linear(4*dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )

    def forward(self, x, num_frames):
        x_in = x

        if self.improve:
            # add positional embedding
            b_t, l, c = x.shape # c=dim
            b = b_t // num_frames
            device = x.device
            pe = build_sincos_position_embedding(num_frames, 4*c, device)
            pe = self.time_embed(pe)
            pe = repeat(pe, 't c -> (b t) c', b=b).unsqueeze(1) # (b*t, 1, c)
            x = x + pe 

        x = self.block1(x, num_frames=num_frames)
        x = self.block2(x, num_frames=num_frames)
        x = self.block3(x, num_frames=num_frames)
        x = self.block4(x, num_frames=num_frames) # zero-initialized

        x = x + x_in
        return x

class ContextualizerBlock(nn.Module):
    """
    x = context: (b*t,l,c)
    """
    def __init__(self, dim, n_heads, d_head, last=False, improve=False):
        super().__init__()
        self.last = last
        self.improve = improve

        # Self-Attention
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, context_dim=None)
        self.attn2 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, context_dim=None)

        # FFN
        self.ff1 = FeedForward(dim, glu=True)
        if last:
            self.ff2 = zero_module(FeedForward(dim, glu=True))
        else:
            self.ff2 = FeedForward(dim, glu=True)

        # Pre-LN
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

    def forward(self, x, num_frames):
        b_t, l, c = x.shape
        b = b_t // num_frames

        # Self-Attention + FFN
        x = rearrange(x, '(b t) l c -> b (t l) c', b=b, t=num_frames)
        x = self.attn1(self.norm1(x)) + x
        x = self.ff1(self.norm2(x)) + x
 
        if self.improve:
            # Adjacent-Attention
            device = x.device
            mask = build_local_mask(num_frames, radius=1, device=device) # [t, t]
            mask = mask.repeat_interleave(l, dim=0).repeat_interleave(l, dim=1)  # (t*l, t*l)
            mask = mask.unsqueeze(0).repeat(b, 1, 1) # [b, tl, tl]
            x = self.attn2(self.norm3(x), mask=mask) + x # (b, tl, c)
            x = rearrange(x, 'b (t l) c -> (b t) l c', b=b, t=num_frames)
        else:
            # Temporal-Attention
            x = rearrange(x, 'b (t l) c -> (b l) t c', b=b, t=num_frames)
            x = self.attn2(self.norm3(x)) + x
            x = rearrange(x, '(b l) t c -> (b t) l c', b=b, t=num_frames)

        # FFN
        if self.last:
            x = self.ff2(self.norm4(x)) # connect to input
        else:
            x = self.ff2(self.norm4(x)) + x
        
        return x

def build_sincos_position_embedding(n_pos: int, dim: int, device: torch.device):
    """(n_pos, dim) sinusoidal positional embedding (token-wise)."""
    position = torch.arange(n_pos, device=device).float().unsqueeze(1)  # (n,1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-(math.log(10000.0) / dim)))
    pe = torch.zeros(n_pos, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (n_pos, dim)

def build_local_mask(t: int, radius: int = 1, device=None):
    # True=通す, False=切る（あなたのCrossAttentionは mask>0.5 で通す実装）
    ar = torch.arange(t, device=device)
    mask = (ar[:, None] - ar[None, :]).abs() <= radius   # [t, t] boolean
    return mask.float()   # 後段で repeat して使う



# -----------------------------------------------------------
# Other attention modules (not used)
# -----------------------------------------------------------

class RelativePosition(nn.Module):
    """ https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py """

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings


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


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_
