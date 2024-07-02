from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, hidden_size, num_attention_heads, attention_head_dim, attention_dropout=0.0):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim

        self.scale = self.head_dim**-0.5
        self.dropout = attention_dropout

        self.inner_dim = self.head_dim * self.num_heads

        self.k_proj = nn.Linear(self.embed_dim, self.inner_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.inner_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.inner_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, mult=4):
        super().__init__()
        self.activation_fn = nn.SiLU()
        self.fc1 = nn.Linear(hidden_size, intermediate_size * mult)
        self.fc2 = nn.Linear(intermediate_size * mult, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Transformer(nn.Module):
    def __init__(self, depth=12):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(depth)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        causal_attention_mask: torch.Tensor = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
            )

        return hidden_states


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_attention_heads=12,
        attention_head_dim=64,
        attention_dropout=0.0,
        dropout=0.0,
        eps=1e-5,
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.self_attn = Attention(
            hidden_size=hidden_size, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=eps)
        self.mlp = MLP(hidden_size=hidden_size, intermediate_size=hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        causal_attention_mask: torch.Tensor = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs[0]


class DiffusionTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_attention_heads=12,
        attention_head_dim=64,
        attention_dropout=0.0,
        dropout=0.0,
        eps=1e-5,
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.self_attn = Attention(
            hidden_size=hidden_size, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=eps)
        self.mlp = MLP(hidden_size=hidden_size, intermediate_size=hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=eps)
        self.output_token = nn.Parameter(torch.randn(1, hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        causal_attention_mask: torch.Tensor = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        output_token = self.output_token.unsqueeze(0).repeat(hidden_states.shape[0], 1, 1)
        hidden_states = torch.cat([output_token, hidden_states], dim=1)
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs[0][:, 0:1, ...]


class V2AMapperMLP(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, expansion_rate=4):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim * expansion_rate)
        self.silu = nn.SiLU()
        self.layer_norm = nn.LayerNorm(input_dim * expansion_rate)
        self.linear2 = nn.Linear(input_dim * expansion_rate, output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.silu(x)
        x = self.layer_norm(x)
        x = self.linear2(x)

        return x


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

        self.zero_initialize_last_layer()

    def zero_initialize_last_layer(module):
        last_layer = None
        for module_name, layer in module.named_modules():
            if isinstance(layer, torch.nn.Linear):
                last_layer = layer

        if last_layer is not None:
            last_layer.weight.data.zero_()
            last_layer.bias.data.zero_()

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class VisionAudioAdapter(torch.nn.Module):
    def __init__(
        self,
        embedding_size=768,
        expand_dim=4,
        token_num=4,
    ):
        super().__init__()

        self.mapper = V2AMapperMLP(
            embedding_size,
            embedding_size,
            expansion_rate=expand_dim,
        )

        self.proj = ImageProjModel(
            cross_attention_dim=embedding_size,
            clip_embeddings_dim=embedding_size,
            clip_extra_context_tokens=token_num,
        )

    def forward(self, image_embeds):
        image_embeds = self.mapper(image_embeds)
        image_embeds = self.proj(image_embeds)
        return image_embeds
