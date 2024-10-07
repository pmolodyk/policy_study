# Based on MaskGIT codebase:
# https://github.com/google-research/maskgit/blob/main/maskgit/nets/maskgit_transformer.py
# https://github.com/dome272/MaskGIT-pytorch/blob/main/bidirectional_transformer.py

from typing import Any, Callable, Dict, Iterable, Optional, Text, Tuple, Union

import torch
import torch.nn as nn


LAYERNORM_EPSILON = 1e-12  # Layer norm from BERT

class Attention(nn.Module):
    """Attention layer that is part of each Transformer layer."""

    def __init__(self, 
                 hidden_size: int,
                 num_attention_heads: int,
                 attention_probs_dropout_prob: float):
      self.dim = hidden_size // num_attention_heads
      self.q, self.k, self.v = nn.Linear(hidden_size, self.dim), nn.Linear(hidden_size, self.dim), nn.Linear(hidden_size, self.dim)
      self.attention = nn.MultiheadAttention(num_heads=num_attention_heads,
                                             embed_dim=hidden_size,
                                             dropout=attention_probs_dropout_prob)
      

    def forward(self, x):
        # TODO: need to build the required mask, missing in PyTorch impl
        attention_output = self.attention(query=self.q(x), key=self.k(x), value=self.v(x))

        return attention_output


class Mlp(nn.Module):
    # MLP layer that is part of each Transformer layer
    def __init__(self,
                 intermediate_dim,
                 hidden_dim,
                 mlp_dropout_p):
        self.l1 = nn.Linear(hidden_dim, intermediate_dim)
        self.act1 = nn.GELU()
        self.l2 = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(mlp_dropout_p)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=LAYERNORM_EPSILON)
    
    def forward(self, x):
       layer1_output = self.act1(self.l1(x))
       layer2_output = self.dropout(self.l2(layer1_output))
       return self.layer_norm(layer2_output + x)

class TransformerLayer(nn.Module):
  """A single Transformer layer."""
  intermediate_size: int
  hidden_size: int
  hidden_dropout_prob: float
  num_attention_heads: int
  attention_probs_dropout_prob: float
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, layer_input: jnp.ndarray, input_mask: jnp.ndarray,
               deterministic: bool) -> jnp.ndarray:
    attention_output = Attention(
        hidden_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        num_attention_heads=self.num_attention_heads,
        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        initializer_fn=self.initializer_fn)(
            layer_input=layer_input,
            input_mask=input_mask,
            deterministic=deterministic)

    layer_output = Mlp(
        hidden_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        intermediate_size=self.intermediate_size,
        initializer_fn=self.initializer_fn)(
            attention_output=attention_output, deterministic=deterministic)

    return layer_output


class Embed(nn.Module):
  """Embeds visual tokens."""
  embedding_size: int
  hidden_dropout_prob: float
  vocab_size: int
  max_position_embeddings: int
  initializer_fn: InitializerType
  hidden_size: Optional[int] = None

  @nn.compact
  def __call__(self, input_ids: jnp.ndarray,
               deterministic: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    seq_length = input_ids.shape[-1]
    position_ids = jnp.arange(seq_length)[None, :]

    word_embedder = nn.Embed(
        num_embeddings=self.vocab_size,
        features=self.embedding_size,
        embedding_init=self.initializer_fn,
        name='word_embeddings')
    word_embeddings = word_embedder(input_ids)
    position_embeddings = nn.Embed(
        num_embeddings=self.max_position_embeddings,
        features=self.embedding_size,
        embedding_init=self.initializer_fn,
        name='position_embeddings')(
            position_ids)

    input_embeddings = nn.LayerNorm(
        epsilon=LAYERNORM_EPSILON, name='embeddings_ln')(
            word_embeddings + position_embeddings)
    if self.hidden_size:
      input_embeddings = nn.Dense(
          features=self.hidden_size,
          kernel_init=self.initializer_fn,
          name='embedding_hidden_mapping')(
              input_embeddings)
    input_embeddings = nn.Dropout(rate=self.hidden_dropout_prob)(
        input_embeddings, deterministic=deterministic)

    return input_embeddings


class Bias(nn.Module):
  """Adds a learnable bias to the input.

  Attributes:
    dtype: the dtype of the computation (default: float32).
    bias_init: initializer function for the bias.
  """
  dtype: Any = jnp.float32
  bias_init: Callable[[Any, Tuple[int], Any], Any] = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)

    bias_shape = inputs.shape[-1]
    bias = self.param('bias', self.bias_init, bias_shape)
    bias = jnp.asarray(bias, self.dtype)
    bias = jnp.broadcast_to(bias, inputs.shape)

    return inputs + bias


class MlmLayer(nn.Module):
  """MLM layer for masked token prediction."""
  hidden_size: int
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, last_layer: jnp.ndarray,
               embeddings: jnp.ndarray) -> jnp.ndarray:
    mlm_hidden = nn.Dense(
        features=self.hidden_size,
        kernel_init=self.initializer_fn,
        name='mlm_dense')(
            last_layer)
    mlm_hidden = jax.nn.gelu(mlm_hidden)
    mlm_hidden = nn.LayerNorm(
        epsilon=LAYERNORM_EPSILON, name='mlm_ln')(
            mlm_hidden)
    output_weights = jnp.transpose(embeddings)
    logits = jnp.matmul(mlm_hidden, output_weights)
    logits = Bias(name='mlm_bias')(logits)
    return logits


class Transformer(nn.Module):
  """Transformer modified from BERT."""
  vocab_size: int
  hidden_size: int = 768
  num_hidden_layers: int = 12
  num_attention_heads: int = 12
  intermediate_size: int = 3072
  hidden_dropout_prob: float = 0.1
  attention_probs_dropout_prob: float = 0.1
  max_position_embeddings: int = 256
  initializer_range: float = 0.02

  @nn.compact
  def __call__(self,
               input_ids: jnp.ndarray,
               deterministic: bool = True) -> Dict[Text, jnp.ndarray]:
    input_ids = input_ids.astype('int32')
    input_embeddings = Embed(
        embedding_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        vocab_size=self.vocab_size,
        max_position_embeddings=self.max_position_embeddings,
        initializer_fn=truncated_normal(self.initializer_range))(
            input_ids=input_ids, deterministic=deterministic)

    layer_input = input_embeddings
    for _ in range(self.num_hidden_layers):
      layer_output = TransformerLayer(
          intermediate_size=self.intermediate_size,
          hidden_size=self.hidden_size,
          hidden_dropout_prob=self.hidden_dropout_prob,
          num_attention_heads=self.num_attention_heads,
          attention_probs_dropout_prob=self.attention_probs_dropout_prob,
          initializer_fn=truncated_normal(self.initializer_range))(
              layer_input=layer_input,
              input_mask=jnp.ones_like(input_ids, dtype=jnp.int32),
              deterministic=deterministic)
      layer_input = layer_output

    word_embedding_matrix = self.variables['params']['Embed_0'][
        'word_embeddings']['embedding']
    logits = MlmLayer(
        hidden_size=self.hidden_size,
        initializer_fn=truncated_normal(self.initializer_range))(
            last_layer=layer_output, embeddings=word_embedding_matrix)

    return logits