# Based on MaskGIT codebase:
# https://github.com/google-research/maskgit/blob/main/maskgit/nets/maskgit_transformer.py
# https://github.com/dome272/MaskGIT-pytorch/blob/main/bidirectional_transformer.py
import torch
import torch.nn as nn


LAYERNORM_EPSILON = 1e-12  # Layer norm from BERT

class Attention(nn.Module):
    """Attention layer that is part of each Transformer layer."""

    def __init__(self, 
                 hidden_dim: int,
                 num_attention_heads: int,
                 attention_probs_dropout_prob: float):
      self.dim = hidden_dim // num_attention_heads
      self.q, self.k, self.v = nn.Linear(hidden_dim, self.dim), nn.Linear(hidden_dim, self.dim), nn.Linear(hidden_dim, self.dim)
      self.attention = nn.MultiheadAttention(num_heads=num_attention_heads,
                                             embed_dim=hidden_dim,
                                             dropout=attention_probs_dropout_prob)
      

    def forward(self, x, attn_mask):
        attention_output = self.attention(query=self.q(x), key=self.k(x), value=self.v(x), attn_mask=attn_mask)
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
    def __init__(self,
                 intermediate_dim,
                 hidden_dim,
                 mlp_dropout_prob,
                 attn_dropout_prob,
                 num_attn_heads):
       self.attention = Attention(hidden_dim=hidden_dim,
                                  num_attention_heads=num_attn_heads,
                                  attention_probs_dropout_prob=attn_dropout_prob)
       self.mlp = Mlp(intermediate_dim=intermediate_dim,
                      hidden_dim=hidden_dim,
                      mlp_dropout_p=mlp_dropout_prob)
    
    def forward(self, x, attn_mask):
        attention_output = self.attention(x, attn_mask)
        layer_output = self.mlp(attention_output)
        return layer_output


class Embedding(nn.Module):
    def __init__(self,
                 embedding_dim,
                 vocab_size,
                 max_pos_embed,
                 dropout_p):
        self.input_embedding = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embedding_dim)
        self.positional_embedding = nn.Embedding(num_embeddings=max_pos_embed,
                                                 embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=LAYERNORM_EPSILON)
    
    def forward(self, x):
       content_embeddings = self.input_embedding(x)
       position_ids = torch.range(x.shape[-1]).unsqueeze(0)
       position_embeddings = self.positional_embedding(position_ids)

       input_embeddings = self.layer_norm(content_embeddings + position_embeddings)
       embedded_output = self.dropout(input_embeddings)
       
       return embedded_output

class BiasLayer(torch.nn.Module):
    def __init__(self):
        self.bias_param = torch.nn.Parameter(torch.zeros((1)))
    
    def forward(self, x):
        return x + self.bias_param

class MlmLayer(nn.Module):
    """MLM layer for masked token prediction."""
    def __init__(self, hidden_dim):
        self.layer = nn.Linear()
        self.act = nn.GELU()
        self.bias = BiasLayer()
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=LAYERNORM_EPSILON)
    
    def forward(self, x, embedding_table):
       layer_output = self.layer_norm(self.act(self.layer(x)))
       logits = self.bias(layer_output @ embedding_table.T)
       return logits

class Transformer(nn.Module):
    def __init__(self,
                vocab_size,
                hidden_dim = 768,
                num_hidden_layers = 12,
                num_attention_heads = 12,
                intermediate_dim = 3072,
                mlp_dropout_prob = 0.1,
                attn_dropout_prob = 0.1,
                max_position_embed = 256):
        self.embedding = Embedding(embedding_dim=hidden_dim,
                                  vocab_size=vocab_size,
                                  max_pos_embed=max_position_embed,
                                  dropout_p=mlp_dropout_prob)
        self.transformer_layers = []
        for _ in range(num_hidden_layers):
            self.transformer_layers.append(TransformerLayer(intermediate_dim=intermediate_dim,
                                                            hidden_dim=hidden_dim,
                                                            mlp_dropout_prob=mlp_dropout_prob,
                                                            attn_dropout_prob=attn_dropout_prob,
                                                            num_attn_heads=num_attention_heads))
        self.mlm = MlmLayer(hidden_dim=hidden_dim)
    
    def forward(self, input_ids):
        layer_input = self.embedding(input_ids)

        for transformer_layer in self.transformer_layers:
            layer_input = transformer_layer(layer_input, attn_mask = torch.ones(layer_input.shape, dtype=torch.int32))
        
        embedding_table = self.embedding.weight.data
        logits = self.mlm(layer_input, embedding_table=embedding_table)

        return logits