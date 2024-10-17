import torch
from .base_action_tokenizer import BaseActionTokenizer

class QuantileActionTokenizer(BaseActionTokenizer):
    # Vocab size is the same for each dimension
    def __init__(self, action_dim, vocab_size, min_q=0.01, max_q=0.99):
        super().__init__(action_dim)
        self.NUM_SPECIAL_TOKENS = 2
        self.ACTIONS_START_TOKEN = vocab_size - 2
        self.EMPTY_ACTION_TOKEN = vocab_size - 1
        
        self.action_dim = action_dim
        self.vocab_size = vocab_size
        self.min_q = min_q
        self.max_q = max_q
        self.num_bins = self.vocab_size - self.NUM_SPECIAL_TOKENS # Active bins for tokens
        self.bin_widths = None
    
    # Fit to training data
    def fit(self, action_sequences):
        assert(len(action_sequences.shape) == 3) # Must be (N x SeqLen x action_dim)
        assert(action_sequences.shape[-1] == self.action_dim)

        self.quantiles = torch.quantile(action_sequences.flatten(0, 1), q=torch.tensor([self.min_q, self.max_q]), dim=-2)
        self.min_quants = self.quantiles[0, :] # (1 x action_dim)
        self.max_quants = self.quantiles[1, :]

        self.bin_widths = (self.max_quants - self.min_quants) / self.num_bins

    # Encode action sequence
    def encode(self, action_sequence):
        assert(len(action_sequence.shape) == 3) # Must be (B x SeqLen x action_dim)
        assert(action_sequence.shape[-1] == self.action_dim)

        device = action_sequence.device
        encoded_sequence = torch.floor((action_sequence - self.min_quants[None, None, :].to(device)) / self.bin_widths[None, None, :].to(device))

        return encoded_sequence
    
    # Decode token sequence back to actions
    def decode(self, token_sequence):
        assert(len(token_sequence.shape) == 3) # Must be (B x SeqLen x action_dim)
        assert(token_sequence.shape[-1] == self.action_dim)

        device = token_sequence.device
        decoded_sequence = token_sequence * self.bin_widths[None, None, :].to(device) + self.min_quants[None, None, :].to(device)

        return decoded_sequence
