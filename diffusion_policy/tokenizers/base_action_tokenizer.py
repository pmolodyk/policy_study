class BaseActionTokenizer:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def fit(self, action_sequences):
        raise NotImplementedError()

    def encode(self, action_sequence):
        raise NotImplementedError()

    def decode(self, token_sequence):
        raise NotImplementedError