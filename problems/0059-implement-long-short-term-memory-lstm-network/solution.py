
import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        # Initialize biases
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x, initial_hidden_state, initial_cell_state):

        h = initial_hidden_state
        c = initial_cell_state
        hidden_states = []

        # x has shape (T, D)
        for t in range(x.shape[0]):

            # Extract current timestep: (D, 1)
            xt = x[t, :].reshape(-1, 1)

            # Concatenate hidden state and input
            combined = np.vstack((h, xt))

            # Forget gate
            f = self.sigmoid(self.Wf @ combined + self.bf)

            # Input gate
            i = self.sigmoid(self.Wi @ combined + self.bi)

            # Candidate cell state
            c_tilde = self.tanh(self.Wc @ combined + self.bc)

            # Output gate
            o = self.sigmoid(self.Wo @ combined + self.bo)

            # Update cell state
            c = f * c + i * c_tilde

            # Update hidden state
            h = o * self.tanh(c)

            hidden_states.append(h.copy())

        # Shape: (T, H, 1)
        hidden_states = np.stack(hidden_states, axis=0)

        return hidden_states, h, c