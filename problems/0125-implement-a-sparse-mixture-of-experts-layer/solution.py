import numpy as np

def moe(x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, top_k: int) -> np.ndarray:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        top_k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
    import numpy as np

def moe(x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, top_k: int) -> np.ndarray:
    B, L, d = x.shape
    tokens = x.reshape(-1, d)                                   # (T, d)

    # Router: one score per expert per token
    logits = tokens @ Wg                                        # (T, E)

    # Top-k experts per token
    top_idx = np.argsort(logits, axis=-1)[:, ::-1][:, :top_k]   # (T, k)
    top_logits = np.take_along_axis(logits, top_idx, axis=-1)   # (T, k)

    # Softmax over the selected logits only (stable)
    top_logits = top_logits - top_logits.max(axis=-1, keepdims=True)
    gates = np.exp(top_logits)
    gates /= gates.sum(axis=-1, keepdims=True)                  # (T, k)

    # Apply each token's selected experts: x_t · We[i]
    expert_W = We[top_idx]                                      # (T, k, d, d)
    expert_out = np.einsum('td,tkde->tke', tokens, expert_W)    # (T, k, d)

    # Weighted sum over the k experts, then reshape back
    out = np.einsum('tk,tke->te', gates, expert_out)            # (T, d)
    return out.reshape(B, L, d)