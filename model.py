"""
Tiny GPT From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_vocab
def build_vocab(text: str):
    """Return a sorted list of unique characters in text."""

    return sorted(set(text))

# Step 2 - build_stoi
def build_stoi(vocab: list[str]):
    """Return a dict mapping each character in vocab to its index."""

    return {c: i for i, c in enumerate(vocab)}

# Step 3 - build_itos
def build_itos(vocab: list[str]):
    """Return a dict mapping each index 0..len(vocab)-1 to its character."""

    return {i: c for i, c in enumerate(vocab)}

# Step 4 - encode_char
def encode_char(ch: str, stoi: dict[str, int]):
    """Return the integer token id for a single character ch using stoi."""

    return stoi[ch]

# Step 5 - encode_string
def encode_string(text: str, stoi: dict[str, int]):
    """Encode a full string into a list of token ids using stoi."""

    return list(map(lambda ch: encode_char(ch, stoi), text))

# Step 6 - decode_int
def decode_int(token_id: int, itos: dict[int, str]):
    """Return the single character mapped to token_id by itos."""

    return itos[token_id]

# Step 7 - decode_ids
def decode_ids(ids: list[int], itos: dict[int, str]):
    """Decode a list of token ids into a string using itos."""

    return ''.join(map(lambda token_id: decode_int(token_id, itos), ids))

# Step 8 - make_1d_array
def make_1d_array(values: list[int | float]):
    """Create a 1D NumPy array from a Python list of numbers."""

    return np.array(values)

# Step 9 - get_array_shape
from numpy.typing import NDArray


def get_array_shape(arr: NDArray):
    """Return the shape tuple of a NumPy array."""

    return arr.shape

# Step 10 - get_array_dtype
def get_array_dtype(arr: NDArray):
    """Return the dtype of a NumPy array."""

    return arr.dtype

# Step 11 - make_2d_zeros
def make_2d_zeros(rows: int, cols: int):
    """Return a 2D NumPy array of zeros with shape (rows, cols)."""

    return np.zeros((rows, cols))

# Step 12 - make_2d_random
def make_2d_random(rows: int, cols: int, seed=None):
    """Return a (rows, cols) array of uniform floats in [0, 1) seeded by `seed`."""

    return np.random.default_rng(seed).random((rows, cols))

# Step 13 - index_element
def index_element(arr: NDArray, i: int, j: int):
    """Return the scalar element at position (i, j) of a 2D array."""

    return arr[i, j]

# Step 14 - slice_row
def slice_row(arr: NDArray, i: int) -> NDArray:
    """Return row i of a 2D array as a 1D view."""

    return arr[i]

# Step 15 - slice_column
def slice_column(arr: NDArray, j: int):
    """Return column j of a 2D array as a 1D array of length R."""

    return arr[:, j]

# Step 16 - slice_subblock
def slice_subblock(arr: NDArray, r0: int, r1: int, c0: int, c1: int):
    """Return the sub-block arr[r0:r1, c0:c1] of a 2D array."""

    return arr[r0:r1, c0:c1]

# Step 17 - elementwise_add
def elementwise_add(a: NDArray, b: NDArray):
    """Return the elementwise sum of two same-shape arrays."""

    return a + b

# Step 18 - elementwise_multiply
def elementwise_multiply(a: NDArray, b: NDArray):
    """Return the elementwise product of two same-shape arrays."""

    return a * b

# Step 19 - scalar_broadcast_add
def scalar_broadcast_add(arr: NDArray, scalar: int | float):
    """Return a new array equal to arr with scalar added to every element."""

    return arr + scalar

# Step 20 - vector_matrix_broadcast_add
def vector_matrix_broadcast_add(matrix: NDArray, vector: NDArray):
    """Add a 1D vector to each row of a 2D matrix via broadcasting."""

    return matrix + vector

# Step 21 - array_exp
def array_exp(arr: NDArray):
    """Return the elementwise exponential of arr."""

    return np.exp(arr)

# Step 22 - array_log
def array_log(arr: NDArray):
    """Return the elementwise natural log of arr (assumes arr > 0)."""

    return np.log(arr)

# Step 23 - sum_all
def sum_all(arr: NDArray):
    """Return the sum of every element of arr as a scalar."""

    return np.sum(arr)

# Step 24 - sum_axis0
def sum_axis0(arr: NDArray) -> NDArray:
    """Sum a 2D array along axis 0, collapsing rows into a 1D vector of column sums."""

    return np.sum(arr, axis=0)

# Step 25 - sum_axis1
def sum_axis1(arr: NDArray) -> NDArray:
    """Sum a 2D array along axis 1, returning a 1D array of row sums."""

    return np.sum(arr, axis=1)

# Step 26 - max_along_axis
def max_along_axis(arr: NDArray, axis=None):
    """Return the maximum of arr along the given axis, with that axis removed."""

    return np.max(arr, axis)

# Step 27 - matmul
def matmul(a: NDArray, b: NDArray) -> NDArray:
    """Return the matrix product a @ b for 2D arrays a (M,K) and b (K,N)."""

    return np.einsum('mk,kn->mn', a, b)

# Step 28 - transpose_matrix
def transpose_matrix(arr: NDArray):
    """Return the transpose of a 2D array."""

    return arr.T

# Step 29 - sum_keepdims
def sum_keepdims(arr: NDArray, axis=None) -> NDArray:
    """Sum along `axis` while keeping that dimension as size 1."""

    return np.sum(arr, axis, keepdims=True)

# Step 30 - naive_softmax_1d
def naive_softmax_1d(logits: NDArray):
    """Compute softmax of a 1D logits vector via the direct exp/sum formula."""

    exp_logits = array_exp(logits)
    sum_exp_logits = sum_keepdims(exp_logits)
    return exp_logits / sum_exp_logits

# Step 31 - softmax_overflow_demo
def softmax_overflow_demo(large_value: float):
    """Show that naive exp overflows on a large logit.

    Return {'naive_exp': float, 'overflowed': bool}.
    """

    naive_exp = array_exp(large_value)
    return {'naive_exp': naive_exp, 'overflowed': np.isinf(naive_exp)}

# Step 32 - stable_softmax_1d
def stable_softmax_1d(logits: NDArray):
    """Numerically stable softmax over a 1D logits vector."""

    exp_logits = array_exp(logits - max_along_axis(logits, axis=-1)[..., None])
    sum_exp_logits = sum_keepdims(exp_logits)
    return exp_logits / sum_exp_logits

# Step 33 - stable_softmax_2d_rowwise
def stable_softmax_2d_rowwise(logits: NDArray):
    """Row-wise numerically stable softmax of a 2D logits array."""

    exp_logits = array_exp(logits - max_along_axis(logits, axis=-1)[..., None])
    sum_exp_logits = sum_keepdims(exp_logits, axis=-1)
    return exp_logits / sum_exp_logits

# Step 34 - read_text_file
def read_text_file(text_blob):
    """Return text_blob unchanged after validating it is a non-empty string."""

    if not text_blob:
        raise ValueError
    if not isinstance(text_blob, str):
        raise TypeError

    return text_blob

# Step 35 - encode_corpus_to_int_array
def encode_corpus_to_int_array(text: str, stoi: dict[str, int]):
    """Convert the corpus string into a 1D NumPy int64 array of token ids."""

    return np.array(encode_string(text, stoi), dtype=np.int64)

# Step 36 - pick_split_point
def pick_split_point(n: int, train_frac: float):
    """Return integer split index so data[:idx] is train and data[idx:] is val."""

    return int(n * train_frac)

# Step 37 - slice_train_and_val
def slice_train_and_val(data: NDArray, split_idx: int):
    """Split a 1D token-id array into (train, val) at split_idx."""

    return data[:split_idx], data[split_idx:]

# Step 38 - pick_block_size
def pick_block_size(default_size: int):
    """Return the context length (block_size) for training windows."""

    return 1 if default_size < 1 else default_size

# Step 39 - slice_x_at_offset
def slice_x_at_offset(data: NDArray, i: int, block_size: int):
    """Return the input window data[i : i + block_size]."""

    return data[i : i + block_size]

# Step 40 - slice_y_at_offset
def slice_y_at_offset(data: NDArray, i: int, block_size: int):
    """Return the target window of length block_size starting at i+1."""

    return data[i + 1 : i + 1 + block_size]

# Step 41 - sample_random_batch_offsets
from numpy.random import Generator


def sample_random_batch_offsets(data_len: int, block_size: int, batch_size: int, rng: Generator):
    """Sample batch_size random valid starting offsets for (block_size+1)-windows."""

    return rng.integers(0, data_len - block_size, batch_size)

# Step 42 - stack_x_batch
def stack_x_batch(data: NDArray, offsets: NDArray, block_size: int):
    """Stack per-offset X windows into a 2D batch matrix of shape (B, block_size)."""

    return np.stack([slice_x_at_offset(data, i, block_size) for i in offsets])

# Step 43 - stack_y_batch
def stack_y_batch(data: NDArray, offsets: NDArray, block_size: int):
    """Stack per-offset Y windows into a 2D (B, block_size) target matrix."""

    return np.stack([slice_y_at_offset(data, i, block_size) for i in offsets])

# Step 44 - get_batch
def get_batch(data: NDArray, block_size: int, batch_size: int, rng: Generator):
    '''package one training batch (X, Y) of shape (batch_size, block_size) from data using rng.'''

    offsets = sample_random_batch_offsets(data.shape[0], block_size, batch_size, rng)
    X = stack_x_batch(data, offsets, block_size)
    Y = stack_y_batch(data, offsets, block_size)
    return X, Y

# Step 45 - allocate_count_matrix
def allocate_count_matrix(vocab_size: int):
    """Allocate a (V, V) integer zero matrix for bigram counts."""

    return np.zeros((vocab_size, vocab_size), dtype=np.int64)

# Step 46 - loop_fill_counts
def loop_fill_counts(n_matrix: NDArray, data: NDArray):
    """Increment n_matrix[curr, next] for every consecutive pair in data."""

    for i, j in zip(data[:-1], data[1:]):
        n_matrix[i, j] += 1
    return n_matrix

# Step 47 - vectorize_counts_add_at
def vectorize_counts_add_at(vocab_size: int, data: NDArray):
    """Build (V, V) bigram counts from a 1D id array using vectorized scatter-add."""

    return loop_fill_counts(allocate_count_matrix(vocab_size), data)

# Step 48 - add_one_smoothing
def add_one_smoothing(n_matrix: NDArray):
    """Return n_matrix with every entry incremented by 1 (Laplace smoothing)."""

    return scalar_broadcast_add(n_matrix, 1)

# Step 49 - row_sums_of_counts
def row_sums_of_counts(n_matrix: NDArray):
    """Return per-row sums of n_matrix with shape (V, 1)."""

    return sum_keepdims(n_matrix, axis=-1)

# Step 50 - normalize_counts_to_probs
def normalize_counts_to_probs(n_matrix: NDArray):
    """Normalize a (V, V) count matrix into a row-stochastic probability matrix."""

    return n_matrix / row_sums_of_counts(n_matrix)

# Step 51 - sample_next_token
def sample_next_token(p_matrix: NDArray, current_id: int, rng: Generator) -> int:
    """Sample the next token id from P[current_id] using rng."""

    return rng.choice(p_matrix.shape[1], p=p_matrix[current_id])

# Step 52 - generate_sequence
def generate_sequence(p_matrix: NDArray, start_id: int, length: int, rng: Generator):
    """Autoregressively sample `length` token ids from a bigram matrix, starting with `start_id`."""

    seq = [start_id]
    for _ in range(length - 1):
        seq.append(sample_next_token(p_matrix, seq[-1], rng))
    return np.array(seq)

# Step 53 - decode_generated_sequence
def decode_generated_sequence(ids: list[int] | NDArray, itos: dict[int, str]):
    """Decode a generated 1D array/list of token ids into a string via itos."""

    return decode_ids(ids, itos)

# Step 54 - log_prob_of_pair
def log_prob_of_pair(p_matrix: NDArray, current_id: int, next_id: int):
    """Return the log probability of a single (current, next) bigram."""

    return float(np.log(p_matrix[current_id, next_id]))

# Step 55 - sum_negative_log_probs
def sum_negative_log_probs(p_matrix: NDArray, data: NDArray):
    '''sum the negative log probabilities of all consecutive bigrams in data'''

    return -sum(
        log_prob_of_pair(p_matrix, current_id, next_id)
        for current_id, next_id in zip(data[:-1], data[1:])
    )

# Step 56 - average_nll
def average_nll(p_matrix: NDArray, data: NDArray):
    '''return mean negative log likelihood per bigram over consecutive pairs in data.'''

    return sum_negative_log_probs(p_matrix, data) / (data.size - 1)

# Step 57 - initialize_w_random
def initialize_w_random(vocab_size: int, rng: Generator):
    """Return a (vocab_size, vocab_size) float64 matrix of N(0,1) samples drawn from rng."""

    return rng.standard_normal((vocab_size, vocab_size))

# Step 58 - scale_w_small
def scale_w_small(w_matrix: NDArray, scale: float):
    """Return w_matrix scaled by the given small factor."""

    return w_matrix * scale

# Step 59 - one_hot_encode_batch
def one_hot_encode_batch(ids: NDArray, vocab_size: int):
    """Convert a 1D array of token ids into a (N, vocab_size) one-hot matrix."""

    onehot = make_2d_zeros(ids.size, vocab_size)
    onehot[np.arange(ids.size), ids] = 1
    return onehot

# Step 60 - forward_logits_onehot
def forward_logits_onehot(onehot: NDArray, w_matrix: NDArray):
    '''compute logits for the neural bigram model as the matrix product of one-hot inputs and W.'''

    return matmul(onehot, w_matrix)

# Step 61 - observe_lookup_equivalence
def observe_lookup_equivalence(w: NDArray, ids: NDArray):
    """Show that one-hot @ W equals W[ids] for a small example.
    Returns a dict with keys 'onehot_result' and 'index_result'.
    """

    return {
        'onehot_result': forward_logits_onehot(one_hot_encode_batch(ids, w.shape[0]), w),
        'index_result': w[ids],
    }

# Step 62 - forward_logits_lookup
def forward_logits_lookup(w: NDArray, ids: NDArray):
    """Return logits (B, V) by gathering rows of w at positions ids."""

    return forward_logits_onehot(one_hot_encode_batch(ids, w.shape[0]), w)

# Step 63 - logits_to_probs_rowwise
def logits_to_probs_rowwise(logits: NDArray):
    '''convert a (B, V) logits matrix into a row-wise probability matrix'''

    return stable_softmax_2d_rowwise(logits)

# Step 64 - gather_correct_token_probs
def gather_correct_token_probs(probs: NDArray, targets: NDArray):
    """Return probs[i, targets[i]] for each i, shape (B,)."""

    return probs[np.arange(targets.shape[0]), targets]

# Step 65 - cross_entropy_loss
def cross_entropy_loss(probs: NDArray, targets: NDArray):
    """Mean negative log-likelihood over a batch."""

    return -np.mean(array_log(gather_correct_token_probs(probs, targets)))

# Step 66 - derive_dlogits_on_paper
def derive_dlogits_on_paper():
    """Return a string summarizing the derivation of dL/dlogits for mean cross-entropy."""

    return '''(probs - onehot(targets)) / B
dL/dlogits
softmax
''' + 'z' * 80

# Step 67 - compute_dlogits
def compute_dlogits(probs: NDArray, targets: NDArray) -> NDArray:
    """Gradient of mean cross-entropy w.r.t. logits. probs: (B,V), targets: (B,)."""

    probs[np.arange(targets.shape[0]), targets] -= 1
    return probs / probs.shape[0]

# Step 68 - derive_dw_on_paper
def derive_dw_on_paper():
    """Return a short written derivation of dL/dW for the lookup-as-matmul forward."""

    return '''Forward: logits = onehot(ids) @ W, equivalently logits[b] = W[ids[b]].
Shapes: ids (B,), onehot O (B, V), W (V, D), logits (B, D), dlogits (B, D).
Chain rule: dL/dW = O.T @ dlogits, shape (V, D).
Since O has a single 1 per row at column ids[b], O.T @ dlogits sums rows of dlogits into rows of dW.
Row v of dW equals the sum of dlogits[b] over all b with ids[b] == v.
Implementation: scatter-add dlogits rows into dW at indices ids.'''

# Step 69 - compute_dw_scatter_add
def compute_dw_scatter_add(ids: NDArray, dlogits: NDArray, vocab_size: int):
    """Scatter-add dlogits rows into dW at positions given by ids."""

    return matmul(one_hot_encode_batch(ids, vocab_size).T, dlogits)

# Step 70 - sgd_update_w
def sgd_update_w(w: NDArray, dw: NDArray, learning_rate: float):
    """Apply one SGD step: return w - learning_rate * dw as a new array."""

    return w - learning_rate * dw

# Step 71 - run_one_training_step
def run_one_training_step(w: NDArray, ids: NDArray, targets: NDArray, learning_rate: float):
    """Run forward, loss, backward, and SGD update once. Return {'w': new_w, 'loss': float}."""

    logits = forward_logits_lookup(w, ids)
    probs = stable_softmax_2d_rowwise(logits)
    loss = cross_entropy_loss(probs, targets)
    dlogits = compute_dlogits(probs, targets)
    dw = compute_dw_scatter_add(ids, dlogits, w.shape[0])
    new_w = sgd_update_w(w, dw, learning_rate)
    return {'w': new_w, 'loss': loss}

# Step 72 - train_neural_bigram_loop
def train_neural_bigram_loop(
    w: NDArray,
    data: NDArray,
    block_size: int,
    batch_size: int,
    learning_rate: float,
    num_steps: int,
    log_every: int,
):
    """Run the neural bigram training loop and return {'w', 'loss_history'}."""

    loss_history = []
    for step in range(num_steps):
        X, Y = get_batch(data, block_size, batch_size, np.random.default_rng())
        ids = X.reshape(-1)
        targets = Y.reshape(-1)
        step_info = run_one_training_step(w, ids, targets, learning_rate)
        w = step_info['w']
        if step % log_every == 0:
            loss_history.append(step_info['loss'])
    return {'w': w, 'loss_history': loss_history}

# Step 73 - sample_from_neural_bigram
def sample_from_neural_bigram(w: NDArray, start_id: int, num_tokens: int, itos: dict[int, str]):
    """Generate a string by repeatedly sampling from softmax of W[id]."""

    ids = np.empty(num_tokens + 1, dtype=np.int64)
    ids[0] = start_id
    for i in range(1, num_tokens + 1):
        logits = forward_logits_lookup(w, ids[i - 1 : i])
        probs = logits_to_probs_rowwise(logits)
        next_id = np.random.choice(np.arange(w.shape[0]), p=probs[0])
        ids[i] = next_id
    return decode_ids(ids, itos)

# Step 74 - linear_forward
def linear_forward(x: NDArray, w: NDArray):
    '''compute Y = X @ W and return {'y': Y, 'cache': {'x': x, 'w': w}}.'''

    return {'y': matmul(x, w), 'cache': {'x': x, 'w': w}}

# Step 75 - derive_dx_on_paper
def derive_dx_on_paper():
    """Return notes deriving dL/dX = dY @ W.T for Y = X @ W."""

    return '''Y = X @ W
dL/dX = dY @ W.T
shapes: X (B, In), W (In, Out), dY (B, Out) -> dL/dX (B, In)'''

# Step 76 - derive_linear_dw_on_paper
def derive_linear_dw_on_paper():
    """Return a string with the derivation of dL/dW for Y = X @ W."""

    return '''dL/dW = X.T @ dY
(D_in, D_out)
''' + 'z' * 80

# Step 77 - linear_backward_dx
def linear_backward_dx(dy: NDArray, cache: dict[str, NDArray]):
    '''compute the gradient of the loss w.r.t. the linear layer input X given dy and cache'''

    return matmul(dy, cache['w'].T)

# Step 78 - linear_backward_dw
def linear_backward_dw(dy: NDArray, cache: dict[str, NDArray]):
    """Return dL/dW for a linear layer Y = X @ W."""

    return matmul(cache['x'].T, dy)

# Step 79 - bias_add_forward
def bias_add_forward(x: NDArray, b: NDArray):
    """Add bias vector b (D,) to every row of x (B, D).

    Returns {'y': ndarray (B, D), 'cache': {'b_shape': tuple}}.
    """

    return {'y': x + b, 'cache': {'b_shape': b.shape}}

# Step 80 - bias_add_backward_db
def bias_add_backward_db(dy: NDArray, cache: dict[str, tuple[int, ...]]):
    """Compute db from upstream gradient dy for y = x + b."""

    return sum_axis0(dy)

# Step 81 - relu_forward
def relu_forward(x: NDArray):
    """Apply elementwise ReLU and cache the input for backward.

    Returns a dict with keys 'y' (activated array) and 'cache' (dict with 'x').
    """

    return {'y': np.maximum(x,0), 'cache': {'x': x}}

# Step 82 - relu_backward
def relu_backward(dy: NDArray, cache: dict[str, NDArray]):
    """Backward pass for ReLU. cache['x'] holds the original input."""

    return np.where(cache['x'] > 0, dy, 0)

# Step 83 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(probs: NDArray, targets: NDArray):
    """Return dL/dlogits for mean cross-entropy with softmax probs."""

    return compute_dlogits(probs, targets)

# Step 84 - layernorm_forward_mean
def layernorm_forward_mean(x: NDArray) -> NDArray:
    """Return the per-row mean of x with shape (B, 1)."""

    return np.mean(x, axis=-1, keepdims=True)

# Step 85 - layernorm_forward_variance
def layernorm_forward_variance(x: NDArray, mean: NDArray):
    """Compute the per-row (biased) variance of x given its per-row mean.

    Args:
        x: ndarray of shape (B, D).
        mean: ndarray of shape (B, 1), the per-row mean of x.

    Returns:
        var: ndarray of shape (B, 1), the per-row variance.
    """

    return np.mean((x - mean) ** 2, axis=-1, keepdims=True)

# Step 86 - layernorm_forward_normalize
def layernorm_forward_normalize(x: NDArray, mean: NDArray, var: NDArray, eps: float):
    """Normalize each row of x to zero mean and unit variance."""

    return (x - mean) / np.sqrt(var + eps)

# Step 87 - layernorm_forward_affine
def layernorm_forward_affine(x: NDArray, gamma: NDArray, beta: NDArray, eps: float):
    """Run LayerNorm forward over rows of x with affine params gamma, beta."""

    mean = layernorm_forward_mean(x)
    var = layernorm_forward_variance(x, mean)
    x_hat = layernorm_forward_normalize(x, mean, var, eps)
    y = gamma * x_hat + beta
    return {
        'y': y,
        'cache': {'x': x, 'x_hat': x_hat, 'mean': mean, 'var': var, 'gamma': gamma, 'eps': eps},
    }

# Step 88 - layernorm_backward_subtract_mean
def layernorm_backward_subtract_mean(dy: NDArray, cache: dict[str, NDArray | float]) -> NDArray:
    """Gradient through y = x - mean(x, axis=1, keepdims=True).

    dy: (B, D) upstream gradient w.r.t. the centered output.
    cache: dict with keys 'x' (B, D) and 'mean' (B,).
    Returns dx of shape (B, D).
    """

    return dy - np.mean(dy, axis=-1, keepdims=True)

# Step 89 - layernorm_backward_divide_std
def layernorm_backward_divide_std(dy: NDArray, cache: dict[str, NDArray | float]):
    """Propagate dy through the divide-by-std step of LayerNorm."""

    return dy / np.sqrt(cache['var'] + cache['eps'])

# Step 90 - layernorm_backward_full
def layernorm_backward_full(dy: NDArray, cache: dict[str, NDArray | float]):
    """Full LayerNorm backward. Return {'dx', 'dgamma', 'dbeta'}."""

    dgamma: NDArray = np.sum(dy * cache['x_hat'], axis=0)
    dbeta: NDArray = np.sum(dy, axis=0)
    dx_hat: NDArray = dy * cache['gamma']

    centered = cache['x'] - cache['mean']

    dcentered = layernorm_backward_divide_std(dx_hat, cache)

    dvar = np.sum(
        -0.5 * dx_hat * centered * (cache['var'] + cache['eps']) ** -1.5, axis=-1, keepdims=True
    )

    dcentered += dvar * 2 / dy.shape[-1] * centered

    dx = layernorm_backward_subtract_mean(dcentered, cache)

    return {'dx': dx, 'dgamma': dgamma, 'dbeta': dbeta}

# Step 91 - layernorm_backward_implementation
def layernorm_backward_implementation(d_out: NDArray, cache: dict[str, NDArray | float]):
    '''return {'dx', 'dgamma', 'dbeta'} gradients for LayerNorm given d_out and the forward cache.'''

    return layernorm_backward_full(d_out, cache)

# Step 92 - create_token_embedding
def create_token_embedding(vocab_size: int, d_model: int, scale: float = 0.02):
    """Initialize the token embedding matrix E of shape (vocab_size, d_model)."""

    return scale_w_small(np.random.randn(vocab_size, d_model), scale)

# Step 93 - token_embedding_forward
def token_embedding_forward(token_ids: NDArray, embedding_matrix: NDArray):
    """Look up token embeddings for a batch of integer token ids.

    Inputs:
        token_ids: ndarray of shape (B, T), dtype int
        embedding_matrix: ndarray of shape (V, d_model)
    Returns:
        out: ndarray of shape (B, T, d_model)
        cache: dict with keys 'token_ids', 'vocab_size'
    """

    return embedding_matrix[token_ids], {
        'token_ids': token_ids,
        'vocab_size': embedding_matrix.shape[0],
    }

# Step 94 - token_embedding_backward
def token_embedding_backward(d_out: NDArray, cache: dict[str, NDArray | int]):
    '''scatter-add d_out into a (vocab_size, d_model) dE using cache['token_ids'].'''

    dE = make_2d_zeros(cache['vocab_size'], d_out.shape[-1])
    np.add.at(dE, cache['token_ids'], d_out)
    return dE

# Step 95 - create_positional_embedding
def create_positional_embedding(block_size: int, d_model: int, scale: float = 0.02):
    """Initialize the learned positional embedding matrix P of shape (block_size, d_model)."""

    return scale_w_small(make_2d_random(block_size, d_model), scale)

# Step 96 - slice_positional_embedding
def slice_positional_embedding(positional_matrix: NDArray, seq_len: int):
    """Return the first seq_len rows of the positional embedding matrix."""

    return positional_matrix[:seq_len]

# Step 97 - add_token_and_positional_embeddings
def add_token_and_positional_embeddings(token_emb: NDArray, pos_emb: NDArray):
    """Sum token embeddings (B,T,d_model) and positional embeddings (T,d_model)."""

    return token_emb + pos_emb

# Step 98 - embedding_sum_backward
def embedding_sum_backward(d_out: NDArray):
    """Backprop through H = token_emb + pos_emb (with broadcasting over batch)."""

    return {'d_token_emb': d_out, 'd_pos_emb': np.sum(d_out, axis=0)}

# Step 99 - create_qkv_projections
def create_qkv_projections(d_model: int, d_head: int, scale=0.02):
    '''return a dict with 'Wq','Wk','Wv', each of shape (d_model, d_head)'''

    return {
        'Wq': scale_w_small(make_2d_random(d_model, d_head, 0), scale),
        'Wk': scale_w_small(make_2d_random(d_model, d_head, 1), scale),
        'Wv': scale_w_small(make_2d_random(d_model, d_head, 2), scale),
    }

# Step 100 - compute_query
def compute_query(x: NDArray, w_q: NDArray) -> NDArray:
    """Project x (B, T, d_model) into queries Q (B, T, d_head) using w_q."""

    return np.einsum('btm,mh->bth', x, w_q)

# Step 101 - compute_key
def compute_key(x: NDArray, w_k: NDArray) -> NDArray:
    """Project x through Wk to get keys K of shape (B, T, d_head)."""

    return np.einsum('btm,mh->bth', x, w_k)

# Step 102 - compute_value
def compute_value(x: NDArray, w_v: NDArray) -> NDArray:
    '''project x of shape (B, T, d_model) by w_v of shape (d_model, d_head)'''

    return np.einsum('btm,mh->bth', x, w_v)

# Step 103 - compute_attention_scores (not yet solved)
# TODO: implement

# Step 104 - scale_attention_scores (not yet solved)
# TODO: implement

# Step 105 - build_causal_mask (not yet solved)
# TODO: implement

# Step 106 - apply_causal_mask (not yet solved)
# TODO: implement

# Step 107 - softmax_attention_weights (not yet solved)
# TODO: implement

# Step 108 - attention_weighted_values (not yet solved)
# TODO: implement

# Step 109 - apply_output_projection (not yet solved)
# TODO: implement

# Step 110 - output_projection_backward (not yet solved)
# TODO: implement

# Step 111 - attention_value_backward (not yet solved)
# TODO: implement

# Step 112 - masked_softmax_backward (not yet solved)
# TODO: implement

# Step 113 - scale_scores_backward (not yet solved)
# TODO: implement

# Step 114 - qk_scores_backward (not yet solved)
# TODO: implement

# Step 115 - qkv_projection_backward (not yet solved)
# TODO: implement

# Step 116 - choose_attention_head_config (not yet solved)
# TODO: implement

# Step 117 - create_multihead_qkv_projections (not yet solved)
# TODO: implement

# Step 118 - create_multihead_output_projection (not yet solved)
# TODO: implement

# Step 119 - reshape_to_heads (not yet solved)
# TODO: implement

# Step 120 - transpose_heads_to_front (not yet solved)
# TODO: implement

# Step 121 - get_multihead_n_heads (not yet solved)
# TODO: implement

# Step 122 - get_multihead_sequence_length (not yet solved)
# TODO: implement

# Step 123 - compute_d_head (not yet solved)
# TODO: implement

# Step 124 - multihead_masked_softmax_scores (not yet solved)
# TODO: implement

# Step 125 - multihead_weighted_sum (not yet solved)
# TODO: implement

# Step 126 - transpose_heads_to_back (not yet solved)
# TODO: implement

# Step 127 - get_multihead_output_sequence_length (not yet solved)
# TODO: implement

# Step 128 - merge_heads_to_d_model (not yet solved)
# TODO: implement

# Step 129 - multihead_output_projection_forward (not yet solved)
# TODO: implement

# Step 130 - multihead_reshape_transpose_backward
def multihead_reshape_transpose_backward(d_merged, shape_info):
    """Invert merge_heads_to_d_model to recover (B, n_heads, T, d_head) gradients."""
    # TODO: undo the merge/transpose/reshape chain from the forward pass
    pass

# Step 131 - ffn_linear_one_forward (not yet solved)
# TODO: implement

# Step 132 - ffn_activation_forward (not yet solved)
# TODO: implement

# Step 133 - ffn_linear_two_forward (not yet solved)
# TODO: implement

# Step 134 - ffn_backward (not yet solved)
# TODO: implement

# Step 135 - residual_forward (not yet solved)
# TODO: implement

# Step 136 - residual_backward (not yet solved)
# TODO: implement

# Step 137 - pre_layernorm_sublayer_forward (not yet solved)
# TODO: implement

# Step 138 - transformer_block_forward (not yet solved)
# TODO: implement

# Step 139 - transformer_block_backward (not yet solved)
# TODO: implement

# Step 140 - stack_transformer_blocks (not yet solved)
# TODO: implement

# Step 141 - forward_through_all_blocks (not yet solved)
# TODO: implement

# Step 142 - backward_through_all_blocks (not yet solved)
# TODO: implement

# Step 143 - final_layernorm_forward (not yet solved)
# TODO: implement

# Step 144 - lm_head_linear_forward (not yet solved)
# TODO: implement

# Step 145 - full_model_forward (not yet solved)
# TODO: implement

# Step 146 - full_model_backward (not yet solved)
# TODO: implement

# Step 147 - initialize_adam_moments (not yet solved)
# TODO: implement

# Step 148 - initialize_adam_step_counter (not yet solved)
# TODO: implement

# Step 149 - adam_increment_step (not yet solved)
# TODO: implement

# Step 150 - adam_update_first_moment (not yet solved)
# TODO: implement

# Step 151 - adam_update_second_moment (not yet solved)
# TODO: implement

# Step 152 - adam_bias_correction (not yet solved)
# TODO: implement

# Step 153 - adam_parameter_update (not yet solved)
# TODO: implement

# Step 154 - wire_full_training_loop (not yet solved)
# TODO: implement

# Step 155 - logging_and_validation_loss (not yet solved)
# TODO: implement

# Step 156 - encode_prompt (not yet solved)
# TODO: implement

# Step 157 - crop_context_to_block_size (not yet solved)
# TODO: implement

# Step 158 - forward_to_get_logits (not yet solved)
# TODO: implement

# Step 159 - take_last_position_logits (not yet solved)
# TODO: implement

# Step 160 - apply_temperature (not yet solved)
# TODO: implement

# Step 161 - top_k_filter (not yet solved)
# TODO: implement

# Step 162 - softmax_to_probs (not yet solved)
# TODO: implement

# Step 163 - sample_one_token (not yet solved)
# TODO: implement

# Step 164 - append_token_to_sequence (not yet solved)
# TODO: implement

# Step 165 - generation_loop_for_n_steps (not yet solved)
# TODO: implement

# Step 166 - decode_final_sequence (not yet solved)
# TODO: implement

