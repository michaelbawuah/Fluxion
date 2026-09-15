from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from fluxion.nn.layers import LayerNorm, Linear
from fluxion.tensor import Tensor
from fluxion.transformer.attention import ScaledDotProductAttention
from fluxion.transformer.gpt import GPT
from fluxion.transformer.layers import TransformerBlock

DTYPE = torch.float64
ATOL = 1e-7
RTOL = 1e-6


@dataclass
class ErrorStats:
    max_abs: float
    max_rel: float


def error_stats(actual: np.ndarray, reference: np.ndarray) -> ErrorStats:
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    abs_error = np.abs(actual - reference)
    denom = np.maximum(np.abs(reference), 1e-12)
    return ErrorStats(float(abs_error.max(initial=0.0)), float((abs_error / denom).max(initial=0.0)))


def leaf(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype=DTYPE, requires_grad=True)


def register(mapping: dict[str, tuple[Tensor, torch.Tensor]], name: str, tensor: Tensor) -> torch.Tensor:
    result = leaf(tensor.data)
    mapping[name] = (tensor, result)
    return result


def torch_linear(x: torch.Tensor, layer, mapping, prefix: str) -> torch.Tensor:
    weight = register(mapping, f"{prefix}.weight", layer.weight)
    y = x @ weight
    if layer.bias is not None:
        bias = register(mapping, f"{prefix}.bias", layer.bias)
        y = y + bias
    return y


def torch_layer_norm(x: torch.Tensor, layer, mapping, prefix: str) -> torch.Tensor:
    weight = register(mapping, f"{prefix}.weight", layer.weight)
    bias = register(mapping, f"{prefix}.bias", layer.bias)
    return F.layer_norm(x, (layer.normalized_shape,), weight, bias, layer.eps)


def torch_mha(x: torch.Tensor, wrapper, mapping, prefix: str) -> torch.Tensor:
    mha = wrapper.attention
    q = torch_linear(x, mha.query_projection, mapping, f"{prefix}.q")
    k = torch_linear(x, mha.key_projection, mapping, f"{prefix}.k")
    v = torch_linear(x, mha.value_projection, mapping, f"{prefix}.v")
    batch, seq, _ = x.shape
    q = q.reshape(batch, seq, mha.num_heads, mha.head_dim).permute(0, 2, 1, 3)
    k = k.reshape(batch, seq, mha.num_heads, mha.head_dim).permute(0, 2, 1, 3)
    v = v.reshape(batch, seq, mha.num_heads, mha.head_dim).permute(0, 2, 1, 3)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(mha.head_dim)
    if mha.attention.causal:
        mask = torch.triu(torch.full((seq, seq), float("-inf"), dtype=DTYPE), diagonal=1)
        scores = scores + mask
    attended = torch.softmax(scores, dim=-1) @ v
    attended = attended.permute(0, 2, 1, 3).reshape(batch, seq, mha.embed_dim)
    return torch_linear(attended, mha.output_projection, mapping, f"{prefix}.out")


def torch_block(x: torch.Tensor, block: TransformerBlock, mapping, prefix: str) -> torch.Tensor:
    normalized = torch_layer_norm(x, block.norm1, mapping, f"{prefix}.norm1")
    x = x + torch_mha(normalized, block.attention, mapping, f"{prefix}.attn")
    normalized = torch_layer_norm(x, block.norm2, mapping, f"{prefix}.norm2")
    ff0, _, ff2 = block.feed_forward.modules
    ff = torch.relu(torch_linear(normalized, ff0, mapping, f"{prefix}.ff0"))
    ff = torch_linear(ff, ff2, mapping, f"{prefix}.ff2")
    return x + ff


def torch_gpt(token_ids: np.ndarray, model: GPT, mapping) -> torch.Tensor:
    token_weight = register(mapping, "token_embedding.weight", model.token_embedding.weight)
    pos_weight = register(mapping, "position_embedding.weight", model.position_embedding.embedding.weight)
    ids = torch.tensor(token_ids, dtype=torch.long)
    seq = token_ids.shape[1]
    x = F.embedding(ids, token_weight)
    positions = torch.arange(seq, dtype=torch.long)
    x = x + F.embedding(positions, pos_weight)
    for index, block in enumerate(model.blocks):
        x = torch_block(x, block, mapping, f"blocks.{index}")
    x = torch_layer_norm(x, model.final_norm, mapping, "final_norm")
    return torch_linear(x, model.output_projection, mapping, "output_projection")


def summarize(name: str, output: ErrorStats, grad_errors: list[ErrorStats]) -> None:
    grad_abs = max((e.max_abs for e in grad_errors), default=0.0)
    grad_rel = max((e.max_rel for e in grad_errors), default=0.0)
    passed = output.max_abs <= ATOL + RTOL and grad_abs <= ATOL + RTOL
    print(f"{name:<20} output abs={output.max_abs:.3e} rel={output.max_rel:.3e} | grad abs={grad_abs:.3e} rel={grad_rel:.3e} | {'PASS' if passed else 'CHECK'}")


def compare_mapping(mapping) -> list[ErrorStats]:
    errors = []
    for name, (fluxion_tensor, torch_tensor) in mapping.items():
        if fluxion_tensor.grad is None or torch_tensor.grad is None:
            raise RuntimeError(f"Missing gradient for {name}")
        errors.append(error_stats(fluxion_tensor.grad, torch_tensor.grad.detach().numpy()))
    return errors


def validate_linear() -> None:
    np.random.seed(1)
    layer = Linear(5, 4)
    x_data = np.random.randn(3, 5)
    x = Tensor(x_data, requires_grad=True)
    out = layer(x)
    weights = np.random.randn(*out.shape)
    (out * Tensor(weights)).sum().backward()
    mapping = {}
    tx = leaf(x_data)
    tout = torch_linear(tx, layer, mapping, "linear")
    (tout * torch.tensor(weights, dtype=DTYPE)).sum().backward()
    grads = compare_mapping(mapping) + [error_stats(x.grad, tx.grad.numpy())]
    summarize("Linear", error_stats(out.data, tout.detach().numpy()), grads)


def validate_layernorm() -> None:
    np.random.seed(2)
    layer = LayerNorm(6)
    layer.weight.data[:] = np.random.randn(6)
    layer.bias.data[:] = np.random.randn(6)
    x_data = np.random.randn(2, 3, 6)
    x = Tensor(x_data, requires_grad=True)
    out = layer(x)
    weights = np.random.randn(*out.shape)
    (out * Tensor(weights)).sum().backward()
    mapping = {}
    tx = leaf(x_data)
    tout = torch_layer_norm(tx, layer, mapping, "layernorm")
    (tout * torch.tensor(weights, dtype=DTYPE)).sum().backward()
    grads = compare_mapping(mapping) + [error_stats(x.grad, tx.grad.numpy())]
    summarize("LayerNorm", error_stats(out.data, tout.detach().numpy()), grads)


def validate_attention() -> None:
    np.random.seed(3)
    qd = np.random.randn(2, 4, 6)
    kd = np.random.randn(2, 4, 6)
    vd = np.random.randn(2, 4, 5)
    q, k, v = (Tensor(qd, requires_grad=True), Tensor(kd, requires_grad=True), Tensor(vd, requires_grad=True))
    layer = ScaledDotProductAttention(causal=True)
    out = layer(q, k, v)
    weights = np.random.randn(*out.shape)
    (out * Tensor(weights)).sum().backward()
    tq, tk, tv = leaf(qd), leaf(kd), leaf(vd)
    tout = F.scaled_dot_product_attention(tq, tk, tv, is_causal=True, dropout_p=0.0)
    (tout * torch.tensor(weights, dtype=DTYPE)).sum().backward()
    grads = [error_stats(q.grad, tq.grad.numpy()), error_stats(k.grad, tk.grad.numpy()), error_stats(v.grad, tv.grad.numpy())]
    summarize("Causal attention", error_stats(out.data, tout.detach().numpy()), grads)


def validate_transformer() -> None:
    np.random.seed(4)
    block = TransformerBlock(8, 2, 16)
    x_data = np.random.randn(2, 5, 8)
    x = Tensor(x_data, requires_grad=True)
    out = block(x)
    weights = np.random.randn(*out.shape)
    (out * Tensor(weights)).sum().backward()
    mapping = {}
    tx = leaf(x_data)
    tout = torch_block(tx, block, mapping, "block")
    (tout * torch.tensor(weights, dtype=DTYPE)).sum().backward()
    grads = compare_mapping(mapping) + [error_stats(x.grad, tx.grad.numpy())]
    summarize("TransformerBlock", error_stats(out.data, tout.detach().numpy()), grads)


def validate_gpt() -> None:
    np.random.seed(5)
    model = GPT(vocab_size=17, max_sequence_length=8, embed_dim=8, num_heads=2, hidden_dim=16, num_layers=2)
    token_ids = np.random.randint(0, 17, size=(2, 6))
    out = model(token_ids)
    weights = np.random.randn(*out.shape)
    (out * Tensor(weights)).sum().backward()
    mapping = {}
    tout = torch_gpt(token_ids, model, mapping)
    (tout * torch.tensor(weights, dtype=DTYPE)).sum().backward()
    grads = compare_mapping(mapping)
    summarize("GPT", error_stats(out.data, tout.detach().numpy()), grads)


def main() -> None:
    torch.set_default_dtype(DTYPE)
    torch.set_num_threads(1)
    print("Fluxion ↔ PyTorch numerical validation")
    print("=" * 92)
    validate_linear()
    validate_layernorm()
    validate_attention()
    validate_transformer()
    validate_gpt()


if __name__ == "__main__":
    main()
