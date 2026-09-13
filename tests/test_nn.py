import numpy as np
import torch


from fluxion.transformer.gpt import GPT
from fluxion.transformer.layers import TransformerBlock, PositionalEmbedding
from fluxion.transformer.attention import (
    CausalSelfAttention,
    MultiHeadAttention,
    ScaledDotProductAttention,
    SelfAttention,
)
from fluxion.nn.layers import Linear, NativeLinear, ReLU, Sigmoid, Softmax, LayerNorm, Embedding
from fluxion.nn.losses import MSELoss, CrossEntropyLoss
from fluxion.nn.module import Sequential
from fluxion.tensor import Tensor


def test_linear_shapes_and_parameters():
    layer = Linear(3, 2)

    assert layer.weight.shape == (3, 2)
    assert layer.bias is not None
    assert layer.bias.shape == (2,)

    params = layer.parameters()

    assert layer.weight in params
    assert layer.bias in params
    assert len(params) == 2


def test_linear_forward():
    layer = Linear(3, 2)

    layer.weight.data[:] = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    layer.bias.data[:] = np.array([1.0, -1.0])

    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    y = layer(x)

    expected = np.array(
        [
            [23.0, 27.0],
            [50.0, 63.0],
        ]
    )

    np.testing.assert_allclose(
        y.data,
        expected,
    )


def test_linear_backward():
    layer = Linear(3, 2)

    layer.weight.data[:] = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    layer.bias.data[:] = np.array([1.0, -1.0])

    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None

    assert x.grad.shape == x.shape
    assert layer.weight.grad.shape == layer.weight.shape
    assert layer.bias.grad.shape == layer.bias.shape


def test_zero_grad():
    layer = Linear(3, 2)

    x = Tensor([[1.0, 2.0, 3.0]])

    loss = layer(x).sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert layer.bias.grad is not None

    layer.zero_grad()

    assert layer.weight.grad is None
    assert layer.bias.grad is None


def test_relu_matches_pytorch():
    x = Tensor(
        [[-2.0, -1.0, 0.0, 1.0, 3.0]],
        requires_grad=True,
    )

    relu = ReLU()

    y = relu(x)
    loss = y.sum()
    loss.backward()

    torch_x = torch.tensor(
        [[-2.0, -1.0, 0.0, 1.0, 3.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_y = torch.relu(torch_x)
    torch_loss = torch_y.sum()
    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )


def test_mse_loss_matches_pytorch():
    prediction = Tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        requires_grad=True,
    )

    target = Tensor(
        [[1.5, 1.0], [2.5, 5.0]],
    )

    loss_fn = MSELoss()

    loss = loss_fn(prediction, target)
    loss.backward()

    torch_prediction = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_target = torch.tensor(
        [[1.5, 1.0], [2.5, 5.0]],
        dtype=torch.float64,
    )

    torch_loss = torch.nn.functional.mse_loss(
        torch_prediction,
        torch_target,
    )

    torch_loss.backward()

    np.testing.assert_allclose(
        loss.data,
        torch_loss.detach().numpy(),
    )

    np.testing.assert_allclose(
        prediction.grad,
        torch_prediction.grad.numpy(),
    )


def test_sigmoid_matches_pytorch():
    x = Tensor(
        [[-2.0, -1.0, 0.0, 1.0, 2.0]],
        requires_grad=True,
    )

    sigmoid = Sigmoid()

    y = sigmoid(x)
    loss = y.sum()
    loss.backward()

    torch_x = torch.tensor(
        [[-2.0, -1.0, 0.0, 1.0, 2.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_y = torch.sigmoid(torch_x)
    torch_loss = torch_y.sum()
    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )


def test_sequential_forward():
    model = Sequential(
        Linear(2, 3),
        ReLU(),
        Linear(3, 1),
    )

    first = model.modules[0]
    second = model.modules[2]

    first.weight.data[:] = np.array(
        [
            [1.0, -1.0, 2.0],
            [0.5, 2.0, -1.0],
        ]
    )
    first.bias.data[:] = np.array([0.0, 1.0, 0.5])

    second.weight.data[:] = np.array(
        [
            [1.0],
            [2.0],
            [-1.0],
        ]
    )
    second.bias.data[:] = np.array([0.25])

    x = Tensor([[2.0, 1.0]])

    y = model(x)

    expected = np.array([[1.25]])

    np.testing.assert_allclose(
        y.data,
        expected,
    )


def test_sequential_parameters():
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1),
    )

    params = model.parameters()

    assert len(params) == 4

    assert model.modules[0].weight in params
    assert model.modules[0].bias in params
    assert model.modules[2].weight in params
    assert model.modules[2].bias in params

def test_softmax_matches_pytorch():
    x = Tensor(
        [
            [2.0, 1.0, 0.1],
            [1.0, 3.0, 2.0],
        ],
        requires_grad=True,
    )

    softmax = Softmax(axis=-1)

    y = softmax(x)

    weights = Tensor(
        [
            [1.0, 2.0, 3.0],
            [0.5, -1.0, 2.0],
        ]
    )

    loss = (y * weights).sum()
    loss.backward()

    torch_x = torch.tensor(
        [
            [2.0, 1.0, 0.1],
            [1.0, 3.0, 2.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_y = torch.softmax(
        torch_x,
        dim=-1,
    )

    torch_weights = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [0.5, -1.0, 2.0],
        ],
        dtype=torch.float64,
    )

    torch_loss = (
        torch_y
        * torch_weights
    ).sum()

    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )

    np.testing.assert_allclose(
        y.data.sum(axis=-1),
        np.ones(2),
    )   

def test_cross_entropy_matches_pytorch():
    logits = Tensor(
        [
            [2.0, 1.0, 0.1],
            [0.5, 3.0, 1.2],
            [1.5, 0.2, 2.8],
        ],
        requires_grad=True,
    )

    targets = np.array(
        [0, 1, 2],
        dtype=np.int64,
    )

    loss_fn = CrossEntropyLoss()

    loss = loss_fn(
        logits,
        targets,
    )

    loss.backward()

    torch_logits = torch.tensor(
        [
            [2.0, 1.0, 0.1],
            [0.5, 3.0, 1.2],
            [1.5, 0.2, 2.8],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_targets = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    torch_loss = torch.nn.functional.cross_entropy(
        torch_logits,
        torch_targets,
    )

    torch_loss.backward()

    np.testing.assert_allclose(
        loss.data,
        torch_loss.detach().numpy(),
    )

    np.testing.assert_allclose(
        logits.grad,
        torch_logits.grad.numpy(),
    )

def test_layernorm_matches_pytorch():
    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 7.0],
        ],
        requires_grad=True,
    )

    layer_norm = LayerNorm(
        normalized_shape=3,
        eps=1e-5,
    )

    layer_norm.weight.data[:] = np.array(
        [1.2, 0.8, 1.5]
    )

    layer_norm.bias.data[:] = np.array(
        [0.1, -0.2, 0.3]
    )

    y = layer_norm(x)

    weights = Tensor(
        [
            [1.0, 2.0, -1.0],
            [0.5, -2.0, 3.0],
        ]
    )

    loss = (y * weights).sum()
    loss.backward()

    torch_x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 7.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_layer_norm = torch.nn.LayerNorm(
        normalized_shape=3,
        eps=1e-5,
        dtype=torch.float64,
    )

    with torch.no_grad():
        torch_layer_norm.weight.copy_(
            torch.tensor(
                [1.2, 0.8, 1.5],
                dtype=torch.float64,
            )
        )

        torch_layer_norm.bias.copy_(
            torch.tensor(
                [0.1, -0.2, 0.3],
                dtype=torch.float64,
            )
        )

    torch_y = torch_layer_norm(torch_x)

    torch_weights = torch.tensor(
        [
            [1.0, 2.0, -1.0],
            [0.5, -2.0, 3.0],
        ],
        dtype=torch.float64,
    )

    torch_loss = (
        torch_y * torch_weights
    ).sum()

    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        layer_norm.weight.grad,
        torch_layer_norm.weight.grad.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        layer_norm.bias.grad,
        torch_layer_norm.bias.grad.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

def test_scaled_dot_product_attention_matches_pytorch():
    query = Tensor(
        np.array(
            [
                [
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                ]
            ]
        ),
        requires_grad=True,
    )

    key = Tensor(
        np.array(
            [
                [
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0],
                ]
            ]
        ),
        requires_grad=True,
    )

    value = Tensor(
        np.array(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ]
            ]
        ),
        requires_grad=True,
    )

    attention = ScaledDotProductAttention()

    output = attention(
        query,
        key,
        value,
    )

    weights = Tensor(
        np.array(
            [
                [
                    [1.0, -1.0],
                    [0.5, 2.0],
                ]
            ]
        )
    )

    loss = (
        output * weights
    ).sum()

    loss.backward()

    torch_query = torch.tensor(
        query.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_key = torch.tensor(
        key.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_value = torch.tensor(
        value.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_output = torch.nn.functional.scaled_dot_product_attention(
        torch_query,
        torch_key,
        torch_value,
        dropout_p=0.0,
    )

    torch_weights = torch.tensor(
        weights.data,
        dtype=torch.float64,
    )

    torch_loss = (
        torch_output * torch_weights
    ).sum()

    torch_loss.backward()

    np.testing.assert_allclose(
        output.data,
        torch_output.detach().numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        query.grad,
        torch_query.grad.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        key.grad,
        torch_key.grad.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        value.grad,
        torch_value.grad.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

def test_permute_matches_pytorch():
    x = Tensor(
        np.arange(24.0).reshape(2, 3, 4),
        requires_grad=True,
    )

    y = x.permute(0, 2, 1)

    weights = Tensor(
        np.arange(24.0).reshape(2, 4, 3) + 1.0,
    )

    loss = (y * weights).sum()
    loss.backward()

    torch_x = torch.tensor(
        np.arange(24.0).reshape(2, 3, 4),
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_y = torch_x.permute(0, 2, 1)

    torch_weights = torch.tensor(
        np.arange(24.0).reshape(2, 4, 3) + 1.0,
        dtype=torch.float64,
    )

    torch_loss = (
        torch_y * torch_weights
    ).sum()

    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )


def test_multi_head_attention_matches_pytorch():
    np.random.seed(0)

    attention = MultiHeadAttention(
        embed_dim=4,
        num_heads=2,
    )

    x = Tensor(
        np.array(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [2.0, 1.0, 0.0, 1.0],
                    [0.5, 1.5, 2.5, 3.5],
                ]
            ]
        ),
        requires_grad=True,
    )

    output = attention(
        x,
        x,
        x,
    )

    output_weights = Tensor(
        np.array(
            [
                [
                    [1.0, -1.0, 0.5, 2.0],
                    [0.5, 1.0, -0.5, 1.5],
                    [2.0, 0.5, 1.0, -1.0],
                ]
            ]
        )
    )

    loss = (
        output * output_weights
    ).sum()

    loss.backward()

    torch_x = torch.tensor(
        x.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_q_weight = torch.tensor(
        attention.query_projection.weight.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_q_bias = torch.tensor(
        attention.query_projection.bias.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_k_weight = torch.tensor(
        attention.key_projection.weight.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_k_bias = torch.tensor(
        attention.key_projection.bias.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_v_weight = torch.tensor(
        attention.value_projection.weight.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_v_bias = torch.tensor(
        attention.value_projection.bias.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_o_weight = torch.tensor(
        attention.output_projection.weight.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_o_bias = torch.tensor(
        attention.output_projection.bias.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_q = (
        torch_x @ torch_q_weight
        + torch_q_bias
    )

    torch_k = (
        torch_x @ torch_k_weight
        + torch_k_bias
    )

    torch_v = (
        torch_x @ torch_v_weight
        + torch_v_bias
    )

    batch_size = torch_x.shape[0]
    sequence_length = torch_x.shape[1]
    num_heads = 2
    head_dim = 2

    torch_q = (
        torch_q
        .reshape(
            batch_size,
            sequence_length,
            num_heads,
            head_dim,
        )
        .permute(0, 2, 1, 3)
    )

    torch_k = (
        torch_k
        .reshape(
            batch_size,
            sequence_length,
            num_heads,
            head_dim,
        )
        .permute(0, 2, 1, 3)
    )

    torch_v = (
        torch_v
        .reshape(
            batch_size,
            sequence_length,
            num_heads,
            head_dim,
        )
        .permute(0, 2, 1, 3)
    )

    torch_scores = (
        torch_q
        @ torch_k.transpose(-1, -2)
    ) / np.sqrt(head_dim)

    torch_attention_weights = torch.softmax(
        torch_scores,
        dim=-1,
    )

    torch_attended = (
        torch_attention_weights
        @ torch_v
    )

    torch_attended = (
        torch_attended
        .permute(0, 2, 1, 3)
        .reshape(
            batch_size,
            sequence_length,
            4,
        )
    )

    torch_output = (
        torch_attended @ torch_o_weight
        + torch_o_bias
    )

    torch_output_weights = torch.tensor(
        output_weights.data,
        dtype=torch.float64,
    )

    torch_loss = (
        torch_output
        * torch_output_weights
    ).sum()

    torch_loss.backward()

    np.testing.assert_allclose(
        output.data,
        torch_output.detach().numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        attention.query_projection.weight.grad,
        torch_q_weight.grad.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        attention.key_projection.weight.grad,
        torch_k_weight.grad.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        attention.value_projection.weight.grad,
        torch_v_weight.grad.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        attention.output_projection.weight.grad,
        torch_o_weight.grad.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )


def test_self_attention_shape_and_parameters():
    attention = SelfAttention(
        embed_dim=8,
        num_heads=2,
    )

    x = Tensor(
        np.random.randn(2, 5, 8),
        requires_grad=True,
    )

    output = attention(x)

    assert output.shape == (2, 5, 8)

    parameters = attention.parameters()

    assert len(parameters) == 8


def test_causal_self_attention_blocks_future_tokens():
    np.random.seed(1)

    attention = CausalSelfAttention(
        embed_dim=4,
        num_heads=2,
    )

    original = np.array(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
            ]
        ]
    )

    modified = original.copy()

    modified[0, 1:] = np.array(
        [
            [100.0, 200.0, 300.0, 400.0],
            [-100.0, -200.0, -300.0, -400.0],
        ]
    )

    output_original = attention(
        Tensor(original)
    )

    output_modified = attention(
        Tensor(modified)
    )

    np.testing.assert_allclose(
        output_original.data[:, 0, :],
        output_modified.data[:, 0, :],
        rtol=1e-6,
        atol=1e-7,
    )

def test_transformer_block_matches_pytorch():
    np.random.seed(7)

    block = TransformerBlock(
        embed_dim=4,
        num_heads=2,
        hidden_dim=8,
    )

    x = Tensor(
        np.array(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [2.0, 0.5, 1.5, 3.0],
                    [0.0, 1.0, 2.0, 1.0],
                ]
            ]
        ),
        requires_grad=True,
    )

    output = block(x)

    output_weights = Tensor(
        np.array(
            [
                [
                    [1.0, -1.0, 0.5, 2.0],
                    [0.5, 1.0, -0.5, 1.5],
                    [2.0, 0.5, 1.0, -1.0],
                ]
            ]
        )
    )

    loss = (
        output * output_weights
    ).sum()

    loss.backward()

    torch_x = torch.tensor(
        x.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    def torch_linear(
        tensor,
        fluxion_linear,
    ):
        weight = torch.tensor(
            fluxion_linear.weight.data,
            dtype=torch.float64,
            requires_grad=True,
        )

        bias = torch.tensor(
            fluxion_linear.bias.data,
            dtype=torch.float64,
            requires_grad=True,
        )

        return (
            tensor @ weight + bias
        ), weight, bias

    norm1_weight = torch.tensor(
        block.norm1.weight.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    norm1_bias = torch.tensor(
        block.norm1.bias.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_norm1 = torch.nn.functional.layer_norm(
        torch_x,
        (4,),
        norm1_weight,
        norm1_bias,
        block.norm1.eps,
    )

    mha = block.attention.attention

    torch_q, _, _ = torch_linear(
        torch_norm1,
        mha.query_projection,
    )

    torch_k, _, _ = torch_linear(
        torch_norm1,
        mha.key_projection,
    )

    torch_v, _, _ = torch_linear(
        torch_norm1,
        mha.value_projection,
    )

    batch_size = torch_x.shape[0]
    sequence_length = torch_x.shape[1]
    num_heads = 2
    head_dim = 2

    torch_q = (
        torch_q
        .reshape(
            batch_size,
            sequence_length,
            num_heads,
            head_dim,
        )
        .permute(0, 2, 1, 3)
    )

    torch_k = (
        torch_k
        .reshape(
            batch_size,
            sequence_length,
            num_heads,
            head_dim,
        )
        .permute(0, 2, 1, 3)
    )

    torch_v = (
        torch_v
        .reshape(
            batch_size,
            sequence_length,
            num_heads,
            head_dim,
        )
        .permute(0, 2, 1, 3)
    )

    torch_scores = (
        torch_q
        @ torch_k.transpose(-1, -2)
    ) / np.sqrt(head_dim)

    causal_mask = torch.triu(
        torch.full(
            (
                sequence_length,
                sequence_length,
            ),
            float("-inf"),
            dtype=torch.float64,
        ),
        diagonal=1,
    )

    torch_scores = (
        torch_scores
        + causal_mask
    )

    torch_attention_weights = torch.softmax(
        torch_scores,
        dim=-1,
    )

    torch_attended = (
        torch_attention_weights
        @ torch_v
    )

    torch_attended = (
        torch_attended
        .permute(0, 2, 1, 3)
        .reshape(
            batch_size,
            sequence_length,
            4,
        )
    )

    torch_attended, _, _ = torch_linear(
        torch_attended,
        mha.output_projection,
    )

    torch_x1 = (
        torch_x + torch_attended
    )

    norm2_weight = torch.tensor(
        block.norm2.weight.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    norm2_bias = torch.tensor(
        block.norm2.bias.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_norm2 = torch.nn.functional.layer_norm(
        torch_x1,
        (4,),
        norm2_weight,
        norm2_bias,
        block.norm2.eps,
    )

    linear1 = block.feed_forward.modules[0]
    linear2 = block.feed_forward.modules[2]

    torch_ff1, _, _ = torch_linear(
        torch_norm2,
        linear1,
    )

    torch_relu = torch.relu(
        torch_ff1
    )

    torch_ff2, _, _ = torch_linear(
        torch_relu,
        linear2,
    )

    torch_output = (
        torch_x1 + torch_ff2
    )

    torch_output_weights = torch.tensor(
        output_weights.data,
        dtype=torch.float64,
    )

    torch_loss = (
        torch_output
        * torch_output_weights
    ).sum()

    torch_loss.backward()

    np.testing.assert_allclose(
        output.data,
        torch_output.detach().numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

def test_embedding_matches_pytorch_with_repeated_indices():
    np.random.seed(11)

    embedding = Embedding(
        num_embeddings=6,
        embedding_dim=4,
    )

    indices = np.array(
        [
            [1, 3, 1],
            [2, 3, 4],
        ]
    )

    output = embedding(indices)

    weights = Tensor(
        np.arange(
            output.size,
            dtype=float,
        ).reshape(output.shape)
        + 1.0
    )

    loss = (
        output * weights
    ).sum()

    loss.backward()

    torch_weight = torch.tensor(
        embedding.weight.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_indices = torch.tensor(
        indices,
        dtype=torch.long,
    )

    torch_output = torch.nn.functional.embedding(
        torch_indices,
        torch_weight,
    )

    torch_weights = torch.tensor(
        weights.data,
        dtype=torch.float64,
    )

    torch_loss = (
        torch_output * torch_weights
    ).sum()

    torch_loss.backward()

    np.testing.assert_allclose(
        output.data,
        torch_output.detach().numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        embedding.weight.grad,
        torch_weight.grad.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

def test_positional_embedding_matches_pytorch():
    np.random.seed(13)

    positional_embedding = PositionalEmbedding(
        max_sequence_length=8,
        embed_dim=4,
    )

    output = positional_embedding(
        sequence_length=5,
    )

    weights = Tensor(
        np.arange(
            output.size,
            dtype=float,
        ).reshape(output.shape)
        + 1.0
    )

    loss = (
        output * weights
    ).sum()

    loss.backward()

    torch_weight = torch.tensor(
        positional_embedding.embedding.weight.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_positions = torch.arange(
        5,
        dtype=torch.long,
    )

    torch_output = torch.nn.functional.embedding(
        torch_positions,
        torch_weight,
    )

    torch_weights = torch.tensor(
        weights.data,
        dtype=torch.float64,
    )

    torch_loss = (
        torch_output * torch_weights
    ).sum()

    torch_loss.backward()

    np.testing.assert_allclose(
        output.data,
        torch_output.detach().numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        positional_embedding.embedding.weight.grad,
        torch_weight.grad.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

def test_gpt_forward_parameters_and_backward():
    np.random.seed(21)

    model = GPT(
        vocab_size=10,
        max_sequence_length=6,
        embed_dim=4,
        num_heads=2,
        hidden_dim=8,
        num_layers=2,
    )

    token_ids = np.array(
        [
            [1, 2, 3, 4],
            [4, 3, 2, 1],
        ],
        dtype=np.int64,
    )

    logits = model(token_ids)

    assert logits.shape == (
        2,
        4,
        10,
    )

    parameters = model.parameters()

    assert len(parameters) > 0

    weights = Tensor(
        np.arange(
            logits.size,
            dtype=float,
        ).reshape(logits.shape)
        + 1.0
    )

    loss = (
        logits * weights
    ).sum()

    loss.backward()

    assert model.token_embedding.weight.grad is not None

    assert (
        model.position_embedding
        .embedding
        .weight
        .grad
        is not None
    )

    assert model.output_projection.weight.grad is not None

    for block in model.blocks:
        for parameter in block.parameters():
            assert parameter.grad is not None

def test_language_model_cross_entropy_matches_pytorch():
    logits = Tensor(
        np.array(
            [
                [
                    [2.0, 1.0, 0.0, -1.0],
                    [0.5, 1.5, -0.5, 2.0],
                    [1.0, 0.0, 2.0, 0.5],
                ],
                [
                    [1.2, -0.2, 0.7, 2.1],
                    [0.0, 1.0, 2.0, 3.0],
                    [2.5, 1.5, 0.5, -0.5],
                ],
            ]
        ),
        requires_grad=True,
    )

    targets = np.array(
        [
            [0, 3, 2],
            [3, 1, 0],
        ],
        dtype=np.int64,
    )

    criterion = CrossEntropyLoss()

    loss = criterion(
        logits,
        targets,
    )

    loss.backward()

    torch_logits = torch.tensor(
        logits.data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_targets = torch.tensor(
        targets,
        dtype=torch.long,
    )

    torch_loss = torch.nn.functional.cross_entropy(
        torch_logits.reshape(-1, 4),
        torch_targets.reshape(-1),
    )

    torch_loss.backward()

    np.testing.assert_allclose(
        loss.data,
        torch_loss.detach().numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        logits.grad,
        torch_logits.grad.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

def test_native_linear_matches_linear_forward_and_backward():
    np.random.seed(0)

    batch_size = 4
    input_dim = 3
    output_dim = 5

    x_data = np.random.randn(
        batch_size,
        input_dim,
    )

    weight_data = np.random.randn(
        input_dim,
        output_dim,
    )

    bias_data = np.random.randn(
        output_dim,
    )

    x_regular = Tensor(
        x_data.copy(),
        requires_grad=True,
    )

    x_native = Tensor(
        x_data.copy(),
        requires_grad=True,
    )

    regular = Linear(
        input_dim,
        output_dim,
    )

    native = NativeLinear(
        input_dim,
        output_dim,
    )

    regular.weight.data = weight_data.copy()
    regular.bias.data = bias_data.copy()

    native.weight.data = weight_data.copy()
    native.bias.data = bias_data.copy()

    regular_output = regular(
        x_regular
    )

    native_output = native(
        x_native
    )

    assert np.allclose(
        regular_output.data,
        native_output.data,
    )

    regular_loss = regular_output.sum()
    native_loss = native_output.sum()

    regular_loss.backward()
    native_loss.backward()

    assert np.allclose(
        x_regular.grad,
        x_native.grad,
    )

    assert np.allclose(
        regular.weight.grad,
        native.weight.grad,
    )

    assert np.allclose(
        regular.bias.grad,
        native.bias.grad,
    )