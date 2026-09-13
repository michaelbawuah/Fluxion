import numpy as np

from fluxion.nn.layers import Linear, ReLU
from fluxion.nn.losses import MSELoss
from fluxion.nn.module import Module
from fluxion.optim.sgd import SGD
from fluxion.tensor import Tensor


class TinyNetwork(Module):
    def __init__(self) -> None:
        self.layer1 = Linear(1, 8)
        self.relu = ReLU()
        self.layer2 = Linear(8, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


np.random.seed(0)

x = Tensor(
    [
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ]
)

target = Tensor(
    [
        [2.0],
        [4.0],
        [6.0],
        [8.0],
        [10.0],
    ]
)

model = TinyNetwork()
loss_fn = MSELoss()

optimizer = SGD(
    model.parameters(),
    lr=0.01,
)

for epoch in range(1000):
    prediction = model(x)

    loss = loss_fn(prediction, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Loss: {loss.data.item():.6f}"
        )


print("\nFinal predictions:")

final_prediction = model(x)

for input_value, predicted_value, target_value in zip(
    x.data,
    final_prediction.data,
    target.data,
):
    print(
        f"x={input_value.item():.1f} | "
        f"prediction={predicted_value.item():.3f} | "
        f"target={target_value.item():.1f}"
    )

print("\nPredictions on unseen data:")

unseen_x = Tensor(
    [
        [6.0],
        [7.0],
        [8.0],
        [10.0],
    ]
)

unseen_prediction = model(unseen_x)

for input_value, predicted_value in zip(
    unseen_x.data,
    unseen_prediction.data,
):
    print(
        f"x={input_value.item():.1f} | "
        f"prediction={predicted_value.item():.3f}"
    )    