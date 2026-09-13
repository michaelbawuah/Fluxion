from __future__ import annotations

import numpy as np

from fluxion.nn.losses import CrossEntropyLoss
from fluxion.optim.adam import Adam
from fluxion.transformer.gpt import GPT


# ---------------------------------------------------------
# 1. Training text
# ---------------------------------------------------------

text = """
cornell engineering builds great systems
cornell engineering builds great systems
cornell engineering builds great systems
cornell engineering builds great systems
"""


# ---------------------------------------------------------
# 2. Build a simple character-level vocabulary
# ---------------------------------------------------------

characters = sorted(set(text))

char_to_id = {
    character: index
    for index, character in enumerate(characters)
}

id_to_char = {
    index: character
    for character, index in char_to_id.items()
}

encoded = np.array(
    [
        char_to_id[character]
        for character in text
    ],
    dtype=np.int64,
)

vocab_size = len(characters)

print(
    f"Vocabulary size: {vocab_size}"
)


# ---------------------------------------------------------
# 3. Create training sequences
# ---------------------------------------------------------

sequence_length = 16

inputs = []
targets = []

for start in range(
    len(encoded) - sequence_length
):
    inputs.append(
        encoded[
            start : start + sequence_length
        ]
    )

    targets.append(
        encoded[
            start + 1 : start + sequence_length + 1
        ]
    )

inputs = np.array(
    inputs,
    dtype=np.int64,
)

targets = np.array(
    targets,
    dtype=np.int64,
)

print(
    f"Training examples: {len(inputs)}"
)


# ---------------------------------------------------------
# 4. Create Fluxion GPT
# ---------------------------------------------------------

np.random.seed(42)

model = GPT(
    vocab_size=vocab_size,
    max_sequence_length=sequence_length,
    embed_dim=16,
    num_heads=4,
    hidden_dim=32,
    num_layers=2,
)

criterion = CrossEntropyLoss()

optimizer = Adam(
    model.parameters(),
    lr=0.01,
)


# ---------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------

epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()

    logits = model(
        inputs
    )

    loss = criterion(
        logits,
        targets,
    )

    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:3d} | "
            f"Loss: {float(loss.data):.6f}"
        )


# ---------------------------------------------------------
# 6. Autoregressive text generation
# ---------------------------------------------------------

def generate(
    model: GPT,
    prompt: str,
    max_new_characters: int,
) -> str:
    generated = prompt

    for _ in range(max_new_characters):
        context = generated[
            -sequence_length:
        ]

        context_ids = np.array(
            [
                [
                    char_to_id[character]
                    for character in context
                ]
            ],
            dtype=np.int64,
        )

        logits = model(
            context_ids
        )

        next_token_logits = logits.data[
            0,
            -1,
        ]

        predicted_id = int(
            np.argmax(next_token_logits)
        )

        predicted_character = id_to_char[
            predicted_id
        ]

        generated += predicted_character

    return generated


prompt = "cornell engineer"

generated_text = generate(
    model=model,
    prompt=prompt,
    max_new_characters=40,
)

print()
print(
    f"Prompt: {prompt!r}"
)

print()
print("Generated text:")
print(generated_text)