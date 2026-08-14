# nn-from-scratch-rs

A neural network built from nothing — no dependencies, no linear algebra crate,
no autodiff. Pure `f64` arithmetic. The goal is understanding, not performance.

```sh
cargo run     # current demo
cargo test    # 24 tests, including numerical gradient checks
```

## Where we are

Working: a fully-connected feedforward network with sigmoid activations,
trained by backpropagation with batch gradient descent. It learns XOR from
random weights.

## Module map

| File | Holds |
|---|---|
| `src/neuron.rs` | `Neuron` — weights, bias, `forward`, and single-neuron gradients |
| `src/layer.rs` | `Layer` — neurons side by side; `backward` lives here |
| `src/network.rs` | `Network` — stacked layers, backprop, training loop |
| `src/loss.rs` | Squared error and its derivative |
| `src/rng.rs` | xorshift64* — only used to initialise weights |

## The path we took

Each commit is one concept, in order:

1. **`a851d91` one neuron** — `y = sigmoid(w·x + b)`. Hand-picked weights build
   AND, OR, NOT. A single neuron is a straight line through input space, and
   nothing more — so XOR is provably impossible.
2. **`67bfbfe` layers** — stack two layers, hand-derive XOR from
   `(x1 OR x2) AND NOT(x1 AND x2)`. The hidden layer re-plots the data into a
   space where the output neuron's straight line is finally enough.
3. **`49e95bc` learning** — squared-error loss, gradients by chain rule,
   gradient descent. One neuron teaches itself AND, arriving at `w≈[9.7, 9.7]`,
   `b≈-14.6` — essentially the weights that were hand-picked in step 1. The same
   code on XOR flatlines at loss 0.25, predicting 0.5 for everything.
4. **`4d3fae9` backpropagation** — XOR learned from random weights.

## Ideas worth not forgetting

**Why an activation function at all.** Without a non-linearity, stacked layers
collapse: a weighted sum of weighted sums is still one weighted sum, so depth
would buy nothing.

**`delta`.** `delta = dL/dz = dL/dout * sigmoid'(out)` — "how wrong this
neuron's pre-activation was". It appears in the single-neuron case and is
reused unchanged in backprop. Get comfortable with this one quantity and the
rest follows.

**`sigmoid'(z) = y(1-y)`.** The derivative in terms of the *output*, so `z`
never needs storing. Note it peaks at 0.25 and collapses to ~0 as `y`
approaches 0 or 1: a confident neuron barely learns. This is **saturation**,
and it is the reason ReLU exists.

**The backward signature is the whole algorithm.**

```
forward:    inputs        ──>  outputs
backward:   dL/d(outputs) ──>  dL/d(inputs)
```

A layer's inputs are the previous layer's outputs, so `backward`'s return value
is exactly what the previous layer needs. Chain them and the error signal walks
to the front. `Network::gradients` is a three-line reverse loop over that.

Backprop never asks *"what should this hidden neuron have output?"* — there is
no answer. It asks *"if this output had been slightly larger, would the loss
have risen or fallen?"*, which needs no target.

**Backprop finds *a* solution, not *your* solution.** The learned XOR hidden
layer is not OR/AND and is only partly interpretable. Any representation that
makes the cases linearly separable is acceptable, and gradient descent has no
preference for the legible one.

## Gotchas paid for already

**Always gradient-check.** Compare hand-derived gradients against
`(L(w+h) - L(w-h)) / 2h`. This caught a real bug here: per-sample loss was
averaged over outputs while the gradient seed was not, making every gradient
exactly `n_outputs` times too large.

The bug is worth remembering because **it still trained.** A uniformly scaled
gradient just behaves like a different learning rate, so XOR (1 output) passed
fine and nothing looked wrong. Wrong gradients don't crash — they cost you a
day of guessing at hyperparameters. See
`network::tests::backprop_matches_numerical_gradients`.

**Never initialise weights to zero.** Every neuron in a layer would compute the
same thing, receive the same gradient, and stay identical forever. Randomness
is what lets neurons specialise.

**Per-sample loss sums over outputs, it does not average.** That keeps
`dL/dy_j = 2(y_j - t_j)` exact. Averaging happens across the batch only.

## Possible next steps

1. **Real data (MNIST).** The biggest jump in realism — brings minibatching,
   train/test splits, and the first encounter with overfitting. Needs an IDX
   file parser, which is about 30 lines.
2. **ReLU + softmax + cross-entropy.** Fixes the saturation problem above, and
   is why nobody trains a real classifier with MSE. Cross-entropy's gradient
   through softmax simplifies beautifully to `(y - t)`.
3. **Refactor to matrices.** Collapse the per-neuron loops into matrix
   multiplication — faster, and much closer to how real frameworks are built.

Also unaddressed: momentum/Adam, regularisation, learning-rate schedules,
and any notion of a validation set.
