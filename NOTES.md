# nn-from-scratch-rs

A neural network built from nothing — no dependencies, no linear algebra crate,
no autodiff. Pure `f64` arithmetic. The goal is understanding, not performance.

```sh
./fetch-mnist.sh        # one-off, downloads ~11 MB into ./data (gitignored)
cargo run --release     # trains on MNIST — takes ~100s, do NOT use debug
cargo test              # 34 tests, including numerical gradient checks
```

`--release` is not optional for MNIST: debug builds are roughly 30x slower here.

## Where we are

A fully-connected feedforward network with sigmoid activations, trained by
backpropagation with minibatch gradient descent.

**95.63% test accuracy on MNIST** — `[784, 30, 10]`, 23,860 parameters,
30 epochs, batch size 32, learning rate 3.0, ~100 seconds.

## Module map

| File | Holds |
|---|---|
| `src/neuron.rs` | `Neuron` — weights, bias, `forward`, and single-neuron gradients |
| `src/layer.rs` | `Layer` — neurons side by side; `backward` lives here |
| `src/network.rs` | `Network` — stacked layers, backprop, training loop |
| `src/loss.rs` | Squared error and its derivative |
| `src/mnist.rs` | IDX file parser, one-hot encoding, ASCII digit rendering |
| `src/rng.rs` | xorshift64* — weight init and shuffling |

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
5. **MNIST** — real data. 95.63% on the held-out test set.

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

**Minibatching.** Full-batch descent computes a perfect gradient and takes
*one* step with it. Over 60,000 examples that is one update per pass. Batches
of 32 give ~1,875 updates per pass from noisier gradients, and the noise is
affordable — even mildly useful for escaping shallow local minima. Shuffle
every epoch: MNIST is not stored in random order.

**Train/test split.** Training accuracy reached 98.06% while test accuracy
stalled at ~95.6% and wobbled after epoch ~17. That ~2.4-point gap is
**overfitting** — the network memorising training specifics that do not
generalise. The test set is the only honest number, which is why it must never
be trained on.

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

**Scale your inputs.** Pixels are divided by 255 in `mnist.rs`. Feeding raw
0-255 values would produce huge weighted sums, pin every sigmoid at 0 or 1
where its derivative vanishes, and the network would simply not learn.

**Look at the data before trusting the parser.** `mnist::render` prints a digit
as ASCII art. An off-by-one or transposed parse yields plausible numbers and a
network that mysteriously won't train.

**One-hot the labels.** Ten outputs, ten targets. Regressing the raw digit with
one output would imply 4 is "nearly right" when the answer is 3 — digits have
no such ordering.

## Possible next steps

1. **ReLU + softmax + cross-entropy.** The biggest accuracy win available.
   Fixes saturation, and is why nobody trains a classifier with MSE.
   Cross-entropy's gradient through softmax simplifies beautifully to `(y - t)`
   — cleaner than what we have now. Should reach ~97-98%.
2. **Refactor to matrices.** Collapse the per-neuron loops into matrix
   multiplication — far faster than the current ~3.3 s/epoch, and much closer
   to how real frameworks are built.
3. **Fight the overfitting** seen above: L2 regularisation, dropout, or just
   early stopping on a proper validation split carved out of the training set.

Also unaddressed: momentum/Adam, learning-rate schedules, convolutions (the
architecture that actually suits images), and saving/loading trained weights —
every run currently starts from scratch.
