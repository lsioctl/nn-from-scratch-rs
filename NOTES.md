# nn-from-scratch-rs

A neural network built from nothing — no dependencies, no linear algebra crate,
no autodiff. Pure `f64` arithmetic. The goal is understanding, not performance.

```sh
./fetch-mnist.sh        # one-off, downloads ~11 MB into ./data (gitignored)
cargo run --release     # trains BOTH networks on MNIST — takes ~2 min
cargo test              # 59 tests, including numerical gradient checks
```

`--release` is not optional for MNIST: debug builds are roughly 30x slower here.

## Where we are

A fully-connected feedforward network trained by backpropagation with
minibatch gradient descent, implemented with matrices — a whole minibatch goes
through in one matrix product.

**98.05% test accuracy on MNIST** — `[784, 100, 10]`, ReLU hidden + softmax
output + cross-entropy, 30 epochs, batch 32, learning rate 0.3, ~46 s.

`main` trains both the old and new setup side by side from the same seed:

| | accuracy | errors | time |
|---|---|---|---|
| sigmoid + squared error | 97.78% | 222 | 73 s |
| ReLU + softmax + cross-entropy | **98.05%** | **195** | **46 s** |

## Module map

| File | Holds |
|---|---|
| `src/matrix.rs` | `Matrix` — row-major dense f64, `matmul`, `transpose`, … |
| `src/activation.rs` | `Activation` enum — sigmoid, ReLU, softmax, plus init scales |
| `src/layer.rs` | `Layer` — a weight matrix + biases; forward and backward |
| `src/network.rs` | `Network` — stacked layers, backprop, training loop |
| `src/loss.rs` | Squared error and cross-entropy, and their derivatives |
| `src/mnist.rs` | IDX file parser, one-hot encoding, ASCII digit rendering |
| `src/rng.rs` | xorshift64* — weight init and shuffling |
| `src/neuron.rs` | **Reference only.** The original one-neuron-at-a-time code |

`neuron.rs` is no longer live. It is kept because it is the clearest statement
of what a neuron *is*, and because it is an executable specification:
`layer::tests::matrix_layer_agrees_with_the_neuron_implementation` builds the
same layer both ways and demands identical outputs.

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
6. **Matrix refactor** — same algorithm, same accuracy, 3.6x faster.
7. **ReLU + softmax + cross-entropy** — 98.05%, and 1.6x faster per epoch.

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

**Train/test split.** The classifier reaches **100.00%** on training data by
epoch 20 while test accuracy sits at 98.05% and stops moving. That is
**overfitting** made unusually vivid: training loss keeps falling (0.00104 ->
0.00051) and buys nothing at all. Once training accuracy is saturated, the
gradient is only sharpening answers it already has right. The test set is the
only honest number, which is why it must never be trained on.

**The matrix formulation.** With `B` = batch, `I` = inputs, `O` = neurons:

```
X (B,I)   W (I,O)   b (O)

forward:   Z = X·W + b        A = sigmoid(Z)
backward:  dZ = dA ⊙ sigmoid'(A)
           dW = Xᵀ·dZ         db = column sums of dZ
           dX = dZ·Wᵀ
```

`W` is stored so that a *column* is one neuron's weights — the transpose of
the list-of-neurons view. That is what makes `X·W` work out.

The two transposes are not arbitrary. Forward, `X·W` contracts over inputs;
going backward you must contract over the *other* index each time, and
transposing is how you say that. Useful check when deriving: only one
arrangement of each product has shapes that line up at all.

**Loop order in `matmul` is most of the performance.** `i,k,j` beats the
textbook `i,j,k` several times over. With `i,j,k` the inner loop reads
`rhs[k][j]` for successive `k`, striding `n` floats through memory each step.
With `i,k,j` it walks one contiguous row of `rhs` and one of the output,
accumulating into them — cache-friendly and vectorisable.

Skipping zero multipliers in `matmul` is worth real time on MNIST, where ~80%
of every image is blank background.

**ReLU.** `max(0, x)`, derivative **1** for any positive input rather than
sigmoid's 0.25 peak. The gradient passes back untouched however deep the
network, instead of being multiplied by <=0.25 at every layer. That single
property is most of why deep networks became trainable. The cost: a neuron
pushed firmly negative gets derivative 0 and stops learning permanently — a
"dead" neuron. Enough survive that it rarely matters.

**Softmax** turns arbitrary scores into a distribution: all positive, summing
to 1. Unlike ten independent sigmoids, the outputs now *compete* — raising one
lowers the others. Always subtract the row max before exponentiating; `exp(1000)`
is `inf` and a mid-training network will produce scores that large. Subtracting
a constant leaves the result mathematically identical because it cancels
between numerator and denominator.

**Cross-entropy** is `-ln(y_correct)` for one-hot targets — only the
probability of the true answer matters. It is unbounded, so a confident mistake
costs 4.6 (at p=0.01) or 20+ (at p=1e-10), where squared error is capped at 1.0
per output no matter how wrong.

**The fusion is the whole point.** Softmax's true derivative is a full Jacobian;
cross-entropy's is `-t/y`, which explodes as `y -> 0`. Composed, everything
cancels:

```
dL/dz = y - t
```

Prediction minus target. No Jacobian, no division, nothing to overflow. This is
why the two are *always* used as a pair. Contrast sigmoid + squared error, whose
`dL/dz` carries a factor of `y(1-y)` — so a confidently wrong output barely
learns from its worst mistake. Softmax + cross-entropy has no such factor: the
more wrong it is, the harder it is pushed.

Structurally this needed `Layer::backward` split in two, since softmax has no
elementwise derivative to apply. `backward_from_pre_activation` takes `dL/dz`
directly; everything after that point is identical for every activation.

**Weight initialisation scale matters far more than it looks.** The original
`uniform(-1, 1)` ignored layer width entirely. Switching to width-aware ranges
— He `sqrt(6/fan_in)` for ReLU, Xavier `sqrt(6/(fan_in+fan_out))` otherwise —
lifted even the *unchanged* sigmoid network from 95.4% to 97.8%. Bad init does
not announce itself; it just quietly costs you two points.

**Cross-entropy needs a much smaller learning rate** — 0.3 here versus 3.0 for
squared error — precisely because its gradients are not shrunk by a `y(1-y)`
factor.

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

1. **Fight the overfitting**, now the clearest bottleneck — training accuracy
   is pinned at 100% while test sits at 98%. L2 regularisation, dropout, or
   early stopping on a validation split carved out of the training data.
2. **Momentum or Adam.** Plain SGD with a fixed learning rate is the last
   genuinely old-fashioned piece left. Momentum is ~5 lines and usually worth
   half a point; a learning-rate schedule would also stop the late-training
   wobble.
3. **Convolutions.** The architecture that actually suits images — weights
   shared across positions, so a stroke detector learned in one corner works
   everywhere. This is the step from ~98% to ~99.3%, and a big one.
4. **Save and load trained weights.** Every run currently starts from scratch.
5. **Go faster still.** Parallelise `matmul` across rows with threads; cut the
   allocation churn (`backward` clones two matrices per layer per batch).

Also unaddressed: batch normalisation, data augmentation, and any principled
hyperparameter search — the learning rates here were found by trying a few.
