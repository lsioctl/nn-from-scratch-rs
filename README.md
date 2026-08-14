# NN from scratch, with Claude's help

I juste asked claude to teach me how neural networks works, buy building one, from scratch, from
one neuron

## Step 3a: make one neuron learn its own weights. No backprop yet — a single neuron lets you see the entire derivative at once.

A neuron just taught itself the AND gate. Clean build, 21 tests pass.

Look at what it learned

I hand-picked in step 1:   w = [10.000, 10.000]   b = -15.000
Gradient descent found:    w = [ 9.688,  9.688]   b = -14.618

Nobody told it about boolean algebra. It saw four (input, target) pairs and a rule for going downhill, and arrived at essentially the weights I'd derived on a napkin. That equivalence is the whole promise of the method.

How it works — three pieces

1. Loss (src/loss.rs) — one number to minimise. L = (output - target)². Squared rather than absolute because absolute value has a kink at zero where the derivative doesn't exist, and the entire method runs on derivatives.

2. The gradient (src/neuron.rs:88) — chain rule, twice:

dL/dwᵢ  =   dL/dy   ×   dy/dz   ×  dz/dwᵢ
        =  2(y - t) ×  y(1 - y) ×    xᵢ
             ↑            ↑           ↑
        how wrong    how steep    the input that
        the answer   the sigmoid  weight scaled

Two details worth keeping:

- sigmoid'(z) = y(1-y) — the derivative in terms of the output. You never need to store z or call exp twice.
- delta = 2(y-t) · y(1-y) is the shared prefix, meaning "how wrong this neuron's pre-activation was." Memorise this one. In backprop, delta is the thing that travels backwards.

3. The step — the gradient points uphill, so subtract it: w -= learning_rate * gradient.

The test that matters most

analytic_gradients_match_numerical_ones (src/neuron.rs:150) checks my calculus against the definition of a derivative — nudge a weight by h, see how much the loss actually moved:

dL/dw  ≈  (L(w+h) - L(w-h)) / 2h

This matters because a wrong gradient doesn't crash. It just quietly fails to learn, and you lose a day guessing at learning rates. Gradient-check every new layer type you write.

And the failure, which is the interesting half

Same code, same 20,000 epochs, one line of data changed:

epoch      0  |  0.27838
epoch   2500  |  0.25000
epoch  20000  |  0.25000   ← dead flat

learned:  w = [+0.000, +0.000]  b = -0.000
predictions:  0.5, 0.5, 0.5, 0.5

It didn't error. It didn't diverge. It drove its own weights to exactly zero and now answers 0.5 to everything — 0.25 is precisely the loss of hedging on all four examples. The gradients from the two →1 cases and the two →0 cases cancel perfectly, and it settles into paralysis.

This is step 1's geometric claim, arrived at empirically. It's also what real training failure looks like: not a crash, just a loss curve that flatlines somewhere useless.

recap: You're building a neural network from scratch in Rust, step by step. A single neuron now learns AND on its own via gradient descent, but stalls on XOR. Next up is step 3b: backpropagation through layers, then retraining XOR. (disable recaps in /config)

Step 3b: backpropagation. The problem to solve — a hidden neuron has no target. Nobody can say what the OR neuron "should" have output.

XOR learned from random weights. Loss 0.32 → 0.000024. Clean build, 24 tests pass.

The one idea

A hidden neuron has no target — that was the blocker. Backprop sidesteps it entirely. It never asks "what should this neuron have output?" It asks:

▎ If this neuron's output had been slightly larger, would the loss have gone up or down?

That question needs no target, only the chain rule. And the answer for a hidden neuron is assembled from the neurons it feeds: blame flows back through the same weights the signal came forward through, in proportion to their size. A neuron connected by a big weight caused more of the error, so it gets more of the blame.

The shape that makes it work

Layer::backward is the whole trick, and it's in the signature:

forward:    inputs        ──>  outputs
backward:   dL/d(outputs) ──>  dL/d(inputs)

A layer is handed "how much the loss cares about each of my outputs" and returns "how much the loss cares about each of my inputs." Since its inputs are the previous layer's outputs, the return value is exactly what the previous layer needs as its input. Chain them and the signal walks to the front on its own.

Network::gradients (src/network.rs:110) is then almost anticlimactic:

let mut dl_doutputs = /* 2(y - t) — the ONLY place a target is used */;

for (k, layer) in self.layers.iter().enumerate().rev() {
    let (layer_gradients, dl_dinputs) =
        layer.backward(&activations[k], &activations[k + 1], &dl_doutputs);
    gradients.push(layer_gradients);
    dl_doutputs = dl_dinputs;   // hand it to the layer behind
}

Three lines of loop. That's backpropagation.

Per neuron it's the same delta from step 3a — delta = dL/dout × sigmoid'(out) — used twice: once for its own gradients (delta × input), once to send backwards (delta × weight). Nothing new was invented for the multi-layer case.

The gradient check earned its keep

It failed on the first run:

layer 0 neuron 0 weight 0: analytic -0.02697 vs numerical -0.01348

Exactly 2×, on a network with 2 outputs. I was averaging the per-sample loss over outputs while seeding the gradient with an unaveraged 2(y-t). Every gradient was off by a constant factor.

Worth dwelling on: that bug would still have trained. A uniformly scaled gradient just acts like a different learning rate — XOR (1 output) passed fine. It would have surfaced later as "why does this need a weird learning rate," on a bigger network, with no clue where to look. Numerical gradient checking is how you avoid losing a day to that.

I fixed it by summing over outputs rather than averaging, which keeps dL/dyⱼ = 2(yⱼ - tⱼ) exact with no bookkeeping.

What it invented

   x1   x2  |    h0     h1     h2     h3
     0    0  |  0.28   0.89   0.61   0.05
     0    1  |  0.26   0.04   0.02   0.00
     1    0  |  0.70   1.00   0.05   0.96
     1    1  |  0.68   0.98   0.00   0.01

Not OR and AND. h2 fires only for (0,0), h3 only for (1,0), h0 roughly tracks x1. It's a representation I wouldn't have designed and can only partly read — and it works as well as my hand-derived one.

That's the real lesson of step 3. Backprop doesn't find your solution; it finds one of the many that work. Any hidden layer making the four cases linearly separable is acceptable, and gradient descent has no reason to prefer the interpretable one. At scale this is why network internals resist explanation.

---
The core is done: you have a working neural network, from scratch, no dependencies. Natural next steps, roughly in order of payoff:

- Real data (MNIST digits) — where you meet minibatching, train/test splits, and overfitting. The biggest step up in realism.
- Better activations (ReLU) and softmax + cross-entropy — fixes the saturation problem you saw in sigmoid_derivative, and the reason nobody trains classifiers with MSE.
- Refactor to matrices — collapse the per-neuron loops into matrix ops; faster, and closer to how real frameworks are written.
