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
