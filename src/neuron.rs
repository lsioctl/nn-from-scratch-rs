//! Step 1: a single neuron.
//!
//! A neuron is a very small machine. It takes several numbers in, and puts
//! one number out. It does this in two stages:
//!
//!   1. Weighted sum:  z = w1*x1 + w2*x2 + ... + wn*xn + b
//!   2. Activation:    y = f(z)
//!
//! That's the whole thing. The `weights` say how much each input matters
//! (and a negative weight means "this input argues against firing"). The
//! `bias` shifts how eager the neuron is to fire at all.

//! NOTE: since the matrix refactor this is no longer the working
//! implementation — `layer.rs` is. It is kept because it is the clearest
//! statement of what a neuron *is*, and because it serves as an executable
//! specification: `layer::tests::matrix_layer_agrees_with_the_neuron_implementation`
//! builds the same layer both ways and demands identical outputs.

use crate::activation::{sigmoid, sigmoid_derivative_from_output};
use crate::loss::{squared_error, squared_error_derivative};
use crate::rng::Rng;

// `sigmoid` and `sigmoid_derivative_from_output` used to live here. They now
// live in `activation.rs`, since an activation applies to a whole batch and is
// not really a property of an individual neuron.

/// A single neuron: one weight per input, plus one bias.
#[derive(Debug, Clone)]
pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl Neuron {
    pub fn new(weights: Vec<f64>, bias: f64) -> Self {
        Self { weights, bias }
    }

    /// Stage 1: the weighted sum, *before* the activation is applied.
    ///
    /// This value is usually called `z`, the "pre-activation" or "logit".
    /// We give it its own method because later, when we do backpropagation,
    /// we will need it separately from the final output.
    pub fn net_input(&self, inputs: &[f64]) -> f64 {
        assert_eq!(
            inputs.len(),
            self.weights.len(),
            "neuron has {} weights but got {} inputs",
            self.weights.len(),
            inputs.len()
        );

        // zip pairs up (weight, input), map multiplies each pair,
        // sum adds them all together. Then add the bias.
        self.weights
            .iter()
            .zip(inputs)
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + self.bias
    }

    /// Stage 1 + stage 2: the neuron's actual output.
    pub fn forward(&self, inputs: &[f64]) -> f64 {
        sigmoid(self.net_input(inputs))
    }

    /// A neuron with random weights, ready to be trained.
    ///
    /// Small random values, not zeros. Zeros would be a disaster in a *layer*:
    /// every neuron would compute the same thing, receive the same gradient,
    /// and stay identical forever. Randomness is what lets neurons specialise.
    pub fn random(n_inputs: usize, rng: &mut Rng) -> Self {
        Self {
            weights: (0..n_inputs).map(|_| rng.uniform(-1.0, 1.0)).collect(),
            bias: rng.uniform(-1.0, 1.0),
        }
    }

    /// How much the loss would change if each weight (and the bias) nudged up.
    ///
    /// This is the chain rule, applied twice. The loss depends on `y`, which
    /// depends on `z`, which depends on each `w`:
    ///
    ///     dL/dw_i = dL/dy  *  dy/dz  *  dz/dw_i
    ///             = 2(y-t) * y(1-y) *  x_i
    ///                 |         |        |
    ///          loss slope   sigmoid   the input that
    ///                        slope    weight was scaling
    ///
    /// The bias is the same story with `dz/db = 1`, since the bias is added
    /// rather than multiplied by anything:
    ///
    ///     dL/db = 2(y-t) * y(1-y)
    ///
    /// That shared prefix `dL/dz = 2(y-t) * y(1-y)` gets its own name:
    /// **delta**. It means "how wrong this neuron's pre-activation was".
    /// Remember it — in backpropagation, delta is the quantity that travels
    /// backwards through the network.
    ///
    /// Returns `(weight gradients, bias gradient)`.
    pub fn gradients(&self, inputs: &[f64], target: f64) -> (Vec<f64>, f64) {
        let y = self.forward(inputs);

        let delta = squared_error_derivative(y, target) * sigmoid_derivative_from_output(y);

        // Each weight's gradient is delta scaled by the input it multiplies.
        // A weight attached to an input of 0 gets no gradient at all — it had
        // no influence on this prediction, so it earns no blame.
        let weight_gradients = inputs.iter().map(|x| delta * x).collect();

        (weight_gradients, delta)
    }

    /// Take one step *downhill*.
    ///
    /// The gradient points in the direction that *increases* the loss, so we
    /// subtract it. `learning_rate` controls the step size: too small and
    /// training crawls, too large and you overshoot the valley and diverge.
    pub fn apply_gradients(&mut self, weight_gradients: &[f64], bias_gradient: f64, learning_rate: f64) {
        for (w, g) in self.weights.iter_mut().zip(weight_gradients) {
            *w -= learning_rate * g;
        }
        self.bias -= learning_rate * bias_gradient;
    }

    /// One full pass over the dataset: measure, average, step once.
    ///
    /// This is "batch" gradient descent — we accumulate gradients from every
    /// example before changing anything. Averaging over the batch gives a
    /// smoother, more trustworthy direction than reacting to one example at
    /// a time.
    ///
    /// Returns the mean loss *before* this update, which is what you plot to
    /// watch training progress.
    pub fn train_epoch(&mut self, samples: &[(Vec<f64>, f64)], learning_rate: f64) -> f64 {
        let mut summed_weight_gradients = vec![0.0; self.weights.len()];
        let mut summed_bias_gradient = 0.0;
        let mut total_loss = 0.0;

        for (inputs, target) in samples {
            total_loss += squared_error(self.forward(inputs), *target);

            let (weight_gradients, bias_gradient) = self.gradients(inputs, *target);
            for (acc, g) in summed_weight_gradients.iter_mut().zip(&weight_gradients) {
                *acc += g;
            }
            summed_bias_gradient += bias_gradient;
        }

        let n = samples.len() as f64;
        for g in &mut summed_weight_gradients {
            *g /= n;
        }
        self.apply_gradients(&summed_weight_gradients, summed_bias_gradient / n, learning_rate);

        total_loss / n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_is_centred_on_zero() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert!(sigmoid(-10.0) < 0.001);
        assert!(sigmoid(10.0) > 0.999);
    }

    #[test]
    fn net_input_is_a_weighted_sum() {
        let n = Neuron::new(vec![2.0, -3.0], 1.0);
        // 2*1 + (-3)*4 + 1 = 2 - 12 + 1 = -9
        assert_eq!(n.net_input(&[1.0, 4.0]), -9.0);
    }

    /// The single most useful test in this whole project.
    ///
    /// We derived `gradients` with calculus on paper. Here we check it against
    /// the *definition* of a derivative — nudge a weight by a tiny amount and
    /// see how much the loss actually moved:
    ///
    ///     dL/dw  ~=  (L(w + h) - L(w - h)) / 2h
    ///
    /// If the hand-derived formula disagrees with reality, this catches it.
    /// Any time you add a new layer or activation, gradient-check it first —
    /// a wrong gradient doesn't crash, it just quietly fails to learn.
    #[test]
    fn analytic_gradients_match_numerical_ones() {
        let neuron = Neuron::new(vec![0.3, -0.8], 0.15);
        let inputs = [0.7, 0.4];
        let target = 1.0;

        let (grad_w, grad_b) = neuron.gradients(&inputs, target);
        let h = 1e-6;

        for i in 0..neuron.weights.len() {
            let mut up = neuron.clone();
            up.weights[i] += h;
            let mut down = neuron.clone();
            down.weights[i] -= h;

            let numerical = (squared_error(up.forward(&inputs), target)
                - squared_error(down.forward(&inputs), target))
                / (2.0 * h);

            assert!(
                (grad_w[i] - numerical).abs() < 1e-7,
                "weight {i}: analytic {} vs numerical {numerical}",
                grad_w[i]
            );
        }

        let mut up = neuron.clone();
        up.bias += h;
        let mut down = neuron.clone();
        down.bias -= h;
        let numerical = (squared_error(up.forward(&inputs), target)
            - squared_error(down.forward(&inputs), target))
            / (2.0 * h);
        assert!((grad_b - numerical).abs() < 1e-7);
    }

    #[test]
    fn sigmoid_derivative_peaks_in_the_middle() {
        // Steepest where the neuron is undecided...
        assert_eq!(sigmoid_derivative_from_output(0.5), 0.25);
        // ...and nearly flat where it is confident. This is saturation.
        assert!(sigmoid_derivative_from_output(0.99) < 0.01);
        assert!(sigmoid_derivative_from_output(0.01) < 0.01);
    }

    #[test]
    fn gradient_descent_learns_the_and_gate() {
        let samples = vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 0.0),
            (vec![1.0, 0.0], 0.0),
            (vec![1.0, 1.0], 1.0),
        ];

        let mut rng = Rng::new(42);
        let mut neuron = Neuron::random(2, &mut rng);

        let first_loss = neuron.train_epoch(&samples, 5.0);
        let mut last_loss = first_loss;
        for _ in 0..20_000 {
            last_loss = neuron.train_epoch(&samples, 5.0);
        }

        assert!(last_loss < first_loss, "loss should go down");
        assert!(last_loss < 0.01, "should learn AND, got loss {last_loss}");

        for (inputs, target) in &samples {
            let got = neuron.forward(inputs);
            assert!((got - target).abs() < 0.2, "{inputs:?} -> {got}, want {target}");
        }
    }

    /// The empirical version of the claim from step 1: one neuron is one line,
    /// and no line solves XOR. Training does not fail with an error — it just
    /// stalls, hedging at 0.5 for every input.
    #[test]
    fn gradient_descent_cannot_learn_xor() {
        let samples = vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 1.0),
            (vec![1.0, 0.0], 1.0),
            (vec![1.0, 1.0], 0.0),
        ];

        let mut rng = Rng::new(42);
        let mut neuron = Neuron::random(2, &mut rng);

        let mut loss = 0.0;
        for _ in 0..20_000 {
            loss = neuron.train_epoch(&samples, 5.0);
        }

        // 0.25 is exactly the loss of giving up and answering 0.5 every time.
        assert!(loss > 0.2, "expected it to stall near 0.25, got {loss}");
        for (inputs, _) in &samples {
            assert!((neuron.forward(inputs) - 0.5).abs() < 0.1);
        }
    }

    #[test]
    fn hand_built_and_gate() {
        // Fires only when BOTH inputs are 1.
        let and = Neuron::new(vec![10.0, 10.0], -15.0);
        assert!(and.forward(&[0.0, 0.0]) < 0.01);
        assert!(and.forward(&[1.0, 0.0]) < 0.01);
        assert!(and.forward(&[0.0, 1.0]) < 0.01);
        assert!(and.forward(&[1.0, 1.0]) > 0.99);
    }
}
