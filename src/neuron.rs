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

/// The logistic "sigmoid" activation.
///
/// Squashes any real number into the open range (0, 1):
///   sigmoid(-inf) -> 0     sigmoid(0) = 0.5     sigmoid(+inf) -> 1
///
/// Why squash at all? Because without it, stacking neurons would be
/// pointless: a sum of sums of sums is still just one big weighted sum, so
/// a 100-layer network would collapse into something a single layer could
/// do. The non-linearity is what makes depth worth anything.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

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
