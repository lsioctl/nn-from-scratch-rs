//! Activation functions.
//!
//! Up to now every layer used sigmoid, which has two problems for a deep-ish
//! classifier:
//!
//!   * **Saturation.** `sigmoid'(y) = y(1-y)` peaks at 0.25 and collapses to
//!     ~0 when the neuron is confident. Gradients are multiplied by this at
//!     every layer on the way back, so they shrink fast. This is the
//!     "vanishing gradient" problem, and it is why deep sigmoid networks were
//!     nearly untrainable before ~2010.
//!   * **Outputs are not probabilities.** Ten independent sigmoids can all say
//!     0.9. Nothing makes them a distribution over digits.
//!
//! ReLU fixes the first, softmax the second.

use crate::matrix::Matrix;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Sigmoid,
    Relu,
    /// Only valid on the output layer, and only paired with cross-entropy.
    Softmax,
}

/// The logistic sigmoid: squashes into (0, 1).
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of sigmoid in terms of its output. Peaks at 0.25.
pub fn sigmoid_derivative_from_output(y: f64) -> f64 {
    y * (1.0 - y)
}

/// The rectifier: `max(0, x)`. Embarrassingly simple, and better.
///
/// Its derivative is **1** for any positive input — not 0.25, not less. The
/// gradient passes back through untouched, however many layers deep. That one
/// property is most of why modern networks can be deep at all.
///
/// The cost is that negative inputs get a derivative of exactly 0, so a
/// neuron pushed firmly negative for every input stops learning permanently.
/// This is a "dead" neuron. In practice enough survive that it does not
/// matter, and variants like leaky ReLU exist for when it does.
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Derivative of ReLU, from its output.
///
/// Works from the output because `relu(z) > 0` exactly when `z > 0`. At
/// exactly 0 the function has a kink and no true derivative; everyone picks 0
/// and moves on.
pub fn relu_derivative_from_output(y: f64) -> f64 {
    if y > 0.0 { 1.0 } else { 0.0 }
}

/// Softmax, applied to each row independently.
///
///     softmax(z)_i = exp(z_i) / sum_j exp(z_j)
///
/// This turns ten arbitrary scores into ten positive numbers that sum to 1 —
/// an actual probability distribution over the digits. Unlike ten separate
/// sigmoids, the outputs now *compete*: raising one lowers the others.
///
/// **The max subtraction is not optional.** `exp(1000)` is infinity in f64,
/// and a network in mid-training can easily produce a score that large.
/// Subtracting the row maximum before exponentiating leaves the result
/// mathematically identical — the constant cancels between numerator and
/// denominator — but keeps every exponent at 0 or below, so the largest term
/// is exp(0) = 1 and nothing overflows.
pub fn softmax_rows_in_place(matrix: &mut Matrix) {
    for r in 0..matrix.rows {
        let row = matrix.row_mut(r);

        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let mut total = 0.0;
        for value in row.iter_mut() {
            *value = (*value - max).exp();
            total += *value;
        }
        for value in row.iter_mut() {
            *value /= total;
        }
    }
}

impl Activation {
    /// Apply this activation to a batch of pre-activations, in place.
    pub fn apply(&self, matrix: &mut Matrix) {
        match self {
            Activation::Sigmoid => matrix.map_in_place(sigmoid),
            Activation::Relu => matrix.map_in_place(relu),
            Activation::Softmax => softmax_rows_in_place(matrix),
        }
    }

    /// Elementwise derivative, given the activation's output.
    ///
    /// Softmax has no elementwise derivative — each output depends on every
    /// input, so the true derivative is a full Jacobian per row. We never need
    /// it, because pairing softmax with cross-entropy collapses the whole
    /// thing to `y - t`. See `loss::softmax_cross_entropy_gradient`.
    pub fn derivative_from_output(&self, y: f64) -> f64 {
        match self {
            Activation::Sigmoid => sigmoid_derivative_from_output(y),
            Activation::Relu => relu_derivative_from_output(y),
            Activation::Softmax => {
                unreachable!("softmax's gradient is handled jointly with cross-entropy")
            }
        }
    }

    /// A sensible weight-initialisation range, `[-limit, limit)`.
    ///
    /// This matters far more than it looks. Our old `uniform(-1, 1)` ignored
    /// layer width entirely: with 784 inputs, the pre-activations start out
    /// enormous, which pins sigmoid in saturation and makes ReLU explode.
    ///
    /// The fix is to shrink the range as the layer gets wider, so the variance
    /// of the weighted sum stays around 1 regardless of shape:
    ///
    ///   * **He** (`sqrt(6 / fan_in)`) for ReLU, which zeroes half its inputs
    ///     and so needs larger weights to compensate.
    ///   * **Xavier/Glorot** (`sqrt(6 / (fan_in + fan_out))`) for the
    ///     symmetric, saturating activations.
    pub fn init_limit(&self, fan_in: usize, fan_out: usize) -> f64 {
        match self {
            Activation::Relu => (6.0 / fan_in as f64).sqrt(),
            _ => (6.0 / (fan_in + fan_out) as f64).sqrt(),
        }
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
    fn sigmoid_derivative_peaks_in_the_middle() {
        assert_eq!(sigmoid_derivative_from_output(0.5), 0.25);
        assert!(sigmoid_derivative_from_output(0.99) < 0.01);
    }

    #[test]
    fn relu_clips_negatives_and_passes_positives() {
        assert_eq!(relu(-3.0), 0.0);
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(2.5), 2.5);
    }

    /// The point of ReLU: gradient 1, not 0.25.
    #[test]
    fn relu_gradient_does_not_shrink() {
        assert_eq!(relu_derivative_from_output(0.001), 1.0);
        assert_eq!(relu_derivative_from_output(1000.0), 1.0);
        assert_eq!(relu_derivative_from_output(0.0), 0.0, "dead side");
    }

    #[test]
    fn activation_derivatives_match_numerical_estimates() {
        let h = 1e-6;
        for z in [-3.0, -0.5, 0.5, 3.0] {
            let numerical = (sigmoid(z + h) - sigmoid(z - h)) / (2.0 * h);
            assert!((sigmoid_derivative_from_output(sigmoid(z)) - numerical).abs() < 1e-7);

            // Skip z = 0, where ReLU has a kink and no derivative exists.
            let numerical = (relu(z + h) - relu(z - h)) / (2.0 * h);
            assert!((relu_derivative_from_output(relu(z)) - numerical).abs() < 1e-7);
        }
    }

    #[test]
    fn softmax_rows_become_probability_distributions() {
        let mut m = Matrix::from_rows(&[vec![1.0, 2.0, 3.0], vec![0.0, 0.0, 0.0]]);
        softmax_rows_in_place(&mut m);

        for r in 0..m.rows {
            let sum: f64 = m.row(r).iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "row {r} sums to {sum}");
            assert!(m.row(r).iter().all(|&p| p > 0.0 && p < 1.0));
        }

        // Equal scores -> equal probabilities.
        assert!(m.row(1).iter().all(|&p| (p - 1.0 / 3.0).abs() < 1e-12));
        // Bigger score -> bigger probability, order preserved.
        assert!(m.get(0, 0) < m.get(0, 1) && m.get(0, 1) < m.get(0, 2));
    }

    /// Without the max subtraction this row would be exp(1000) = inf,
    /// then inf/inf = NaN.
    #[test]
    fn softmax_survives_enormous_scores() {
        let mut m = Matrix::from_rows(&[vec![1000.0, 1001.0, 1002.0]]);
        softmax_rows_in_place(&mut m);

        assert!(m.data.iter().all(|p| p.is_finite()), "got {:?}", m.data);
        assert!((m.row(0).iter().sum::<f64>() - 1.0).abs() < 1e-12);

        // Must equal softmax([0, 1, 2]) — shifting scores changes nothing.
        let mut shifted = Matrix::from_rows(&[vec![0.0, 1.0, 2.0]]);
        softmax_rows_in_place(&mut shifted);
        for (a, b) in m.data.iter().zip(&shifted.data) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn init_limits_shrink_as_layers_widen() {
        let narrow = Activation::Relu.init_limit(10, 10);
        let wide = Activation::Relu.init_limit(1000, 10);
        assert!(wide < narrow, "wider fan-in must mean smaller weights");

        // He is larger than Xavier for the same shape — ReLU zeroes half its
        // inputs, so it needs more to compensate.
        assert!(Activation::Relu.init_limit(100, 100) > Activation::Sigmoid.init_limit(100, 100));
    }
}
