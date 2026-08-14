//! Activation functions.
//!
//! These used to live in `neuron.rs`, back when the neuron was the central
//! abstraction. Now that layers are matrices, the activation is a function
//! applied elementwise to a whole batch at once, and belongs on its own.

/// The logistic "sigmoid".
///
/// Squashes any real number into the open range (0, 1):
///   sigmoid(-inf) -> 0     sigmoid(0) = 0.5     sigmoid(+inf) -> 1
///
/// Without a non-linearity here, stacked layers would collapse: a weighted
/// sum of weighted sums is still one weighted sum, and depth would buy
/// nothing.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The derivative of sigmoid, in terms of its *output*.
///
///     sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
///
/// so given `y = sigmoid(z)`, the slope is `y * (1 - y)` — no need to keep
/// `z` around, and no second call to `exp`.
///
/// It peaks at 0.25 when y = 0.5 and collapses toward 0 as y approaches 0
/// or 1: a confident neuron barely learns. This is **saturation**.
pub fn sigmoid_derivative_from_output(y: f64) -> f64 {
    y * (1.0 - y)
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
        assert!(sigmoid_derivative_from_output(0.01) < 0.01);
    }

    /// The closed form must agree with a numerical derivative.
    #[test]
    fn derivative_matches_a_numerical_estimate() {
        let h = 1e-6;
        for z in [-3.0, -0.5, 0.0, 0.5, 3.0] {
            let numerical = (sigmoid(z + h) - sigmoid(z - h)) / (2.0 * h);
            let analytic = sigmoid_derivative_from_output(sigmoid(z));
            assert!((analytic - numerical).abs() < 1e-7, "at z = {z}");
        }
    }
}
