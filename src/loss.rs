//! Step 3a, part 1: measuring how wrong the network is.
//!
//! Learning needs a single number to minimise. That number is the *loss*:
//! low means "good predictions", high means "bad predictions". Everything
//! that follows — gradients, descent, backprop — is machinery for pushing
//! this one number down.
//!
//! We use squared error:
//!
//!     L = (output - target)^2
//!
//! Why squared, rather than just |output - target|?
//!
//!   * It is smooth everywhere. Absolute value has a kink at zero where the
//!     derivative doesn't exist, and our entire method is built on
//!     derivatives.
//!   * It punishes big mistakes disproportionately. Being wrong by 0.2 costs
//!     0.04; being wrong by 0.8 costs 0.64 — sixteen times as much, not four.
//!   * Its derivative is trivial: 2 * (output - target).

/// Squared error for one prediction.
pub fn squared_error(output: f64, target: f64) -> f64 {
    (output - target).powi(2)
}

/// d(squared_error) / d(output).
///
/// This is the *starting point* of every gradient calculation: it answers
/// "if the output nudged up by a little, how much would the loss change?"
/// Note the sign — when output is too high the derivative is positive, so
/// descent will push the output back down.
pub fn squared_error_derivative(output: f64, target: f64) -> f64 {
    2.0 * (output - target)
}

/// Mean squared error across a whole dataset.
///
/// Averaging (rather than summing) keeps the loss — and therefore a good
/// learning rate — independent of how many examples you have.
pub fn mse(outputs: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(
        outputs.len(),
        targets.len(),
        "got {} outputs but {} targets",
        outputs.len(),
        targets.len()
    );

    let total: f64 = outputs
        .iter()
        .zip(targets)
        .map(|(&o, &t)| squared_error(o, t))
        .sum();

    total / outputs.len() as f64
}

/// Smallest probability we will take the log of. `ln(0)` is -infinity, and a
/// softmax output can underflow to exactly 0.
const EPSILON: f64 = 1e-12;

/// Cross-entropy loss for one sample: `-sum_i t_i * ln(y_i)`.
///
/// With a one-hot target this reduces to `-ln(y_correct)` — the *only* term
/// that survives is the probability assigned to the right answer. Read it as
/// "how surprised was the network by the truth?":
///
///     y_correct = 1.00  ->  loss 0.00   (certain and right)
///     y_correct = 0.50  ->  loss 0.69
///     y_correct = 0.10  ->  loss 2.30
///     y_correct = 0.01  ->  loss 4.61   (confidently wrong, punished hard)
///
/// Compare squared error, whose worst case is bounded at 1.0 per output no
/// matter how badly wrong the network was. Cross-entropy grows without limit,
/// so confident mistakes produce big gradients instead of vanishing ones.
/// That is exactly the behaviour you want from a classifier.
pub fn cross_entropy(outputs: &[f64], targets: &[f64]) -> f64 {
    outputs
        .iter()
        .zip(targets)
        .map(|(&y, &t)| if t == 0.0 { 0.0 } else { -t * y.max(EPSILON).ln() })
        .sum()
}

/// `dL/dz` for softmax followed by cross-entropy — the pre-activation
/// gradient, not the output gradient.
///
/// Here is the payoff. Softmax's true derivative is a full Jacobian: every
/// output depends on every input. Cross-entropy's derivative is `-t/y`, which
/// blows up as `y` approaches 0. Composed by hand, both are unpleasant.
///
/// Chained together, almost everything cancels and what remains is:
///
///     dL/dz = y - t
///
/// Just the prediction minus the target. No Jacobian, no division, no
/// exponentials, nothing to overflow. This is *the* reason softmax and
/// cross-entropy are always used as a pair — separately they are awkward and
/// numerically fragile, together they are a subtraction.
///
/// Contrast the sigmoid + squared-error path, where `dL/dz` carries a factor
/// of `y(1-y)`. A confidently wrong sigmoid output has `y(1-y) ~ 0`, so it
/// barely learns from its worst mistakes. Softmax + cross-entropy has no such
/// factor: the more wrong it is, the harder it is pushed.
pub fn softmax_cross_entropy_gradient(output: f64, target: f64) -> f64 {
    output - target
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_prediction_costs_nothing() {
        assert_eq!(squared_error(0.7, 0.7), 0.0);
        assert_eq!(squared_error_derivative(0.7, 0.7), 0.0);
    }

    #[test]
    fn big_mistakes_cost_disproportionately_more() {
        let small = squared_error(0.5, 0.3); // off by 0.2
        let big = squared_error(1.0, 0.2); // off by 0.8, i.e. 4x
        assert!((big / small - 16.0).abs() < 1e-9, "should be 4^2 as costly");
    }

    #[test]
    fn derivative_points_away_from_the_target() {
        // Output too high -> positive derivative -> descent lowers it.
        assert!(squared_error_derivative(0.9, 0.1) > 0.0);
        // Output too low -> negative derivative -> descent raises it.
        assert!(squared_error_derivative(0.1, 0.9) < 0.0);
    }

    #[test]
    fn mse_averages() {
        // errors: 0.0, 0.04  ->  mean 0.02
        assert!((mse(&[0.5, 0.3], &[0.5, 0.5]) - 0.02).abs() < 1e-12);
    }

    #[test]
    fn cross_entropy_only_counts_the_true_class() {
        let target = [0.0, 1.0, 0.0];
        // Only y[1] matters; the other two are ignored entirely.
        let a = cross_entropy(&[0.1, 0.8, 0.1], &target);
        let b = cross_entropy(&[0.15, 0.8, 0.05], &target);
        assert!((a - b).abs() < 1e-12);
        assert!((a - -(0.8_f64).ln()).abs() < 1e-12);
    }

    #[test]
    fn cross_entropy_punishes_confident_mistakes_without_bound() {
        let target = [1.0, 0.0];

        assert!(cross_entropy(&[1.0, 0.0], &target) < 1e-9, "certain and right");
        assert!((cross_entropy(&[0.5, 0.5], &target) - 0.693).abs() < 0.001);
        assert!(cross_entropy(&[0.01, 0.99], &target) > 4.6);

        // Squared error, by contrast, is bounded at 1.0 per output however
        // wrong the answer is.
        assert!(squared_error(0.01, 1.0) < 1.0);
        assert!(cross_entropy(&[1e-10, 1.0], &target) > 20.0);
    }

    #[test]
    fn cross_entropy_does_not_blow_up_on_zero_probability() {
        let loss = cross_entropy(&[0.0, 1.0], &[1.0, 0.0]);
        assert!(loss.is_finite(), "got {loss}");
        assert!(loss > 25.0, "should still be a huge penalty");
    }

    #[test]
    fn softmax_cross_entropy_gradient_is_just_the_difference() {
        // Approximate, not exact: 0.7 - 1.0 is -0.30000000000000004 in binary
        // floating point.
        assert!((softmax_cross_entropy_gradient(0.7, 1.0) - -0.3).abs() < 1e-15);
        assert!((softmax_cross_entropy_gradient(0.2, 0.0) - 0.2).abs() < 1e-15);
        assert_eq!(softmax_cross_entropy_gradient(1.0, 1.0), 0.0);
    }

    /// Verify the "everything cancels" claim numerically: perturb a softmax
    /// input and check the cross-entropy loss moves by `y - t`.
    #[test]
    fn fused_gradient_matches_a_numerical_derivative() {
        use crate::activation::softmax_rows_in_place;
        use crate::matrix::Matrix;

        let scores = [0.4, -1.1, 2.0];
        let targets = [0.0, 1.0, 0.0];
        let h = 1e-6;

        let loss_at = |z: &[f64]| {
            let mut m = Matrix::row_vector(z);
            softmax_rows_in_place(&mut m);
            cross_entropy(&m.data, &targets)
        };

        let mut probabilities = Matrix::row_vector(&scores);
        softmax_rows_in_place(&mut probabilities);

        for i in 0..scores.len() {
            let mut up = scores;
            up[i] += h;
            let mut down = scores;
            down[i] -= h;

            let numerical = (loss_at(&up) - loss_at(&down)) / (2.0 * h);
            let analytic = softmax_cross_entropy_gradient(probabilities.data[i], targets[i]);

            assert!(
                (analytic - numerical).abs() < 1e-7,
                "index {i}: analytic {analytic} vs numerical {numerical}"
            );
        }
    }
}
