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
}
