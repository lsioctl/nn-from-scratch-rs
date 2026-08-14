//! Step 2a: a layer — several neurons side by side.
//!
//! A layer is just a `Vec<Neuron>` where every neuron sees the *same* inputs
//! and produces its own output. There is no communication between neurons
//! within a layer; they are completely independent.
//!
//!            x1 ─┬──────> [neuron 0] ──> y0
//!                │    ┌─> [neuron 1] ──> y1
//!            x2 ─┴────┴─> [neuron 2] ──> y2
//!
//! So if one neuron draws one line through the input space (step 1), a layer
//! of three neurons draws three lines at once. `n` inputs go in, `n_neurons`
//! outputs come out — and note that those two numbers need not match. A layer
//! is free to change the width of the data flowing through it.

use crate::neuron::Neuron;

#[derive(Debug, Clone)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(neurons: Vec<Neuron>) -> Self {
        assert!(!neurons.is_empty(), "a layer needs at least one neuron");

        // Every neuron in a layer reads the same input vector, so they must
        // all agree on how long that vector is. Catching this here gives a
        // clear error instead of a confusing panic deep inside `forward`.
        let width = neurons[0].weights.len();
        assert!(
            neurons.iter().all(|n| n.weights.len() == width),
            "all neurons in a layer must accept the same number of inputs"
        );

        Self { neurons }
    }

    /// How many inputs this layer expects.
    pub fn n_inputs(&self) -> usize {
        self.neurons[0].weights.len()
    }

    /// How many outputs this layer produces (one per neuron).
    pub fn n_outputs(&self) -> usize {
        self.neurons.len()
    }

    /// Run every neuron on the same inputs and collect their outputs.
    ///
    /// This is the whole layer. `map` over the neurons, ask each for its
    /// `forward`, gather the results into a `Vec`.
    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons.iter().map(|n| n.forward(inputs)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A layer holding the AND and OR gates from step 1.
    fn and_or_layer() -> Layer {
        Layer::new(vec![
            Neuron::new(vec![10.0, 10.0], -15.0), // AND
            Neuron::new(vec![10.0, 10.0], -5.0),  // OR
        ])
    }

    #[test]
    fn layer_maps_inputs_to_one_output_per_neuron() {
        let layer = and_or_layer();
        assert_eq!(layer.n_inputs(), 2);
        assert_eq!(layer.n_outputs(), 2);

        let out = layer.forward(&[1.0, 0.0]);
        assert_eq!(out.len(), 2);
        assert!(out[0] < 0.01, "AND should be quiet for (1, 0)");
        assert!(out[1] > 0.99, "OR should fire for (1, 0)");
    }

    #[test]
    fn a_layer_may_change_the_width_of_the_data() {
        // 2 inputs in, 3 outputs out.
        let layer = Layer::new(vec![
            Neuron::new(vec![1.0, 0.0], 0.0),
            Neuron::new(vec![0.0, 1.0], 0.0),
            Neuron::new(vec![1.0, 1.0], 0.0),
        ]);
        assert_eq!(layer.n_inputs(), 2);
        assert_eq!(layer.forward(&[1.0, 1.0]).len(), 3);
    }

    #[test]
    #[should_panic(expected = "same number of inputs")]
    fn neurons_must_agree_on_input_width() {
        Layer::new(vec![
            Neuron::new(vec![1.0, 2.0], 0.0),
            Neuron::new(vec![1.0], 0.0),
        ]);
    }
}
