//! Step 2b: a network — layers stacked front to back.
//!
//! Each layer's outputs become the next layer's inputs:
//!
//!     inputs ──> [layer 0] ──> [layer 1] ──> ... ──> outputs
//!                  hidden        output
//!
//! Layers in the middle are called "hidden" layers, for the unglamorous
//! reason that you never observe their values directly — you only see what
//! goes into the network and what comes out.
//!
//! This is where depth starts to pay. One neuron carves the input space with
//! a single straight line. A hidden layer carves it with several lines at
//! once, which chops the space into regions. The next layer then works in
//! terms of *those regions* rather than the raw inputs — so it can express
//! boundaries that no single line ever could.

use crate::layer::Layer;

#[derive(Debug, Clone)]
pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Self {
        assert!(!layers.is_empty(), "a network needs at least one layer");

        // Layer k's outputs feed layer k+1's inputs, so the widths have to
        // line up all the way down the stack.
        for (i, pair) in layers.windows(2).enumerate() {
            assert_eq!(
                pair[0].n_outputs(),
                pair[1].n_inputs(),
                "layer {i} emits {} values but layer {} expects {}",
                pair[0].n_outputs(),
                i + 1,
                pair[1].n_inputs()
            );
        }

        Self { layers }
    }

    /// Push inputs through every layer in turn and return the final outputs.
    ///
    /// `fold` carries the running vector through the stack: start with the
    /// inputs, and repeatedly replace them with the current layer's output.
    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.layers
            .iter()
            .fold(inputs.to_vec(), |values, layer| layer.forward(&values))
    }

    /// Like `forward`, but keeps every intermediate result.
    ///
    /// Returns `[inputs, layer0_out, layer1_out, ...]`, so the final element
    /// is what `forward` would have returned.
    ///
    /// Right now this only exists so we can *look* at the hidden layer and
    /// see what it computed. But backpropagation will need exactly this:
    /// to work out how to adjust a weight, you need to know what value was
    /// flowing through it during the forward pass. Same trick as splitting
    /// `net_input` out of `Neuron::forward` in step 1 — build the seam early.
    pub fn forward_all(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        let mut activations = vec![inputs.to_vec()];

        for layer in &self.layers {
            // `.last().unwrap()` is safe: we seeded the vec with the inputs,
            // so it is never empty.
            let next = layer.forward(activations.last().unwrap());
            activations.push(next);
        }

        activations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::Neuron;

    /// XOR, built by hand from three neurons.
    ///
    /// The identity we are exploiting is:
    ///     XOR = (x1 OR x2) AND NOT (x1 AND x2)
    ///          "at least one"      "but not both"
    ///
    /// Hidden layer computes OR and AND. The output neuron then keeps the
    /// OR (weight +10) while the AND vetoes it (weight -20, twice as strong,
    /// so it always wins when it fires).
    pub fn xor_network() -> Network {
        Network::new(vec![
            Layer::new(vec![
                Neuron::new(vec![10.0, 10.0], -5.0),  // OR
                Neuron::new(vec![10.0, 10.0], -15.0), // AND
            ]),
            Layer::new(vec![Neuron::new(vec![10.0, -20.0], -5.0)]),
        ])
    }

    #[test]
    fn hand_built_xor_works() {
        let net = xor_network();
        assert!(net.forward(&[0.0, 0.0])[0] < 0.01);
        assert!(net.forward(&[0.0, 1.0])[0] > 0.99);
        assert!(net.forward(&[1.0, 0.0])[0] > 0.99);
        assert!(net.forward(&[1.0, 1.0])[0] < 0.01);
    }

    #[test]
    fn forward_all_records_every_stage() {
        let net = xor_network();
        let acts = net.forward_all(&[1.0, 1.0]);

        assert_eq!(acts.len(), 3); // inputs, hidden, output
        assert_eq!(acts[0], vec![1.0, 1.0]);
        assert_eq!(acts[1].len(), 2); // the hidden layer's OR and AND
        assert_eq!(acts[2].len(), 1);

        // For (1, 1) both hidden neurons fire — and the AND is what kills
        // the output.
        assert!(acts[1][0] > 0.99, "OR fires");
        assert!(acts[1][1] > 0.99, "AND fires");

        // `forward_all`'s last entry must agree with `forward`.
        assert_eq!(acts[2], net.forward(&[1.0, 1.0]));
    }

    #[test]
    #[should_panic(expected = "layer 0 emits 2 values but layer 1 expects 3")]
    fn mismatched_layer_widths_are_rejected() {
        Network::new(vec![
            Layer::new(vec![
                Neuron::new(vec![1.0, 1.0], 0.0),
                Neuron::new(vec![1.0, 1.0], 0.0),
            ]),
            Layer::new(vec![Neuron::new(vec![1.0, 1.0, 1.0], 0.0)]),
        ]);
    }
}
