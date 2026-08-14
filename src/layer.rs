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

use crate::neuron::{Neuron, sigmoid_derivative_from_output};

#[derive(Debug, Clone)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

/// The gradients for one layer: one gradient per weight, one per bias.
///
/// Same shape as the layer itself — `weights[j][i]` is the gradient for the
/// i-th weight of the j-th neuron.
#[derive(Debug, Clone)]
pub struct LayerGradients {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
}

impl LayerGradients {
    /// An all-zero set of gradients shaped like `layer`, ready to accumulate.
    pub fn zeros_like(layer: &Layer) -> Self {
        Self {
            weights: layer
                .neurons
                .iter()
                .map(|n| vec![0.0; n.weights.len()])
                .collect(),
            biases: vec![0.0; layer.neurons.len()],
        }
    }

    /// Add another sample's gradients into this accumulator.
    pub fn add(&mut self, other: &LayerGradients) {
        for (row, other_row) in self.weights.iter_mut().zip(&other.weights) {
            for (g, o) in row.iter_mut().zip(other_row) {
                *g += o;
            }
        }
        for (b, o) in self.biases.iter_mut().zip(&other.biases) {
            *b += o;
        }
    }

    /// Divide through by the batch size to get an average.
    pub fn scale(&mut self, factor: f64) {
        for row in &mut self.weights {
            for g in row {
                *g *= factor;
            }
        }
        for b in &mut self.biases {
            *b *= factor;
        }
    }
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

    /// The backward pass: gradients for this layer, plus the error signal to
    /// hand to the layer behind it.
    ///
    /// The signature is the important idea. Every layer speaks the same
    /// language in both directions:
    ///
    ///   forward:   inputs           ──>  outputs
    ///   backward:  dL/d(outputs)    ──>  dL/d(inputs)
    ///
    /// A layer is handed "how much the loss cares about each of my outputs",
    /// and returns "how much the loss cares about each of my inputs". Since
    /// its inputs *are* the previous layer's outputs, that return value is
    /// exactly what the previous layer needs. Chain the layers and the signal
    /// walks all the way back to the front. That is backpropagation.
    ///
    /// Three things happen per neuron:
    ///
    ///   1. delta = dL/dout * sigmoid'(out)     -- push through the activation
    ///   2. weight gradient = delta * input     -- blame each weight in
    ///      bias gradient   = delta                proportion to what it saw
    ///   3. send delta * weight backwards       -- blame each input in
    ///                                             proportion to its weight
    ///
    /// Step 3 is the answer to "what should the hidden neuron have output?"
    /// We never find out. Instead we ask: *if* this input had been slightly
    /// larger, would the loss have gone up or down? A hidden neuron connected
    /// through a large weight gets a large share of the blame. It is a
    /// responsibility calculation, and it needs no target.
    pub fn backward(
        &self,
        inputs: &[f64],
        outputs: &[f64],
        dl_doutputs: &[f64],
    ) -> (LayerGradients, Vec<f64>) {
        let mut gradients = LayerGradients::zeros_like(self);

        // What we will hand back to the previous layer. Every neuron in this
        // layer contributes to every one of these, so we sum into it.
        let mut dl_dinputs = vec![0.0; self.n_inputs()];

        for (j, neuron) in self.neurons.iter().enumerate() {
            // 1. Convert "gradient w.r.t. my output" into "gradient w.r.t. my
            //    pre-activation z" by passing through the sigmoid's slope.
            //    This is the same `delta` from step 3a.
            let delta = dl_doutputs[j] * sigmoid_derivative_from_output(outputs[j]);

            // 2. This layer's own gradients — identical to the single-neuron
            //    case, because once you have delta, nothing else differs.
            for (i, g) in gradients.weights[j].iter_mut().enumerate() {
                *g = delta * inputs[i];
            }
            gradients.biases[j] = delta;

            // 3. Propagate. Input i influenced this neuron through weight i,
            //    so it receives delta * weight_i of the blame — accumulated
            //    across every neuron this input fed into.
            for (i, w) in neuron.weights.iter().enumerate() {
                dl_dinputs[i] += delta * w;
            }
        }

        (gradients, dl_dinputs)
    }

    /// Step every weight and bias downhill.
    pub fn apply_gradients(&mut self, gradients: &LayerGradients, learning_rate: f64) {
        for (neuron, (weight_grads, bias_grad)) in self
            .neurons
            .iter_mut()
            .zip(gradients.weights.iter().zip(&gradients.biases))
        {
            neuron.apply_gradients(weight_grads, *bias_grad, learning_rate);
        }
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
