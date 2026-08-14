//! A layer, as matrices.
//!
//! The old `Layer` held `Vec<Neuron>` and looped. This one holds a single
//! weight matrix and processes an entire batch of samples in one product.
//!
//! Shapes, with `B` = batch size, `I` = inputs, `O` = neurons:
//!
//!     X  (B, I)   a batch of inputs, one sample per ROW
//!     W  (I, O)   column j is the j-th neuron's weights
//!     b  (O)      one bias per neuron, broadcast across the batch
//!
//!     Z = X * W + b     (B, O)   pre-activations
//!     A = sigmoid(Z)    (B, O)   outputs
//!
//! Note `W` is stored transposed relative to how you would picture a list of
//! neurons: a *column* of `W` is one neuron's weight vector. That is what
//! makes `X * W` work out, and it means the inner loop of `matmul` walks a
//! contiguous row rather than striding down a column.

use crate::activation::Activation;
use crate::matrix::Matrix;
use crate::rng::Rng;

#[derive(Debug, Clone)]
pub struct Layer {
    /// `(n_inputs, n_neurons)` — column j holds neuron j's weights.
    pub weights: Matrix,
    /// One per neuron.
    pub biases: Vec<f64>,
    /// Applied to this layer's pre-activations.
    pub activation: Activation,
}

/// Gradients for one layer, shaped exactly like the layer itself.
#[derive(Debug, Clone)]
pub struct LayerGradients {
    pub weights: Matrix,
    pub biases: Vec<f64>,
}

impl LayerGradients {
    pub fn zeros_like(layer: &Layer) -> Self {
        Self {
            weights: Matrix::zeros(layer.weights.rows, layer.weights.cols),
            biases: vec![0.0; layer.biases.len()],
        }
    }

    pub fn scale(&mut self, factor: f64) {
        self.weights.scale(factor);
        for b in &mut self.biases {
            *b *= factor;
        }
    }
}

impl Layer {
    pub fn new(weights: Matrix, biases: Vec<f64>, activation: Activation) -> Self {
        assert_eq!(
            weights.cols,
            biases.len(),
            "{} neurons but {} biases",
            weights.cols,
            biases.len()
        );
        Self {
            weights,
            biases,
            activation,
        }
    }

    /// Random weights, scaled to the layer's shape and activation.
    ///
    /// The range comes from `Activation::init_limit` — He for ReLU, Xavier
    /// otherwise — rather than a fixed [-1, 1). With 784 inputs a fixed range
    /// produces pre-activations far too large, which saturates sigmoid and
    /// makes ReLU explode.
    ///
    /// Weights must differ from each other or neurons would receive identical
    /// gradients forever. Biases have no such problem — the weights already
    /// break the symmetry — so they start at zero, which is standard.
    pub fn random(
        n_inputs: usize,
        n_neurons: usize,
        activation: Activation,
        rng: &mut Rng,
    ) -> Self {
        let limit = activation.init_limit(n_inputs, n_neurons);
        let data = (0..n_inputs * n_neurons)
            .map(|_| rng.uniform(-limit, limit))
            .collect();

        Self {
            weights: Matrix::from_vec(n_inputs, n_neurons, data),
            biases: vec![0.0; n_neurons],
            activation,
        }
    }

    pub fn n_inputs(&self) -> usize {
        self.weights.rows
    }

    pub fn n_outputs(&self) -> usize {
        self.weights.cols
    }

    /// `A = sigmoid(X * W + b)` for a whole batch.
    ///
    /// Compare with the old version, which was a `map` over neurons, each
    /// doing its own `zip`/`sum` over weights. Three nested loops became one
    /// matrix product plus two elementwise passes.
    pub fn forward(&self, inputs: &Matrix) -> Matrix {
        assert_eq!(
            inputs.cols,
            self.n_inputs(),
            "layer expects {} inputs, got {}",
            self.n_inputs(),
            inputs.cols
        );

        let mut out = inputs.matmul(&self.weights);
        out.add_row_broadcast(&self.biases);
        self.activation.apply(&mut out);
        out
    }

    /// The backward pass, for a whole batch.
    ///
    /// Identical in meaning to the loop-based version — only the notation
    /// changed. Given `dL/dA` (how the loss responds to this layer's outputs):
    ///
    ///     dZ = dA ⊙ sigmoid'(A)      (B, O)   through the activation
    ///     dW = X^T * dZ              (I, O)   blame each weight
    ///     db = column sums of dZ     (O)      the bias saw every sample
    ///     dX = dZ * W^T              (B, I)   blame each input
    ///
    /// The two transposes are the whole trick, and they are not arbitrary.
    /// Forward, `X * W` contracts over inputs. Going backward you need to
    /// contract over the *other* index each time, and transposing is how you
    /// say that. A useful sanity check when deriving these: only one
    /// arrangement of each product has shapes that line up at all.
    ///
    /// `dX` is what the previous layer receives as *its* `dA`. Same contract
    /// as before — `dL/d(outputs)` in, `dL/d(inputs)` out.
    pub fn backward(
        &self,
        inputs: &Matrix,
        outputs: &Matrix,
        d_outputs: &Matrix,
    ) -> (LayerGradients, Matrix) {
        // dZ = dA ⊙ activation'(A)
        let mut d_z = d_outputs.clone();
        let mut slopes = outputs.clone();
        slopes.map_in_place(|y| self.activation.derivative_from_output(y));
        d_z.multiply_elementwise(&slopes);

        self.backward_from_pre_activation(inputs, &d_z)
    }

    /// The second half of `backward`, entered directly when the caller already
    /// knows `dL/dz`.
    ///
    /// This split exists for softmax. Softmax's derivative is not elementwise
    /// — every output depends on every input — so the `dA ⊙ activation'(A)`
    /// step above simply does not apply to it. But paired with cross-entropy,
    /// `dL/dz` collapses to `y - t`, which the network can compute directly
    /// and hand straight to this method.
    ///
    /// Everything from here on is identical for every activation, because
    /// once you have `dZ`, the activation no longer matters.
    pub fn backward_from_pre_activation(
        &self,
        inputs: &Matrix,
        d_z: &Matrix,
    ) -> (LayerGradients, Matrix) {
        let d_weights = inputs.transpose().matmul(d_z);
        let d_biases = d_z.column_sums();
        let d_inputs = d_z.matmul(&self.weights.transpose());

        (
            LayerGradients {
                weights: d_weights,
                biases: d_biases,
            },
            d_inputs,
        )
    }

    /// Step every weight and bias downhill.
    pub fn apply_gradients(&mut self, gradients: &LayerGradients, learning_rate: f64) {
        for (w, g) in self.weights.data.iter_mut().zip(&gradients.weights.data) {
            *w -= learning_rate * g;
        }
        for (b, g) in self.biases.iter_mut().zip(&gradients.biases) {
            *b -= learning_rate * g;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::Neuron;

    /// The refactor must not change what the network computes.
    ///
    /// `neuron.rs` still holds the original one-neuron-at-a-time
    /// implementation. Here we build the same layer both ways and demand
    /// identical outputs. This is the test that makes the rewrite safe —
    /// the old code earns its keep as an executable specification.
    #[test]
    fn matrix_layer_agrees_with_the_neuron_implementation() {
        // Two neurons, three inputs each.
        let neurons = vec![
            Neuron::new(vec![0.5, -1.2, 0.3], 0.1),
            Neuron::new(vec![-0.7, 0.4, 2.0], -0.6),
        ];

        // Same thing as a matrix: column j is neuron j's weights, so the
        // matrix is the transpose of the list-of-neurons view.
        let layer = Layer::new(
            Matrix::from_rows(&[
                vec![0.5, -0.7], // input 0 -> both neurons
                vec![-1.2, 0.4], // input 1 -> both neurons
                vec![0.3, 2.0],  // input 2 -> both neurons
            ]),
            vec![0.1, -0.6],
            Activation::Sigmoid,
        );

        let batch = Matrix::from_rows(&[
            vec![1.0, 0.5, -0.3],
            vec![0.0, 2.0, 1.0],
            vec![-1.0, -1.0, 0.25],
        ]);

        let got = layer.forward(&batch);

        for r in 0..batch.rows {
            for (j, neuron) in neurons.iter().enumerate() {
                let expected = neuron.forward(batch.row(r));
                assert!(
                    (got.get(r, j) - expected).abs() < 1e-12,
                    "sample {r} neuron {j}: matrix {} vs neuron {expected}",
                    got.get(r, j)
                );
            }
        }
    }

    #[test]
    fn forward_produces_one_row_per_sample() {
        let mut rng = Rng::new(1);
        let layer = Layer::random(4, 3, Activation::Sigmoid, &mut rng);

        let batch = Matrix::zeros(7, 4);
        let out = layer.forward(&batch);

        assert_eq!(out.rows, 7, "one row per sample");
        assert_eq!(out.cols, 3, "one column per neuron");
        // sigmoid(0 + bias) for every row, so all rows are identical here.
        assert_eq!(out.row(0), out.row(6));
    }

    #[test]
    fn backward_shapes_line_up() {
        let mut rng = Rng::new(2);
        let layer = Layer::random(5, 3, Activation::Sigmoid, &mut rng);

        // `vec![..; 4]` rather than `[..; 4]`: array repetition needs Copy,
        // and Vec is only Clone.
        let inputs = Matrix::from_rows(&vec![vec![0.1, 0.2, 0.3, 0.4, 0.5]; 4]);
        let outputs = layer.forward(&inputs);
        let d_outputs = Matrix::from_rows(&vec![vec![1.0, -1.0, 0.5]; 4]);

        let (gradients, d_inputs) = layer.backward(&inputs, &outputs, &d_outputs);

        assert_eq!(gradients.weights.rows, 5, "one gradient row per input");
        assert_eq!(gradients.weights.cols, 3, "one gradient column per neuron");
        assert_eq!(gradients.biases.len(), 3);
        assert_eq!(d_inputs.rows, 4, "one row per sample");
        assert_eq!(d_inputs.cols, 5, "one column per input");
    }

    #[test]
    #[should_panic(expected = "layer expects 4 inputs, got 3")]
    fn wrong_input_width_is_rejected() {
        let mut rng = Rng::new(3);
        let layer = Layer::random(4, 2, Activation::Sigmoid, &mut rng);
        layer.forward(&Matrix::zeros(1, 3));
    }
}
