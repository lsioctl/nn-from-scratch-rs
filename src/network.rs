//! A network of matrix layers, trained on whole batches at once.
//!
//! The algorithm is unchanged from the loop-based version — forward, seed the
//! error at the output, walk backwards handing each layer the gradient of the
//! loss with respect to its outputs. Only now every step processes `B` samples
//! simultaneously, and the per-sample loop that used to wrap the whole thing
//! is gone.

use crate::activation::Activation;
use crate::layer::{Layer, LayerGradients};
use crate::loss::{cross_entropy, softmax_cross_entropy_gradient, squared_error_derivative};
use crate::matrix::Matrix;
use crate::rng::Rng;

#[derive(Debug, Clone)]
pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Self {
        assert!(!layers.is_empty(), "a network needs at least one layer");

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

    /// Build a network of the given shape with random weights.
    ///
    /// `&[784, 30, 10]` means 784 inputs, a hidden layer of 30, 10 outputs.
    /// Every hidden layer gets `hidden`; the final layer gets `output`.
    pub fn random(
        sizes: &[usize],
        hidden: Activation,
        output: Activation,
        rng: &mut Rng,
    ) -> Self {
        assert!(sizes.len() >= 2, "need at least an input and an output size");

        let last = sizes.len() - 2;
        Self::new(
            sizes
                .windows(2)
                .enumerate()
                .map(|(i, pair)| {
                    let activation = if i == last { output } else { hidden };
                    Layer::random(pair[0], pair[1], activation, rng)
                })
                .collect(),
        )
    }

    /// A classifier: ReLU hidden layers, softmax output, cross-entropy loss.
    pub fn classifier(sizes: &[usize], rng: &mut Rng) -> Self {
        Self::random(sizes, Activation::Relu, Activation::Softmax, rng)
    }

    /// The old all-sigmoid network, trained with squared error.
    pub fn sigmoid_network(sizes: &[usize], rng: &mut Rng) -> Self {
        Self::random(sizes, Activation::Sigmoid, Activation::Sigmoid, rng)
    }

    /// True when this network ends in softmax, and therefore uses
    /// cross-entropy rather than squared error.
    pub fn is_classifier(&self) -> bool {
        self.layers.last().unwrap().activation == Activation::Softmax
    }

    /// Push a batch through every layer. `(B, inputs) -> (B, outputs)`.
    pub fn forward(&self, inputs: &Matrix) -> Matrix {
        self.layers
            .iter()
            .fold(inputs.clone(), |values, layer| layer.forward(&values))
    }

    /// `forward`, keeping every intermediate activation for the backward pass.
    ///
    /// Returns `[inputs, layer0_out, layer1_out, ...]`.
    pub fn forward_all(&self, inputs: &Matrix) -> Vec<Matrix> {
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        activations.push(inputs.clone());

        for layer in &self.layers {
            let next = layer.forward(activations.last().unwrap());
            activations.push(next);
        }

        activations
    }

    /// Convenience for a single sample.
    pub fn forward_one(&self, inputs: &[f64]) -> Vec<f64> {
        self.forward(&Matrix::row_vector(inputs)).data
    }

    /// Backpropagation over a batch.
    ///
    /// Returns per-layer gradients (summed over the batch, not yet averaged)
    /// and the summed loss.
    pub fn gradients(&self, inputs: &Matrix, targets: &Matrix) -> (Vec<LayerGradients>, f64) {
        assert_eq!(
            inputs.rows, targets.rows,
            "{} input rows but {} target rows",
            inputs.rows, targets.rows
        );

        // 1. Forward, remembering activations.
        let activations = self.forward_all(inputs);
        let outputs = activations.last().unwrap();

        assert_eq!(
            outputs.cols, targets.cols,
            "network emits {} outputs but got {} targets",
            outputs.cols, targets.cols
        );

        let mut gradients = Vec::with_capacity(self.layers.len());
        let last = self.layers.len() - 1;

        // 2. Seed the error at the output, and 3. take the first step back.
        //
        //    The two paths differ in *what* they hand the layer:
        //
        //    - softmax + cross-entropy gives dL/dz directly (`y - t`), because
        //      softmax has no elementwise derivative to apply. It goes
        //      straight into the second half of the backward pass.
        //    - anything else gives dL/da (`2(y - t)` for squared error) and
        //      lets the layer apply its own activation derivative.
        let (loss, mut d_values) = if self.is_classifier() {
            let mut d_z = Matrix::zeros(outputs.rows, outputs.cols);
            for (i, (&o, &t)) in outputs.data.iter().zip(&targets.data).enumerate() {
                d_z.data[i] = softmax_cross_entropy_gradient(o, t);
            }

            let loss: f64 = (0..outputs.rows)
                .map(|r| cross_entropy(outputs.row(r), targets.row(r)))
                .sum();

            let (layer_gradients, d_inputs) = self.layers[last]
                .backward_from_pre_activation(&activations[last], &d_z);
            gradients.push(layer_gradients);

            (loss, d_inputs)
        } else {
            let mut loss = 0.0;
            let mut d_a = Matrix::zeros(outputs.rows, outputs.cols);
            for (i, (&o, &t)) in outputs.data.iter().zip(&targets.data).enumerate() {
                loss += (o - t) * (o - t);
                d_a.data[i] = squared_error_derivative(o, t);
            }

            let (layer_gradients, d_inputs) = self.layers[last].backward(
                &activations[last],
                &activations[last + 1],
                &d_a,
            );
            gradients.push(layer_gradients);

            (loss, d_inputs)
        };

        // 4. The remaining layers are all ordinary — walk them backwards.
        for k in (0..last).rev() {
            let (layer_gradients, d_inputs) =
                self.layers[k].backward(&activations[k], &activations[k + 1], &d_values);

            gradients.push(layer_gradients);
            d_values = d_inputs;
        }

        gradients.reverse();
        (gradients, loss)
    }

    /// One gradient step on one batch. Returns the summed loss.
    pub fn train_batch(&mut self, inputs: &Matrix, targets: &Matrix, learning_rate: f64) -> f64 {
        let (mut gradients, loss) = self.gradients(inputs, targets);

        let scale = 1.0 / inputs.rows as f64;
        for (layer, layer_gradients) in self.layers.iter_mut().zip(&mut gradients) {
            layer_gradients.scale(scale);
            layer.apply_gradients(layer_gradients, learning_rate);
        }

        loss
    }

    /// One pass over the dataset in shuffled minibatches.
    ///
    /// Returns the mean per-sample loss over the epoch.
    pub fn train_epoch_minibatch(
        &mut self,
        inputs: &Matrix,
        targets: &Matrix,
        batch_size: usize,
        learning_rate: f64,
        rng: &mut Rng,
    ) -> f64 {
        assert!(batch_size > 0, "batch size must be positive");
        assert_eq!(inputs.rows, targets.rows);

        // Shuffle indices rather than the data — cheaper, and it leaves the
        // caller's matrices untouched.
        let mut order: Vec<usize> = (0..inputs.rows).collect();
        rng.shuffle(&mut order);

        let mut total_loss = 0.0;
        for batch in order.chunks(batch_size) {
            let batch_inputs = inputs.select_rows(batch);
            let batch_targets = targets.select_rows(batch);
            total_loss += self.train_batch(&batch_inputs, &batch_targets, learning_rate);
        }

        total_loss / inputs.rows as f64
    }

    /// One pass over the whole dataset as a single batch. Fine for XOR.
    pub fn train_epoch(&mut self, inputs: &Matrix, targets: &Matrix, learning_rate: f64) -> f64 {
        self.train_batch(inputs, targets, learning_rate) / inputs.rows as f64
    }

    /// The index of the largest output — the network's answer for one sample.
    pub fn predict(&self, inputs: &[f64]) -> usize {
        argmax(&self.forward_one(inputs))
    }

    /// Fraction of samples classified correctly, in [0, 1].
    ///
    /// Evaluated as one big batch, which is far faster than sample-by-sample.
    pub fn accuracy(&self, inputs: &Matrix, targets: &Matrix) -> f64 {
        let outputs = self.forward(inputs);

        let correct = (0..outputs.rows)
            .filter(|&r| argmax(outputs.row(r)) == argmax(targets.row(r)))
            .count();

        correct as f64 / outputs.rows as f64
    }

    /// Total number of trainable parameters.
    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.data.len() + l.biases.len())
            .sum()
    }
}

/// Index of the largest value. Ties go to the earliest.
pub fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .fold(
            (0, f64::NEG_INFINITY),
            |(best_i, best), (i, &v)| if v > best { (i, v) } else { (best_i, best) },
        )
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xor_data() -> (Matrix, Matrix) {
        let inputs = Matrix::from_rows(&[
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ]);
        let targets = Matrix::from_rows(&[vec![0.0], vec![1.0], vec![1.0], vec![0.0]]);
        (inputs, targets)
    }

    /// Gradient check for the sigmoid + squared-error path.
    #[test]
    fn backprop_matches_numerical_gradients() {
        let mut rng = Rng::new(7);
        let net = Network::sigmoid_network(&[3, 4, 2], &mut rng);

        let inputs = Matrix::from_rows(&[
            vec![0.6, -0.2, 0.9],
            vec![0.1, 0.5, -0.4],
            vec![-0.8, 0.3, 0.2],
        ]);
        let targets = Matrix::from_rows(&[
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ]);

        check_gradients_numerically(&net, &inputs, &targets);
    }

    /// Gradient check for the ReLU + softmax + cross-entropy path.
    ///
    /// This is the one that matters most in this step. The `y - t` shortcut is
    /// a *claim* that softmax's Jacobian and cross-entropy's `-t/y` cancel;
    /// this verifies the claim end to end, through a ReLU hidden layer, by
    /// nudging every weight and watching the actual cross-entropy loss move.
    #[test]
    fn softmax_cross_entropy_backprop_matches_numerical_gradients() {
        let mut rng = Rng::new(13);
        let net = Network::classifier(&[3, 5, 4], &mut rng);
        assert!(net.is_classifier());

        let inputs = Matrix::from_rows(&[
            vec![0.6, -0.2, 0.9],
            vec![0.1, 0.5, -0.4],
            vec![-0.8, 0.3, 0.2],
        ]);
        // One-hot targets, as a real classifier would have.
        let targets = Matrix::from_rows(&[
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ]);

        check_gradients_numerically(&net, &inputs, &targets);
    }

    /// Nudge every weight and bias by ±h and compare the measured change in
    /// loss against the analytic gradient. Uses whichever loss the network's
    /// output activation implies.
    fn check_gradients_numerically(net: &Network, inputs: &Matrix, targets: &Matrix) {
        let (gradients, _) = net.gradients(inputs, targets);
        let h = 1e-6;

        let is_classifier = net.is_classifier();
        let loss_of = |n: &Network| -> f64 {
            let out = n.forward(inputs);
            if is_classifier {
                (0..out.rows)
                    .map(|r| cross_entropy(out.row(r), targets.row(r)))
                    .sum()
            } else {
                out.data
                    .iter()
                    .zip(&targets.data)
                    .map(|(&o, &t)| (o - t) * (o - t))
                    .sum()
            }
        };

        for k in 0..net.layers.len() {
            for i in 0..net.layers[k].weights.data.len() {
                let mut up = net.clone();
                up.layers[k].weights.data[i] += h;
                let mut down = net.clone();
                down.layers[k].weights.data[i] -= h;

                let numerical = (loss_of(&up) - loss_of(&down)) / (2.0 * h);
                let analytic = gradients[k].weights.data[i];

                assert!(
                    (analytic - numerical).abs() < 1e-6,
                    "layer {k} weight {i}: analytic {analytic} vs numerical {numerical}"
                );
            }

            for j in 0..net.layers[k].biases.len() {
                let mut up = net.clone();
                up.layers[k].biases[j] += h;
                let mut down = net.clone();
                down.layers[k].biases[j] -= h;

                let numerical = (loss_of(&up) - loss_of(&down)) / (2.0 * h);
                let analytic = gradients[k].biases[j];

                assert!(
                    (analytic - numerical).abs() < 1e-6,
                    "layer {k} bias {j}: analytic {analytic} vs numerical {numerical}"
                );
            }
        }
    }

    #[test]
    fn backprop_learns_xor_from_random_weights() {
        let (inputs, targets) = xor_data();
        let mut rng = Rng::new(42);
        let mut net = Network::sigmoid_network(&[2, 4, 1], &mut rng);

        let mut loss = 0.0;
        for _ in 0..20_000 {
            loss = net.train_epoch(&inputs, &targets, 5.0);
        }

        assert!(loss < 0.01, "should have learned XOR, loss was {loss}");
        for r in 0..inputs.rows {
            let got = net.forward_one(inputs.row(r))[0];
            let want = targets.get(r, 0);
            assert!((got - want).abs() < 0.1, "row {r}: {got} vs {want}");
        }
    }

    #[test]
    fn minibatch_training_also_learns_xor() {
        let (inputs, targets) = xor_data();
        let mut rng = Rng::new(42);
        let mut net = Network::sigmoid_network(&[2, 4, 1], &mut rng);

        let mut loss = 0.0;
        for _ in 0..5_000 {
            loss = net.train_epoch_minibatch(&inputs, &targets, 2, 5.0, &mut rng);
        }
        assert!(loss < 0.01, "minibatch should learn XOR too, loss {loss}");
    }

    /// Batching must not change results: a batch of N gives the same outputs
    /// as N separate single-sample passes.
    #[test]
    fn batched_forward_equals_one_at_a_time() {
        let mut rng = Rng::new(11);
        let net = Network::sigmoid_network(&[3, 5, 2], &mut rng);

        let batch = Matrix::from_rows(&[
            vec![0.1, 0.2, 0.3],
            vec![-0.5, 0.9, 0.0],
            vec![1.0, -1.0, 0.5],
        ]);

        let batched = net.forward(&batch);
        for r in 0..batch.rows {
            let single = net.forward_one(batch.row(r));
            for c in 0..batched.cols {
                assert!((batched.get(r, c) - single[c]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn argmax_picks_the_largest() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 1.0]), 0);
        assert_eq!(argmax(&[0.0, 0.0, 7.0]), 2);
        assert_eq!(argmax(&[2.0, 2.0]), 0, "ties go to the earliest");
    }

    #[test]
    fn random_builds_the_requested_shape() {
        let mut rng = Rng::new(1);
        let net = Network::sigmoid_network(&[3, 5, 2], &mut rng);

        assert_eq!(net.layers.len(), 2);
        assert_eq!(net.layers[0].n_inputs(), 3);
        assert_eq!(net.layers[0].n_outputs(), 5);
        assert_eq!(net.layers[1].n_inputs(), 5);
        assert_eq!(net.layers[1].n_outputs(), 2);
        // 3*5 + 5 + 5*2 + 2
        assert_eq!(net.parameter_count(), 32);
    }

    #[test]
    fn accuracy_is_a_fraction() {
        let mut rng = Rng::new(3);
        let net = Network::sigmoid_network(&[2, 3, 2], &mut rng);

        let inputs = Matrix::from_rows(&[vec![0.0, 0.0], vec![1.0, 1.0]]);
        let targets = Matrix::from_rows(&[vec![1.0, 0.0], vec![0.0, 1.0]]);

        let accuracy = net.accuracy(&inputs, &targets);
        assert!([0.0, 0.5, 1.0].contains(&accuracy));
    }

    #[test]
    #[should_panic(expected = "layer 0 emits 2 values but layer 1 expects 3")]
    fn mismatched_layer_widths_are_rejected() {
        Network::new(vec![
            Layer::new(Matrix::zeros(2, 2), vec![0.0, 0.0], Activation::Sigmoid),
            Layer::new(Matrix::zeros(3, 1), vec![0.0], Activation::Sigmoid),
        ]);
    }
}
