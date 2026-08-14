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

use crate::layer::{Layer, LayerGradients};
use crate::loss::{squared_error, squared_error_derivative};
use crate::neuron::Neuron;
use crate::rng::Rng;

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

    /// Build a network of the given shape with random weights.
    ///
    /// `sizes` lists the width of every stage including the input, so
    /// `&[2, 3, 1]` means 2 inputs, a hidden layer of 3, and 1 output.
    pub fn random(sizes: &[usize], rng: &mut Rng) -> Self {
        assert!(sizes.len() >= 2, "need at least an input and an output size");

        let layers = sizes
            .windows(2)
            .map(|pair| {
                let (n_inputs, n_neurons) = (pair[0], pair[1]);
                Layer::new(
                    (0..n_neurons)
                        .map(|_| Neuron::random(n_inputs, rng))
                        .collect(),
                )
            })
            .collect();

        Self::new(layers)
    }

    /// Backpropagation: gradients for every layer, from one training example.
    ///
    /// The whole algorithm is a forward pass, one subtraction, and a loop
    /// running backwards. Read it in that order:
    ///
    ///   1. Go forward, remembering every activation along the way.
    ///   2. Seed the error at the output: dL/dy = 2(y - t). This is the only
    ///      place a target is ever used.
    ///   3. Walk the layers in reverse. Each one turns "gradient w.r.t. my
    ///      outputs" into its own gradients plus "gradient w.r.t. my inputs",
    ///      which is the seed for the layer before it.
    ///
    /// Returns the per-layer gradients and the loss on this example.
    pub fn gradients(&self, inputs: &[f64], targets: &[f64]) -> (Vec<LayerGradients>, f64) {
        // 1. Forward, keeping activations — this is why `forward_all` exists.
        let activations = self.forward_all(inputs);
        let outputs = activations.last().unwrap();

        assert_eq!(
            outputs.len(),
            targets.len(),
            "network emits {} outputs but got {} targets",
            outputs.len(),
            targets.len()
        );

        // A sample's loss is the SUM over its outputs, not the mean. If we
        // averaged here, every gradient would pick up a 1/n_outputs factor
        // that the seed below does not have, and the two would silently
        // disagree by exactly that factor. (This is not hypothetical — the
        // gradient check caught precisely that mistake.) Summing keeps
        // dL/dy_j = 2(y_j - t_j) exactly, with no bookkeeping.
        let loss: f64 = outputs
            .iter()
            .zip(targets)
            .map(|(&o, &t)| squared_error(o, t))
            .sum();

        // 2. Seed: how the loss responds to each output.
        let mut dl_doutputs: Vec<f64> = outputs
            .iter()
            .zip(targets)
            .map(|(&o, &t)| squared_error_derivative(o, t))
            .collect();

        // 3. Walk backwards. Layer k consumed activations[k] and produced
        //    activations[k + 1].
        let mut gradients = Vec::with_capacity(self.layers.len());
        for (k, layer) in self.layers.iter().enumerate().rev() {
            let (layer_gradients, dl_dinputs) =
                layer.backward(&activations[k], &activations[k + 1], &dl_doutputs);

            gradients.push(layer_gradients);

            // This layer's inputs are the previous layer's outputs, so its
            // "dL/d(inputs)" is precisely the next iteration's seed.
            dl_doutputs = dl_dinputs;
        }

        // We collected back-to-front; flip so index k means layer k.
        gradients.reverse();

        (gradients, loss)
    }

    /// Accumulate gradients over the selected samples, average, take one step.
    ///
    /// Returns the *summed* loss, so callers can average however they like.
    fn train_batch(
        &mut self,
        samples: &[(Vec<f64>, Vec<f64>)],
        indices: &[usize],
        learning_rate: f64,
    ) -> f64 {
        let mut totals: Vec<LayerGradients> = self
            .layers
            .iter()
            .map(LayerGradients::zeros_like)
            .collect();
        let mut total_loss = 0.0;

        for &i in indices {
            let (inputs, targets) = &samples[i];
            let (gradients, loss) = self.gradients(inputs, targets);
            total_loss += loss;
            for (acc, g) in totals.iter_mut().zip(&gradients) {
                acc.add(g);
            }
        }

        let scale = 1.0 / indices.len() as f64;
        for (layer, gradients) in self.layers.iter_mut().zip(&mut totals) {
            gradients.scale(scale);
            layer.apply_gradients(gradients, learning_rate);
        }

        total_loss
    }

    /// One pass over the whole dataset as a single batch, one weight update.
    ///
    /// Fine for four XOR examples. Hopeless for 60,000 digits — see
    /// `train_epoch_minibatch`.
    pub fn train_epoch(&mut self, samples: &[(Vec<f64>, Vec<f64>)], learning_rate: f64) -> f64 {
        let indices: Vec<usize> = (0..samples.len()).collect();
        self.train_batch(samples, &indices, learning_rate) / samples.len() as f64
    }

    /// One pass over the dataset in shuffled minibatches — the workhorse.
    ///
    /// Full-batch descent computes a beautifully accurate gradient and then
    /// takes *one* step with it. Across 60,000 examples that is a colossal
    /// amount of arithmetic per update, and training would take days.
    ///
    /// Minibatching computes a rougher gradient from (say) 32 examples and
    /// steps immediately, giving ~1,875 updates per pass instead of one. The
    /// estimate is noisy, but noise is affordable and, in practice, mildly
    /// helpful — it can knock the network out of shallow local minima.
    ///
    /// The shuffle matters: MNIST is not stored in random order, and batches
    /// that were all-the-same-digit would drag the weights around in circles.
    ///
    /// Returns the mean per-example loss over the epoch.
    pub fn train_epoch_minibatch(
        &mut self,
        samples: &[(Vec<f64>, Vec<f64>)],
        batch_size: usize,
        learning_rate: f64,
        rng: &mut Rng,
    ) -> f64 {
        assert!(batch_size > 0, "batch size must be positive");

        let mut order: Vec<usize> = (0..samples.len()).collect();
        rng.shuffle(&mut order);

        let mut total_loss = 0.0;
        for batch in order.chunks(batch_size) {
            total_loss += self.train_batch(samples, batch, learning_rate);
        }

        total_loss / samples.len() as f64
    }

    /// The index of the largest output — the network's actual answer.
    ///
    /// With ten output neurons, "which digit is this?" means "which neuron is
    /// most excited?". The raw values are not probabilities (they do not sum
    /// to 1 — softmax would fix that), but for picking a winner it makes no
    /// difference.
    pub fn predict(&self, inputs: &[f64]) -> usize {
        argmax(&self.forward(inputs))
    }

    /// Fraction of samples classified correctly, in [0, 1].
    ///
    /// This — not the loss — is what you actually care about. Loss is what we
    /// can differentiate; accuracy is what the task is.
    pub fn accuracy(&self, samples: &[(Vec<f64>, Vec<f64>)]) -> f64 {
        let correct = samples
            .iter()
            .filter(|(inputs, targets)| self.predict(inputs) == argmax(targets))
            .count();

        correct as f64 / samples.len() as f64
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

    /// Gradient check, now for the whole network.
    ///
    /// Backprop is easy to get subtly wrong — an index off by one, a
    /// transposed weight, a missing activation derivative — and none of those
    /// crash. They just make the network learn slowly, or not at all. So we
    /// verify every single gradient against the definition of a derivative:
    /// nudge one weight, see how much the loss really moved.
    #[test]
    fn backprop_matches_numerical_gradients() {
        let mut rng = Rng::new(7);
        // Deliberately lopsided (3 -> 4 -> 2) so any index mix-up between
        // layer widths shows up as a length mismatch or a wrong number.
        let net = Network::random(&[3, 4, 2], &mut rng);

        let inputs = [0.6, -0.2, 0.9];
        let targets = [1.0, 0.0];

        let (gradients, _) = net.gradients(&inputs, &targets);
        let h = 1e-6;

        let loss_of = |n: &Network| -> f64 {
            let out = n.forward(&inputs);
            out.iter()
                .zip(&targets)
                .map(|(&o, &t)| squared_error(o, t))
                .sum()
        };

        for (k, layer) in net.layers.iter().enumerate() {
            for (j, neuron) in layer.neurons.iter().enumerate() {
                for i in 0..neuron.weights.len() {
                    let mut up = net.clone();
                    up.layers[k].neurons[j].weights[i] += h;
                    let mut down = net.clone();
                    down.layers[k].neurons[j].weights[i] -= h;

                    let numerical = (loss_of(&up) - loss_of(&down)) / (2.0 * h);
                    let analytic = gradients[k].weights[j][i];

                    assert!(
                        (analytic - numerical).abs() < 1e-7,
                        "layer {k} neuron {j} weight {i}: analytic {analytic} vs numerical {numerical}"
                    );
                }

                let mut up = net.clone();
                up.layers[k].neurons[j].bias += h;
                let mut down = net.clone();
                down.layers[k].neurons[j].bias -= h;

                let numerical = (loss_of(&up) - loss_of(&down)) / (2.0 * h);
                let analytic = gradients[k].biases[j];
                assert!(
                    (analytic - numerical).abs() < 1e-7,
                    "layer {k} neuron {j} bias: analytic {analytic} vs numerical {numerical}"
                );
            }
        }
    }

    /// The payoff: the thing a single neuron provably could not do.
    #[test]
    fn backprop_learns_xor_from_random_weights() {
        let samples = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];

        let mut rng = Rng::new(42);
        let mut net = Network::random(&[2, 4, 1], &mut rng);

        let mut loss = 0.0;
        for _ in 0..20_000 {
            loss = net.train_epoch(&samples, 5.0);
        }

        assert!(loss < 0.01, "should have learned XOR, loss was {loss}");
        for (inputs, targets) in &samples {
            let got = net.forward(inputs)[0];
            assert!(
                (got - targets[0]).abs() < 0.1,
                "{inputs:?} -> {got}, want {}",
                targets[0]
            );
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
    fn accuracy_counts_correct_argmaxes() {
        let net = Network::random(&[2, 3, 2], &mut Rng::new(3));
        let samples = vec![
            (vec![0.0, 0.0], vec![1.0, 0.0]),
            (vec![1.0, 1.0], vec![0.0, 1.0]),
        ];
        let acc = net.accuracy(&samples);
        assert!((0.0..=1.0).contains(&acc));
        // An untrained 2-output net gets 0, 0.5 or 1 on two samples.
        assert!([0.0, 0.5, 1.0].contains(&acc));
    }

    #[test]
    fn minibatch_training_also_learns_xor() {
        let samples = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];

        let mut rng = Rng::new(42);
        let mut net = Network::random(&[2, 4, 1], &mut rng);

        let mut loss = 0.0;
        for _ in 0..5_000 {
            loss = net.train_epoch_minibatch(&samples, 2, 5.0, &mut rng);
        }
        assert!(loss < 0.01, "minibatch should learn XOR too, loss {loss}");
    }

    #[test]
    fn random_builds_the_requested_shape() {
        let mut rng = Rng::new(1);
        let net = Network::random(&[3, 5, 2], &mut rng);
        assert_eq!(net.layers.len(), 2);
        assert_eq!(net.layers[0].n_inputs(), 3);
        assert_eq!(net.layers[0].n_outputs(), 5);
        assert_eq!(net.layers[1].n_inputs(), 5);
        assert_eq!(net.layers[1].n_outputs(), 2);
        assert_eq!(net.forward(&[0.1, 0.2, 0.3]).len(), 2);
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
