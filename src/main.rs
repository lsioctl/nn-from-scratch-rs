// We build this project up one concept at a time, so some modules are fully
// written and tested before `main` has any use for them. Without this, the
// compiler nags about `Layer` and `Network` sitting idle during step 3a.
#![allow(dead_code)]

mod layer;
mod loss;
mod network;
mod neuron;
mod rng;

use neuron::Neuron;
use rng::Rng;

/// A dataset is a list of (inputs, target) pairs.
type Samples = Vec<(Vec<f64>, f64)>;

fn gate_samples(targets: [f64; 4]) -> Samples {
    vec![
        (vec![0.0, 0.0], targets[0]),
        (vec![0.0, 1.0], targets[1]),
        (vec![1.0, 0.0], targets[2]),
        (vec![1.0, 1.0], targets[3]),
    ]
}

/// Train one neuron on one dataset, printing the loss as it goes.
fn train_and_report(name: &str, samples: &Samples, epochs: usize, learning_rate: f64) {
    let mut rng = Rng::new(42);
    let mut neuron = Neuron::random(2, &mut rng);

    println!("=== learning {name} ===");
    println!(
        "  start:  w = [{:+.3}, {:+.3}]  b = {:+.3}   (random)",
        neuron.weights[0], neuron.weights[1], neuron.bias
    );
    println!("\n   epoch  |     loss");
    println!("  --------+-----------");

    for epoch in 0..=epochs {
        let loss = neuron.train_epoch(samples, learning_rate);
        if epoch % (epochs / 8) == 0 {
            println!("  {epoch:>7}  |  {loss:.5}");
        }
    }

    println!(
        "\n  learned:  w = [{:+.3}, {:+.3}]  b = {:+.3}",
        neuron.weights[0], neuron.weights[1], neuron.bias
    );
    println!("\n   x1   x2  |  predicted   target");
    println!("  ----------+--------------------");
    for (inputs, target) in samples {
        println!(
            "  {:>4.0} {:>4.0}  |     {:.4}      {target:.0}",
            inputs[0],
            inputs[1],
            neuron.forward(inputs)
        );
    }
    println!();
}

fn main() {
    // Nobody tells the neuron what AND means. It only ever sees four
    // (input, target) pairs and a rule for going downhill.
    train_and_report("AND", &gate_samples([0.0, 0.0, 0.0, 1.0]), 20_000, 5.0);

    // Same code, same effort, one line of data changed — and it collapses.
    train_and_report("XOR", &gate_samples([0.0, 1.0, 1.0, 0.0]), 20_000, 5.0);

    println!("Notice how XOR fails: no error, no divergence. The loss just");
    println!("parks at 0.25 and the neuron answers 0.5 to everything — the");
    println!("arithmetic of a machine that has given up and is hedging.");
    println!("0.25 is exactly the loss of guessing 0.5 on all four examples.");
}
