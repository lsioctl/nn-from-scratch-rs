// We build this project up one concept at a time, so some modules are fully
// written and tested before `main` has any use for them.
#![allow(dead_code)]

mod layer;
mod loss;
mod network;
mod neuron;
mod rng;

use network::Network;
use rng::Rng;

type Samples = Vec<(Vec<f64>, Vec<f64>)>;

fn xor_samples() -> Samples {
    vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ]
}

fn main() {
    let samples = xor_samples();

    // 2 inputs -> 4 hidden -> 1 output. Every weight starts random; nothing
    // in here knows what XOR is.
    let mut rng = Rng::new(42);
    let mut net = Network::random(&[2, 4, 1], &mut rng);

    println!("Learning XOR by backpropagation  (2 -> 4 -> 1, random init)\n");
    println!("   epoch  |     loss   |  predictions for 00  01  10  11");
    println!("  --------+------------+---------------------------------");

    let epochs = 20_000;
    for epoch in 0..=epochs {
        let loss = net.train_epoch(&samples, 5.0);

        if epoch % 2_500 == 0 {
            let p: Vec<String> = samples
                .iter()
                .map(|(inputs, _)| format!("{:.2}", net.forward(inputs)[0]))
                .collect();
            println!("  {epoch:>7}  |  {loss:.6}  |     {}", p.join("  "));
        }
    }

    println!("\n  targets were:                    0.00  1.00  1.00  0.00\n");

    // What did the hidden layer invent for itself?
    println!("What the 4 hidden neurons learned to compute:\n");
    println!("   x1   x2  |    h0     h1     h2     h3  |  output   want");
    println!("  ----------+------------------------------+---------------");
    for (inputs, targets) in &samples {
        let acts = net.forward_all(inputs);
        let h = &acts[1];
        println!(
            "  {:>4.0} {:>4.0}  | {:>5.2}  {:>5.2}  {:>5.2}  {:>5.2}  |  {:.4}   {:.0}",
            inputs[0], inputs[1], h[0], h[1], h[2], h[3], acts[2][0], targets[0]
        );
    }

    println!("\nCompare with step 2, where I hand-picked an OR and an AND from");
    println!("boolean algebra. Backprop found its own internal representation —");
    println!("probably not OR/AND, and it does not need to be. Any hidden layer");
    println!("that makes the four cases linearly separable will do.");
}
