// We build this project up one concept at a time, so some modules are fully
// written and tested before `main` has any use for them. `neuron.rs` in
// particular is kept as a reference implementation, not as live code.
#![allow(dead_code)]

mod activation;
mod layer;
mod loss;
mod matrix;
mod mnist;
mod network;
mod neuron;
mod rng;

use std::process::ExitCode;
use std::time::Instant;

use matrix::Matrix;
use network::Network;
use rng::Rng;

const DATA_DIR: &str = "data";
const SHAPE: &[usize] = &[784, 100, 10];
const EPOCHS: usize = 30;
const BATCH_SIZE: usize = 32;

/// Cross-entropy produces much larger gradients than squared error — there is
/// no `y(1-y)` factor shrinking them — so the classifier needs a far gentler
/// learning rate than the sigmoid network's 3.0.
const CLASSIFIER_LEARNING_RATE: f64 = 0.3;
const SIGMOID_LEARNING_RATE: f64 = 3.0;

fn main() -> ExitCode {
    let (train, test) = match (mnist::load_training(DATA_DIR), mnist::load_test(DATA_DIR)) {
        (Ok(train), Ok(test)) => (train, test),
        _ => {
            eprintln!("Could not read MNIST from ./{DATA_DIR}");
            eprintln!("Run ./fetch-mnist.sh first.");
            return ExitCode::FAILURE;
        }
    };

    let (train_x, train_y) = train.to_matrices();
    let (test_x, test_y) = test.to_matrices();

    let probe: Vec<usize> = (0..10_000).collect();
    let probe_x = train_x.select_rows(&probe);
    let probe_y = train_y.select_rows(&probe);

    println!("MNIST: {} training digits, {} test digits", train.len(), test.len());
    println!("Network {SHAPE:?}, {EPOCHS} epochs, batch size {BATCH_SIZE}\n");

    // Train both, from the same seed, so the comparison is fair.
    let old = train_one(
        "sigmoid hidden + sigmoid output, squared error",
        Network::sigmoid_network(SHAPE, &mut Rng::new(42)),
        SIGMOID_LEARNING_RATE,
        (&train_x, &train_y),
        (&probe_x, &probe_y),
        (&test_x, &test_y),
    );

    let new = train_one(
        "ReLU hidden + softmax output, cross-entropy",
        Network::classifier(SHAPE, &mut Rng::new(42)),
        CLASSIFIER_LEARNING_RATE,
        (&train_x, &train_y),
        (&probe_x, &probe_y),
        (&test_x, &test_y),
    );

    println!("=======================================================");
    println!("  sigmoid + squared error :  {:.2}%", old.accuracy * 100.0);
    println!("  ReLU + softmax + x-ent  :  {:.2}%", new.accuracy * 100.0);
    println!(
        "  {} errors -> {} errors  ({:.0}% fewer)",
        ((1.0 - old.accuracy) * 10_000.0).round(),
        ((1.0 - new.accuracy) * 10_000.0).round(),
        (1.0 - (1.0 - new.accuracy) / (1.0 - old.accuracy)) * 100.0
    );
    println!("=======================================================\n");

    report_confusions(&new.net, &test_x, &test);
    show_confidence(&new.net, &test_x, &test);

    ExitCode::SUCCESS
}

struct Trained {
    net: Network,
    accuracy: f64,
}

fn train_one(
    label: &str,
    mut net: Network,
    learning_rate: f64,
    train: (&Matrix, &Matrix),
    probe: (&Matrix, &Matrix),
    test: (&Matrix, &Matrix),
) -> Trained {
    println!("--- {label} ---");
    println!("learning rate {learning_rate}\n");
    println!("  epoch |     loss  |   train acc   test acc  |   time");
    println!("  ------+-----------+-------------------------+--------");

    let mut rng = Rng::new(7);
    let started = Instant::now();

    for epoch in 1..=EPOCHS {
        let loss = net.train_epoch_minibatch(train.0, train.1, BATCH_SIZE, learning_rate, &mut rng);

        // Print the first few epochs and then every fifth, to keep it short
        // while still showing how fast the classifier starts.
        if epoch <= 3 || epoch % 5 == 0 {
            println!(
                "  {epoch:>5} |  {loss:.5}  |     {:>5.2}%     {:>5.2}%  |  {:>5.1}s",
                net.accuracy(probe.0, probe.1) * 100.0,
                net.accuracy(test.0, test.1) * 100.0,
                started.elapsed().as_secs_f64(),
            );
        }
    }

    let accuracy = net.accuracy(test.0, test.1);
    println!(
        "\n  final: {:.2}%  ({} of 10000 wrong)  in {:.1}s\n",
        accuracy * 100.0,
        ((1.0 - accuracy) * 10_000.0).round() as usize,
        started.elapsed().as_secs_f64(),
    );

    Trained { net, accuracy }
}

fn report_confusions(net: &Network, inputs: &Matrix, data: &mnist::MnistData) {
    let outputs = net.forward(inputs);

    let mut matrix = [[0usize; 10]; 10];
    for (row, &actual) in data.labels.iter().enumerate() {
        matrix[actual as usize][network::argmax(outputs.row(row))] += 1;
    }

    println!("Confusion matrix — rows are the true digit, columns the guess.\n");
    print!("        ");
    for guess in 0..10 {
        print!("{guess:>5}");
    }
    println!("   accuracy");
    for (actual, row) in matrix.iter().enumerate() {
        print!("  true {actual} ");
        for (guess, &count) in row.iter().enumerate() {
            if count == 0 && actual != guess {
                print!("    .");
            } else {
                print!("{count:>5}");
            }
        }
        let total: usize = row.iter().sum();
        println!("     {:>5.1}%", row[actual] as f64 / total as f64 * 100.0);
    }
    println!();
}

/// Softmax outputs are a real probability distribution, so "confidence" now
/// means something it did not before.
fn show_confidence(net: &Network, inputs: &Matrix, data: &mnist::MnistData) {
    let outputs = net.forward(inputs);

    let mut right = (0.0, 0usize);
    let mut wrong = (0.0, 0usize);

    for row in 0..outputs.rows {
        let scores = outputs.row(row);
        let guess = network::argmax(scores);
        let bucket = if guess == data.labels[row] as usize {
            &mut right
        } else {
            &mut wrong
        };
        bucket.0 += scores[guess];
        bucket.1 += 1;
    }

    println!("Softmax outputs sum to 1, so they read as probabilities:\n");
    println!(
        "  when correct ({:>5} digits):  average confidence {:.1}%",
        right.1,
        right.0 / right.1 as f64 * 100.0
    );
    println!(
        "  when wrong   ({:>5} digits):  average confidence {:.1}%",
        wrong.1,
        wrong.0 / wrong.1 as f64 * 100.0
    );
    println!("\nIt is measurably less sure when it is about to be wrong — which");
    println!("is something the old ten-independent-sigmoids output could not");
    println!("express, since its outputs never had to sum to anything.");

    // One mistake, with the full distribution.
    let mistake = (0..outputs.rows)
        .find(|&r| network::argmax(outputs.row(r)) != data.labels[r] as usize);

    if let Some(index) = mistake {
        let scores = outputs.row(index);
        println!("\nIts first mistake (test image #{index}):\n");
        println!("{}\n", mnist::render(&data.images[index]));
        println!("  true answer: {}", data.labels[index]);
        print!("  distribution:");
        for (digit, &p) in scores.iter().enumerate() {
            if p >= 0.01 {
                print!("  {digit}:{:.0}%", p * 100.0);
            }
        }
        println!();
    }
}
