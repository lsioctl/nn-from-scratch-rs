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

/// 784 pixels in, one hidden layer of 30, 10 digit outputs.
const SHAPE: &[usize] = &[784, 30, 10];

const EPOCHS: usize = 30;
const BATCH_SIZE: usize = 32;
const LEARNING_RATE: f64 = 3.0;

fn main() -> ExitCode {
    let (train, test) = match (mnist::load_training(DATA_DIR), mnist::load_test(DATA_DIR)) {
        (Ok(train), Ok(test)) => (train, test),
        _ => {
            eprintln!("Could not read MNIST from ./{DATA_DIR}");
            eprintln!("Run ./fetch-mnist.sh first.");
            return ExitCode::FAILURE;
        }
    };

    println!("MNIST: {} training digits, {} test digits\n", train.len(), test.len());
    println!("First training digit (label: {}):\n", train.labels[0]);
    println!("{}\n", mnist::render(&train.images[0]));

    // Flatten both splits into matrices once, up front.
    let (train_x, train_y) = train.to_matrices();
    let (test_x, test_y) = test.to_matrices();

    // A fixed 10k slice of the training set, for watching the train/test gap.
    let probe: Vec<usize> = (0..10_000).collect();
    let (probe_x, probe_y) = (train_x.select_rows(&probe), train_y.select_rows(&probe));

    let mut rng = Rng::new(42);
    let mut net = Network::random(SHAPE, &mut rng);

    println!("Network {SHAPE:?}  —  {} parameters", net.parameter_count());
    println!("{EPOCHS} epochs, batch size {BATCH_SIZE}, learning rate {LEARNING_RATE}\n");

    println!("  epoch |     loss  |   train acc   test acc  |   time");
    println!("  ------+-----------+-------------------------+--------");

    let started = Instant::now();
    for epoch in 1..=EPOCHS {
        let loss = net.train_epoch_minibatch(&train_x, &train_y, BATCH_SIZE, LEARNING_RATE, &mut rng);

        println!(
            "  {epoch:>5} |  {loss:.5}  |     {:>5.2}%     {:>5.2}%  |  {:>5.1}s",
            net.accuracy(&probe_x, &probe_y) * 100.0,
            net.accuracy(&test_x, &test_y) * 100.0,
            started.elapsed().as_secs_f64(),
        );
    }

    let elapsed = started.elapsed().as_secs_f64();
    let accuracy = net.accuracy(&test_x, &test_y);

    println!("\nFinal test accuracy: {:.2}%", accuracy * 100.0);
    println!(
        "({} of {} digits correct, {} wrong)",
        (accuracy * test_y.rows as f64).round() as usize,
        test_y.rows,
        ((1.0 - accuracy) * test_y.rows as f64).round() as usize,
    );
    println!(
        "Trained in {elapsed:.1}s — {:.2}s per epoch\n",
        elapsed / EPOCHS as f64
    );

    report_confusions(&net, &test_x, &test);
    show_a_mistake(&net, &test_x, &test);

    ExitCode::SUCCESS
}

/// Which digits does it confuse for which?
fn report_confusions(net: &Network, inputs: &Matrix, data: &mnist::MnistData) {
    // One big forward pass for all 10,000 test digits.
    let outputs = net.forward(inputs);

    let mut matrix = [[0usize; 10]; 10];
    for (row, &actual) in data.labels.iter().enumerate() {
        matrix[actual as usize][network::argmax(outputs.row(row))] += 1;
    }

    println!("Confusion matrix — rows are the true digit, columns the guess.");
    println!("The diagonal is correct answers; everything else is a mistake.\n");
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

/// Show one digit the network got wrong.
fn show_a_mistake(net: &Network, inputs: &Matrix, data: &mnist::MnistData) {
    let outputs = net.forward(inputs);

    let mistake = (0..outputs.rows)
        .find(|&r| network::argmax(outputs.row(r)) != data.labels[r] as usize);

    let Some(index) = mistake else {
        println!("No mistakes at all — suspicious.");
        return;
    };

    let scores = outputs.row(index);
    let actual = data.labels[index];
    let guess = network::argmax(scores);

    println!("A digit it got wrong (test image #{index}):\n");
    println!("{}\n", mnist::render(&data.images[index]));
    println!("  true answer: {actual}");
    println!("  it guessed:  {guess}  (confidence {:.2})", scores[guess]);
    println!("  it gave {actual} a score of {:.2}", scores[actual as usize]);
}
