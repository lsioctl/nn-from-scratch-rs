// We build this project up one concept at a time, so some modules are fully
// written and tested before `main` has any use for them.
#![allow(dead_code)]

mod layer;
mod loss;
mod mnist;
mod network;
mod neuron;
mod rng;

use std::process::ExitCode;
use std::time::Instant;

use network::Network;
use rng::Rng;

const DATA_DIR: &str = "data";

/// 784 pixels in, one hidden layer of 30, 10 digit outputs.
///
/// 30 hidden neurons is deliberately modest: it is enough for ~95% accuracy
/// and small enough to train in seconds. Note the shape of the compression —
/// 784 numbers squeezed through 30 and back out to 10. The hidden layer has
/// no room to memorise images, so it is forced to find features.
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

    // Sanity check: look at the data before trusting it. A parser that
    // silently transposes or offsets its input will produce plausible-looking
    // numbers and a network that mysteriously refuses to learn.
    println!("First training digit (label: {}):\n", train.labels[0]);
    println!("{}\n", mnist::render(&train.images[0]));

    let train_samples = train.to_samples();
    let test_samples = test.to_samples();

    let mut rng = Rng::new(42);
    let mut net = Network::random(SHAPE, &mut rng);

    println!("Network {SHAPE:?}  —  {} weights and biases", count_parameters(&net));
    println!("{EPOCHS} epochs, batch size {BATCH_SIZE}, learning rate {LEARNING_RATE}\n");

    println!("  epoch |     loss  |   train acc   test acc  |   time");
    println!("  ------+-----------+-------------------------+--------");

    let started = Instant::now();
    for epoch in 1..=EPOCHS {
        let loss = net.train_epoch_minibatch(&train_samples, BATCH_SIZE, LEARNING_RATE, &mut rng);

        // Evaluating on all 60k every epoch is slow and tells us little, so
        // we check training accuracy on a fixed 10k slice.
        let train_accuracy = net.accuracy(&train_samples[..10_000]);
        let test_accuracy = net.accuracy(&test_samples);

        println!(
            "  {epoch:>5} |  {loss:.5}  |     {:>5.2}%     {:>5.2}%  |  {:>5.1}s",
            train_accuracy * 100.0,
            test_accuracy * 100.0,
            started.elapsed().as_secs_f64(),
        );
    }

    let test_accuracy = net.accuracy(&test_samples);
    println!("\nFinal test accuracy: {:.2}%", test_accuracy * 100.0);
    println!("({} of {} digits correct, {} wrong)\n",
        (test_accuracy * test_samples.len() as f64).round() as usize,
        test_samples.len(),
        ((1.0 - test_accuracy) * test_samples.len() as f64).round() as usize,
    );

    report_confusions(&net, &test_samples, &test);
    show_a_mistake(&net, &test_samples, &test);

    ExitCode::SUCCESS
}

fn count_parameters(net: &Network) -> usize {
    net.layers
        .iter()
        .flat_map(|l| &l.neurons)
        .map(|n| n.weights.len() + 1)
        .sum()
}

/// Which digits does it confuse for which?
fn report_confusions(
    net: &Network,
    samples: &[(Vec<f64>, Vec<f64>)],
    data: &mnist::MnistData,
) {
    let mut matrix = [[0usize; 10]; 10];
    for ((inputs, _), &actual) in samples.iter().zip(&data.labels) {
        matrix[actual as usize][net.predict(inputs)] += 1;
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
            if actual == guess {
                print!("{:>5}", count);
            } else if count == 0 {
                print!("    .");
            } else {
                print!("{:>5}", count);
            }
        }
        let total: usize = row.iter().sum();
        println!("     {:>5.1}%", row[actual] as f64 / total as f64 * 100.0);
    }
    println!();
}

/// Show one digit the network got wrong — usually humbling, occasionally
/// reassuring, since plenty of MNIST digits are genuinely ambiguous.
fn show_a_mistake(net: &Network, samples: &[(Vec<f64>, Vec<f64>)], data: &mnist::MnistData) {
    let mistake = (0..samples.len())
        .find(|&i| net.predict(&samples[i].0) != data.labels[i] as usize);

    let Some(index) = mistake else {
        println!("No mistakes at all — suspicious.");
        return;
    };

    let inputs = &samples[index].0;
    let actual = data.labels[index];
    let outputs = net.forward(inputs);
    let guess = network::argmax(&outputs);

    println!("A digit it got wrong (test image #{index}):\n");
    println!("{}\n", mnist::render(&data.images[index]));
    println!("  true answer: {actual}");
    println!("  it guessed:  {guess}  (confidence {:.2})", outputs[guess]);
    println!("  it gave {actual} a score of {:.2}", outputs[actual as usize]);
}
