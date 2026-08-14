mod layer;
mod network;
mod neuron;

use layer::Layer;
use network::Network;
use neuron::Neuron;

/// XOR, built by hand from three neurons.
///
///     XOR = (x1 OR x2) AND NOT (x1 AND x2)
///          "at least one"      "but not both"
///
/// The hidden layer computes OR and AND. The output neuron keeps the OR
/// (weight +10) while the AND vetoes it (weight -20 — twice as strong, so
/// it always wins whenever it fires).
fn xor_network() -> Network {
    Network::new(vec![
        Layer::new(vec![
            Neuron::new(vec![10.0, 10.0], -5.0),  // OR
            Neuron::new(vec![10.0, 10.0], -15.0), // AND
        ]),
        Layer::new(vec![Neuron::new(vec![10.0, -20.0], -5.0)]),
    ])
}

fn main() {
    let net = xor_network();

    println!("XOR — \"exactly one of the two inputs is 1\"\n");
    println!("  INPUT     |  HIDDEN LAYER   |  NETWORK SAYS  |  TRUE XOR");
    println!("   x1   x2  |     OR     AND  |  raw    rounded|");
    println!("  ----------+-----------------+----------------+----------");

    for x1 in [0.0, 1.0] {
        for x2 in [0.0, 1.0] {
            // `forward_all` gives us [inputs, hidden, output].
            let acts = net.forward_all(&[x1, x2]);
            let hidden = &acts[1];
            let output = acts[2][0];

            // Round the network's (0, 1) output to an actual bit.
            let rounded = output.round();

            // XOR, computed the boring way, so we have something to compare to.
            let true_xor = if x1 != x2 { 1.0 } else { 0.0 };

            let mark = if rounded == true_xor { "ok" } else { "WRONG" };

            println!(
                "  {x1:>4.0} {x2:>4.0}  | {:>6.3}  {:>6.3}  | {output:.4}    {rounded:.0}   |     {true_xor:.0}   {mark}",
                hidden[0], hidden[1]
            );
        }
    }

    // The plain `forward` when you don't care about the hidden layer.
    let all_correct = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]
    .iter()
    .all(|(input, want)| (net.forward(input)[0] - want).abs() < 0.01);

    println!("\n  all four within 0.01 of the target: {all_correct}");

    println!();
    println!("Read the middle two columns as coordinates. The hidden layer has");
    println!("moved the problem into a new space, and in *that* space the four");
    println!("cases are no longer arranged in a way that defeats a straight line.");
}
