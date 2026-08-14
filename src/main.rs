mod neuron;

use neuron::Neuron;

/// Print a neuron's behaviour on all four two-bit inputs.
fn truth_table(name: &str, n: &Neuron) {
    println!("{name}:");
    println!("   x1   x2  |      z   ->  output");
    println!("  ----------+---------------------");
    for x1 in [0.0, 1.0] {
        for x2 in [0.0, 1.0] {
            let inputs = [x1, x2];
            let z = n.net_input(&inputs);
            let y = n.forward(&inputs);
            println!("  {x1:>4.0} {x2:>4.0}  | {z:>6.1}  ->  {y:.4}");
        }
    }
    println!();
}

fn main() {
    // We are *hand-picking* these weights, not learning them. The point of
    // this step is to see that a neuron is just arithmetic — nothing
    // mysterious is happening yet.

    // AND: needs a big total to overcome a bias of -15, so both inputs
    // must be 1 (10 + 10 = 20 > 15).
    truth_table("AND  w=[10, 10]  b=-15", &Neuron::new(vec![10.0, 10.0], -15.0));

    // OR: a smaller bias of -5, so a single 1 is already enough.
    truth_table("OR   w=[10, 10]  b=-5", &Neuron::new(vec![10.0, 10.0], -5.0));

    // NOT x1: a negative weight means the input argues *against* firing.
    truth_table("NOT  w=[-10, 0]  b=5", &Neuron::new(vec![-10.0, 0.0], 5.0));
}
