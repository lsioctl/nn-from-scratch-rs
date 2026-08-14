# NN from scratch, with Claude's help

I juste asked claude to teach me how neural networks works, buy building one, from scratch, from
one neuron

All Claude'd info are one NOTES.md

## A look at what Claude does when it refactors:

```bash
Bash(python3 - <<'EOF'
  import re, pathlib

  # layer.rs tests
  p = pathlib.Path("src/layer.rs"); s = p.read_text()
  s = s.replace("""            vec![0.1, -0.6],
          );""", """            vec![0.1, -0.6],
              Activation::Sigmoid,
          );""")
  s = s.replace("let layer = Layer::random(4, 3, &mut rng);", "let layer = Layer::random(4, 3, Activation::Sigmoid, &mut rng);")
  s = s.replace("let layer = Layer::random(5, 3, &mut rng);", "let layer = Layer::random(5, 3, Activation::Sigmoid, &mut rng);")
  s = s.replace("let layer = Layer::random(4, 2, &mut rng);", "let layer = Layer::random(4, 2, Activation::Sigmoid, &mut rng);")
  p.write_text(s)

  # network.rs tests
  p = pathlib.Path("src/network.rs"); s = p.read_text()
  s = s.replace("Network::random(&[3, 4, 2], &mut rng)", "Network::sigmoid_network(&[3, 4, 2], &mut rng)")
  s = s.replace("Network::random(&[2, 4, 1], &mut rng)", "Network::sigmoid_network(&[2, 4, 1], &mut rng)")
  s = s.replace("Network::random(&[3, 5, 2], &mut rng)", "Network::sigmoid_network(&[3, 5, 2], &mut rng)")
  s = s.replace("Network::random(&[2, 3, 2], &mut rng)", "Network::sigmoid_network(&[2, 3, 2], &mut rng)")
  s = s.replace("""            Layer::new(Matrix::zeros(2, 2), vec![0.0, 0.0]),
              Layer::new(Matrix::zeros(3, 1), vec![0.0]),""",
  """            Layer::new(Matrix::zeros(2, 2), vec![0.0, 0.0], Activation::Sigmoid),
              Layer::new(Matrix::zeros(3, 1), vec![0.0], Activation::Sigmoid),""")
  p.write_text(s)
  print("patched")
  EOF
  grep -n "Layer::random\|Network::random\|Network::sigmoid_network\|Layer::new(Matrix" src/layer.rs src/network.rs | head -20)
patched
src/network.rs:57:                    Layer::random(pair[0], pair[1], activation, rng)
src/network.rs:296:        let net = Network::sigmoid_network(&[3, 4, 2], &mut rng);
src/network.rs:358:        let mut net = Network::sigmoid_network(&[2, 4, 1], &mut rng);
src/network.rs:377:        let mut net = Network::sigmoid_network(&[2, 4, 1], &mut rng);
src/network.rs:391:        let net = Network::sigmoid_network(&[3, 5, 2], &mut rng);
src/network.rs:419:        let net = Network::sigmoid_network(&[3, 5, 2], &mut rng);
src/network.rs:433:        let net = Network::sigmoid_network(&[2, 3, 2], &mut rng);
src/network.rs:446:            Layer::new(Matrix::zeros(2, 2), vec![0.0, 0.0], Activation::Sigmoid),
src/network.rs:447:            Layer::new(Matrix::zeros(3, 1), vec![0.0], Activation::Sigmoid),
src/layer.rs:256:        let layer = Layer::random(4, 3, Activation::Sigmoid, &mut rng);
src/layer.rs:270:        let layer = Layer::random(5, 3, Activation::Sigmoid, &mut rng);
src/layer.rs:291:        let layer = Layer::random(4, 2, Activation::Sigmoid, &mut rng);
```

Sometimes it just uses sed

```bash
Bash(sed -n '/for k in 0..net.layers.len()/,/^        }$/p' src/network.rs | head -45)
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
```
