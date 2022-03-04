# Smarty Pants

![Crates.io](https://img.shields.io/crates/v/smarty_pants) ![docs.rs](https://img.shields.io/docsrs/smarty_pants) ![Crates.io](https://img.shields.io/crates/l/smarty_pants) ![Crates.io](https://img.shields.io/crates/d/smarty_pants)

This goal of this library is to:

- Produce `NeuralNetworks` that will always give the same result when given the same input.
- Provide methods and functions for the creation, training, running, and parsing of `NeuralNetworks`
- Be relatively light wheight and fast.

## USAGE

Add this to your Cargo.toml:

``` Toml
[dependencies]
smarty_pants = "0.2.0"
```

To create a new network simply call the new function with the wanted parameters and store it somewhere. Make sure it's mutable other wise some of the functions and methods may not work.

``` Rust
use smarty_pants::neural_network::*;

fn main() {
    let mut network:NeuralNetwork = NeuralNetwork::new(1.0,10,10,3);
}
```

Then simply call the `run()` method to run it with the arguments as the input/s.

``` Rust
let output:Vec<f64> = network.run(vec![1.0,2.0,3.0]);
```

It will output a `Vector<f64>` containing the output of the network. For more information please see the [documentation](https://docs.rs/smarty_pants/latest/smarty_pants/) or a more detailed [example](https://github.com/Merlin1846/smarty_pants/tree/master/examples).
