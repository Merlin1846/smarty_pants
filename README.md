# Smarty Pants

This goal of this library is to:

- Produce `NeuralNetworks` that will always give the same result when given the same input.
- Provide methods and functions for the creation, training, running, and parsing of `NeuralNetworks`
- Be relatively light wheight and fast.

## USAGE

Add this to your Cargo.toml:

``` Rust
[dependencies]
smarty_pants = "0.1.0"
```

To create a new network simply call the new function with the wanted parameters and store it somewhere. Make sure it's mutable other wise many of the functions may not work.

``` Rust
use smarty_pants::neural_network::*;

fn main() {
    let mut network:NeuralNetwork = NeuralNetwork::new(1.0,10,10,3);
}
```

Then simply call the `run()` method to run it.

``` Rust
let output:Vec<f64> = network.run();
```

It will output a `Vector<f64>` containing the output of the network. For more information please see the [documentation](https://docs.rs/smart_pants/1.0.0) or a more detailed [example](https://github.com/Merlin1846/smarty_pants/tree/master/examples).
