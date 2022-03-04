/*!
This example's purpose is to show how to save and load a `NeuralNetwork`
with `bincode` and `serde`.
*/

use smarty_pants::neural_network::*;
use std::fs::File;
use std::io::{Write, Read};

fn main() {
    // Create a `NeuralNetwork`.
    let mut network:NeuralNetwork = NeuralNetwork::new(1.0,1,3,1);
    // Run the network and save the output.
    let output:f64 = network.run(&vec![1.0])[0];
    // Save it to file.
    write_file("example.brain", &network).unwrap();
    // Load it.
    let mut loaded_network:NeuralNetwork = read_file("example.brain").unwrap();
    // Call `run()` on the newly loaded `NeuralNetwork`.
    let output2:f64 = loaded_network.run(&vec![1.0])[0];
    // Show that the `NeuralNetwork` functions the same as it did before.
    assert!(output == output2);
}

/// Writes to the file at `path` creating it if necessary.
fn write_file(path:&str, data:&NeuralNetwork) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(&bincode::serialize(data).unwrap())?;
    Ok(())
}

/// Reads the file at `path` and returns the entire file as a single String.
fn read_file(path:&str) -> std::io::Result<NeuralNetwork> {
    let mut file:File = File::open(path)?;
    let mut contents:Vec<u8> = Vec::new();
    file.read_to_end(&mut contents)?;
    Ok(bincode::deserialize(&contents).unwrap())
}