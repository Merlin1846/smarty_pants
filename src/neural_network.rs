/*!
This `module` gives access to `NeuralNetwork` related things and most importantly
it gives access to the `NeuralNetwork` `Type` itself. It also contains some 
convenience functions such as `batch_run()` and `batch_mutate()`
*/

use std::fmt::Error;
use rand::prelude::*;
use serde::{Serialize, Deserialize};

/**
The `NeuralNetwork` type from which all learning functions come from, stores data for the
network and gives access to functions which can be used to access, run, and mutate the
network.

```
use smarty_pants::neural_network::NeuralNetwork;

let mut brain:NeuralNetwork = NeuralNetwork::new(1.0,10,10,3);

brain.set_wheight(10.0,(5,7));
assert!(brain.get_wheight((5,7)).unwrap() == 10.0);

let output:Vec<f64> = brain.run(&vec![1.0,2.0,3.0,4.0,5.0]);
```
*/
//#[derive(Serialize, Deserialize, Debug, Clone)]
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
//#[derive(Clone)]
pub struct NeuralNetwork {
    /// A 2D Vector of all hidden `neuron`, every `neuron` is a `f64` that is its `wheight`
    hidden_layers: Vec<Vec<f64>>,
    /// The `wheight` of the outputs.
    output_weights: Vec<f64>
}

impl NeuralNetwork {
    /// Creates a new `NeuralNetwork` using the specified arguments.
    pub fn new(default_wheight:f64 ,hidden_layers:usize ,hidden_neurons_per_layer:usize ,outputs:usize) -> NeuralNetwork {
        NeuralNetwork {
            hidden_layers: vec![vec![default_wheight;hidden_neurons_per_layer];hidden_layers],
            output_weights: vec![default_wheight;outputs]
        }
    }

    /// Creates a new `NeuralNetwork` using the inputs as the `weights`
    pub fn new_from(hidden_layers:Vec<Vec<f64>>, output_weights:Vec<f64>) -> NeuralNetwork {
        NeuralNetwork {
            hidden_layers: {
                let mut layers:Vec<Vec<f64>> = Vec::with_capacity(hidden_layers.len());
                for layer in hidden_layers.iter() {
                    layers.push(Vec::with_capacity(layer.len()));
                    for neuron in 0..layer.len() {
                        layers.last_mut().unwrap().push(layer[neuron]);
                    }
                }
                layers
            },
            output_weights:output_weights
        }
    }

    /// Runs the NeuralNetwork using the provided arguments, then returns the output
    pub fn run(&mut self, inputs:&Vec<f64>) -> Vec<f64> {
        // Create a place to store the temporary values.
        let mut temp:Vec<Vec<f64>> = vec![vec![0.0;self.hidden_layers[0].len()];self.hidden_layers.len()];

        // For each input pass the value to the first `hidden_layer` and multiply by the `neurons` wheight
        for neuron in 0..inputs.len() {
            for temp_neuron in 0..self.hidden_layers.len() {
                temp[0][temp_neuron] += inputs[neuron]*(self.hidden_layers[0][temp_neuron]);
            }
        }

        // For each `neuron` in each `hidden_layer` push the values forwards towards the last layer
        for layer in 0..(temp.len()-1) {
            for neuron in 0..temp[layer].len() {
                for next_neuron in 0..temp[layer+1].len() {
                    // Set `next_neuron` to the `current neuron`, passing the value forwards through the layers
                    temp[layer+1][next_neuron] += temp[layer][neuron]*self.hidden_layers[layer+1][next_neuron];
                }
            }
        }

        // Now that the values have reached the last layer transfer the values from the last layer to each output. Then return the outputs
        let mut outputs:Vec<f64> = vec![0.0f64;self.output_weights.len()];
        for neuron in temp[temp.len()-1].iter() {
            for output in 0..self.output_weights.len() {
                outputs[output] += neuron*self.output_weights[output];
            }
        }
        outputs
    }

    /**
    Sets the wheight of a single `neuron` in the hidden layers.
    If the specified neuron is out of bounds then it will return an error in the form of a `Option<String>`
    This will contain text that be either outputted or ignored and simply checked if it exists.
    ```
    use smarty_pants::neural_network::NeuralNetwork;

    let mut brain:NeuralNetwork = NeuralNetwork::new(1.0,10,10,3);

    match brain.set_wheight(64f64 ,(16usize,23usize)) {
        None => println!("No error"),
        Some(e) => println!("Error: {}", e)
    }
    ```
    */
    pub fn set_wheight(&mut self, wheight:f64 ,neuron:(usize,usize)) -> Option<String> {
        if neuron.0 < self.hidden_layers.len() {
            if neuron.1 < self.hidden_layers.len() {
                self.hidden_layers[neuron.0][neuron.1] = wheight;
                None
            } else {
                Some("Error setting wheight of a neuron, the neuron is out of bounds on the y axis.".to_owned())
            }
        } else {
            Some("Error setting wheight of a neuron, the neuron is out of bounds on the x axis.".to_owned())
        }
    }

    /**
    Gets the wheight of a single `neuron` in the `hidden_layers`
    Returns an error if the specified `neuron` is greater than the bounds on the `hidden_layers`
    ```
    use smarty_pants::neural_network::NeuralNetwork;

    let mut brain:NeuralNetwork = NeuralNetwork::new(1.0,10,10,3);

    match brain.get_wheight((16usize,23usize)) {
        Ok(_) => println!("No error"),
        Err(e) => println!("Error")
    }
    ```
    */
    pub fn get_wheight(&self, neuron:(usize,usize)) -> Result<f64, Error> {
        if neuron.0 < self.hidden_layers.len() {
            if neuron.1 < self.hidden_layers[neuron.0].len() {
                Ok(self.hidden_layers[neuron.0][neuron.1])
            } else {
                Err(Error)
            }
        } else {
            Err(Error)
        }
    }

    /// Mutates every `wheight` in the `NeuralNetwork` by a random amount that is a maximum of `max`
    /// in both the possitive and negative directions. It does this through addition and subtraction.
    /// if `outputs` is true then it will also mutate the `output_weights`
    pub fn mutate(&mut self, mutation_rate:f64, outputs:bool) {
        let mut rng:ThreadRng = thread_rng();
        for layer in self.hidden_layers.iter_mut() {
            for neuron in layer.iter_mut() {
                *neuron += rng.gen_range(-mutation_rate..=mutation_rate);
            }
        }

        if outputs {
            for neuron in self.output_weights.iter_mut() {
                *neuron += rng.gen_range(-mutation_rate..=mutation_rate);
            }
        }
    }

    /// Returns a `Vector` containing `amount` number of neural networks all with the same starting values.
    /// This function does this by repeatedly calling `NeuralNetwork::new()` so it isn't any more efficent, its simply
    /// here for convenience.
    pub fn batch_new(amount:usize ,default_wheight:f64 ,hidden_layers:usize ,hidden_neurons_per_layer:usize ,outputs:usize) -> Vec<NeuralNetwork> {
        let mut networks: Vec<NeuralNetwork> = Vec::with_capacity(amount);
        networks.reserve_exact(amount);
        for _ in 0..amount {
            networks.push(NeuralNetwork::new(default_wheight, hidden_layers, hidden_neurons_per_layer, outputs));
        }
        networks
    }

    /// Returns the `hidden_layers` `weights` of the network.
    pub fn get_weights(&self) -> Vec<Vec<f64>> {
        let mut weights:Vec<Vec<f64>> = Vec::with_capacity(self.hidden_layers.len());
        for layer in self.hidden_layers.iter() {
            let mut layer_weights:Vec<f64> = Vec::with_capacity(layer.len());
            for wheight in layer.iter() {
                layer_weights.push(*wheight);
            }
            weights.push(layer_weights);
        }
        weights
    }

    /// Returns the `output_weights` of the network.
    pub fn get_output_weights(&self) -> Vec<f64> {
        self.output_weights.clone()
    }
}

/// Returns a `Vector` of `Vector`s that makes up the output of all `NeuralNetworks` given to this function.
/// This function does this by repeatedly calling `NeuralNetwork::run()` so it isn't any more efficent, its simply
/// here for convenience.
pub fn batch_run(networks:&mut Vec<NeuralNetwork>, inputs:&Vec<f64>) -> Vec<Vec<f64>> {
    let mut output: Vec<Vec<f64>> = Vec::with_capacity(networks.len());
    for network in networks.iter_mut() {
        output.push(network.run(&inputs));
    }
    output
}

/// Turns one inputted `NeuralNetwork` into `amount` number of mutated `NeuralNetwork`s mutated by mutation.
/// If `outputs` is true then it will also mutate the output weights. It does this through cloning and
/// calling `NeuralNetwork::mutate(mutation)` so it isn't any more efficent, its simply here for convenience.
pub fn batch_mutate(amount:usize, mutation_rate:f64, network:&NeuralNetwork, outputs:bool) -> Vec<NeuralNetwork> {
    let mut networks: Vec<NeuralNetwork> = Vec::with_capacity(amount);
    for i in 0..amount {
        networks.push(network.clone());
        networks[i].mutate(mutation_rate, outputs);
    }
    networks
}
