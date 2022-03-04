pub mod neural_network;

/// Just a testing module, only here for developers of the library itself.
/// The functions defined in this module are in this order so that if a
/// previous test worked then in theory the next one should work.
#[cfg(test)]
mod tests {
    use crate::neural_network::NeuralNetwork;

    #[test]
    /// Test if `NeuralNetwork::new()` is working.
    fn new() {
        #[allow(unused_variables)]
        let network:NeuralNetwork = NeuralNetwork::new(1.0,10,10,3);
    }

    #[test]
    /// Test if `NeuralNetwork::new_from()` is working.
    fn new_from() {
        let hidden_layers:Vec<Vec<f64>> = vec![vec![1.0,1.0,1.0]];
        let output_weights:Vec<f64> = vec![1.0,1.0];
        #[allow(unused_variables)]
        let network:NeuralNetwork = NeuralNetwork::new_from(hidden_layers,output_weights);
    }

    #[test]
    /// Test if `NeuralNetwork::batch_new()` is working.
    fn batch_new() {
        #[allow(unused_variables)]
        let networks: Vec<NeuralNetwork> = NeuralNetwork::batch_new(5,1.0,1,3,2);
    }

    #[test]
    /// Test if the `set_wheight` and `get_wheight` functions are working
    fn weights() {
        let mut network:NeuralNetwork = NeuralNetwork::new(1.0,10,10,3);
        assert!(network.set_wheight(42.4242424242, (5,8)) == None);
        assert!(network.get_wheight((5,8)).unwrap() == 42.4242424242);
    }

    #[test]
    /// Test if `NeuralNetwork::mutate()` is working.
    fn mutate() {
        let mut network:NeuralNetwork = NeuralNetwork::new(1.0,10,10,3);
        network.mutate(10.0f64, true);
        assert!(1.0 != network.get_wheight((0,0)).unwrap());
    }

    #[test]
    /// Test if `neural_network::NeuralNetwork::batch_mutate()` is working.
    fn batch_mutate() {
        let networks: Vec<NeuralNetwork> = crate::neural_network::batch_mutate(5,0.25,&mut NeuralNetwork::new(1.0,1,3,2), true);
        for network in networks.iter() {
            assert!(1.0 != network.get_wheight((0,0)).unwrap());
        }
    }

    #[test]
    /// Test if `NeuralNetwork::run()` is working.
    fn run() {
        let mut network:NeuralNetwork = NeuralNetwork::new(1.0,10,10,3);
        let data:Vec<f64> = network.run(&vec![1.0,1.0,1.0]);
        assert!(data[0] == 30000000000.0);
        assert!(data[1] == 30000000000.0);
        assert!(data[2] == 30000000000.0);
    }

    #[test]
    /// Test if `neural_network::NeuralNetwork::batch_run()` is working.
    fn batch_run() {
        let mut networks: Vec<NeuralNetwork> = NeuralNetwork::batch_new(5,1.0,10,10,3);
        let data:Vec<Vec<f64>> = crate::neural_network::batch_run(&mut networks, &vec![1.0,1.0,1.0]);
        for output in data.iter() {
            assert!(output[0] == 30000000000.0);
            assert!(output[1] == 30000000000.0);
            assert!(output[2] == 30000000000.0);
        }
    }
}
