class NeuralNetwork {
    constructor(layers) {
        this.layers = [];
        this.learning_rate = 0.01;

        // Create layers
        for (let i = 0; i < layers.length - 1; i++) {
            this.layers.push(new Layer(layers[i], layers[i + 1]));
        }
    }
    
    // Forward pass through the network
    forward(input) {
        let output = input;
        for (const layer of this.layers) {
            let sigmoid = layer.output_size === 10; // Use sigmoid for output layer
            output = layer.forward(output, sigmoid);
        }
        return output;
    }
    
    // Load model from serialized data
    fromJSON(data) {
        this.learning_rate = data.learning_rate || 0.01;
        this.layers = [];
        
        for (let i = 0; i < data.layers.length; i++) {
            const layerData = data.layers[i];
            const layer = new Layer(layerData.input_size, layerData.output_size);
            layer.weights = layerData.weights;
            layer.biases = layerData.biases;
            this.layers.push(layer);
        }
    }
}

class Layer {
    constructor(input_size, output_size) {
        this.input_size = input_size;
        this.output_size = output_size;
        this.weights = Array.from({ length: output_size }, () => 
            Array.from({ length: input_size }, () => Math.random() * 0.2 - 0.1));
        this.biases = Array.from({ length: output_size }, () => Math.random() * 0.2 - 0.1);
    }

    // Forward pass through the layer
    forward(input, use_sigmoid) {
        if (input.length !== this.input_size) {
            throw new Error(`Input size does not match layer input size. Expected ${this.input_size}, got ${input.length}.`);
        }
        const output = Array(this.output_size).fill(0);
        for (let i = 0; i < this.output_size; i++) {
            for (let j = 0; j < this.input_size; j++) {
                output[i] += input[j] * this.weights[i][j];
            }
            output[i] += this.biases[i];

            if (use_sigmoid) {
                // Apply sigmoid activation function
                output[i] = 1 / (1 + Math.exp(-output[i]));
            } else {
                // Apply ReLU activation function
                output[i] = Math.max(0, output[i]);
            }
        }

        return output;
    }
}