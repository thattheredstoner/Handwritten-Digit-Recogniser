use byteorder::{BigEndian, ReadBytesExt};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, Read, Result, Write};
use std::time::Instant;
use std::sync::{Arc, Mutex};
use std::net::{TcpListener, TcpStream};
use std::thread;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    cost: f64,
    learning_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    input_size: usize,
    output_size: usize,
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    // For momentum-based optimization
    weight_momentum: Vec<Vec<f64>>,
    bias_momentum: Vec<f64>,
}

#[derive(Debug)]
pub struct MnistData {
    pub images: Vec<Vec<u8>>,
    pub count: usize,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingRequest {
    pub epochs: Option<usize>,
    pub batch_size: Option<usize>,
    pub learning_rate: Option<f64>,
    pub momentum: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingResponse {
    pub status: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingStatus {
    pub status: String, // "idle", "running", "completed", "failed"
    pub progress: f64,  // 0.0 to 100.0
    pub current_epoch: usize,
    pub total_epochs: usize,
    pub current_batch: usize,
    pub total_batches: usize,
    pub current_loss: f64,
    pub current_accuracy: f64,
    pub elapsed_time: f64,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TestResult {
    pub actual: u8,
    pub predicted: u8,
    pub confidence: f64,
    pub prediction_scores: Vec<f64>,
    pub is_correct: bool,
    pub image: Vec<u8>, // Raw image data (28x28 = 784 bytes)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TestData {
    pub results: Vec<TestResult>,
    pub accuracy: f64,
    pub total_correct: usize,
    pub total_samples: usize,
    pub misclassified_count: usize,
}

// Global state - single training session
type TrainingState = Arc<Mutex<TrainingStatus>>;
type ModelState = Arc<Mutex<Option<NeuralNetwork>>>;
type TestDataState = Arc<Mutex<Option<TestData>>>;

impl Default for TrainingStatus {
    fn default() -> Self {
        Self {
            status: "idle".to_string(),
            progress: 0.0,
            current_epoch: 0,
            total_epochs: 0,
            current_batch: 0,
            total_batches: 0,
            current_loss: 0.0,
            current_accuracy: 0.0,
            elapsed_time: 0.0,
            message: "Ready to start training".to_string(),
        }
    }
}

impl Default for TrainingRequest {
    fn default() -> Self {
        Self {
            epochs: Some(10),
            batch_size: Some(32),
            learning_rate: Some(0.01),
            momentum: Some(0.9),
        }
    }
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = thread_rng();
        
        // Xavier/Glorot initialization
        let limit = (6.0 / (input_size + output_size) as f64).sqrt();
        
        let weights = (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.gen_range(-limit..limit))
                    .collect()
            })
            .collect();
        
        let biases = vec![0.0; output_size];
        let weight_momentum = vec![vec![0.0; input_size]; output_size];
        let bias_momentum = vec![0.0; output_size];
        
        Self {
            input_size,
            output_size,
            weights,
            biases,
            weight_momentum,
            bias_momentum,
        }
    }
    
    pub fn forward(&self, input: &[f64], use_sigmoid: bool) -> Vec<f64> {
        let mut output = vec![0.0; self.output_size];
        
        for i in 0..self.output_size {
            let mut sum = self.biases[i];
            for j in 0..self.input_size {
                sum += input[j] * self.weights[i][j];
            }
            
            if use_sigmoid {
                // Sigmoid activation for output layer
                output[i] = 1.0 / (1.0 + (-sum).exp());
            } else {
                // ReLU activation for hidden layers
                output[i] = if sum > 0.0 { sum } else { 0.0 };
            }
        }
        
        output
    }
    
    pub fn backward_with_momentum(
        &mut self, 
        input: &[f64], 
        output_gradient: &[f64], 
        learning_rate: f64,
        momentum: f64,
        is_output_layer: bool
    ) -> Vec<f64> {
        let mut input_gradient = vec![0.0; self.input_size];
        
        // Get the current layer output for derivative calculation
        let output = if is_output_layer {
            self.forward(input, true)
        } else {
            self.forward(input, false)
        };
        
        for i in 0..self.output_size {
            let derivative = if is_output_layer {
                // Sigmoid derivative
                output[i] * (1.0 - output[i])
            } else {
                // ReLU derivative
                if output[i] > 0.0 { 1.0 } else { 0.0 }
            };
            
            let error = output_gradient[i] * derivative;
            
            for j in 0..self.input_size {
                // Calculate gradient for input
                input_gradient[j] += error * self.weights[i][j];
                
                // Update momentum for weights
                let weight_gradient = error * input[j];
                self.weight_momentum[i][j] = momentum * self.weight_momentum[i][j] + learning_rate * weight_gradient;
                
                // Update weights with momentum
                self.weights[i][j] += self.weight_momentum[i][j];
            }
            
            // Update momentum for biases
            self.bias_momentum[i] = momentum * self.bias_momentum[i] + learning_rate * error;
            
            // Update biases with momentum
            self.biases[i] += self.bias_momentum[i];
        }
        
        input_gradient
    }
}

impl NeuralNetwork {
    pub fn new(layers: &[usize]) -> Self {
        let mut network_layers = Vec::new();
        
        for i in 0..layers.len() - 1 {
            network_layers.push(Layer::new(layers[i], layers[i + 1]));
        }
        
        Self {
            layers: network_layers,
            cost: f64::INFINITY,
            learning_rate: 0.01,
        }
    }
    
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut output = input.to_vec();
        
        for (i, layer) in self.layers.iter().enumerate() {
            let is_output_layer = i == self.layers.len() - 1;
            output = layer.forward(&output, is_output_layer);
        }
        
        output
    }
    
    pub fn train_batch(
        &mut self, 
        batch_inputs: &[Vec<f64>], 
        batch_outputs: &[Vec<f64>], 
        learning_rate: f64,
        momentum: f64
    ) {
        let batch_size = batch_inputs.len();
        let mut total_cost = 0.0;
        
        for (input, expected_output) in batch_inputs.iter().zip(batch_outputs.iter()) {
            // Forward pass to get all outputs
            let mut outputs = vec![input.clone()];
            let mut current_output = input.clone();
            
            for (i, layer) in self.layers.iter().enumerate() {
                let is_output_layer = i == self.layers.len() - 1;
                current_output = layer.forward(&current_output, is_output_layer);
                outputs.push(current_output.clone());
            }
            
            // Calculate output layer error (using cross-entropy for classification)
            let final_output = &outputs[outputs.len() - 1];
            let mut output_errors = Vec::new();
            
            let mut cost = 0.0;
            for i in 0..expected_output.len() {
                let error = expected_output[i] - final_output[i];
                output_errors.push(error);
                cost += error * error; // MSE for simplicity
            }
            cost /= expected_output.len() as f64;
            total_cost += cost;
            
            // Backward pass through all layers
            let mut layer_errors = output_errors;
            for i in (0..self.layers.len()).rev() {
                let is_output_layer = i == self.layers.len() - 1;
                layer_errors = self.layers[i].backward_with_momentum(
                    &outputs[i], 
                    &layer_errors, 
                    learning_rate / batch_size as f64, // Scale learning rate by batch size
                    momentum,
                    is_output_layer
                );
            }
        }
        
        self.cost = total_cost / batch_size as f64;
    }
    
    pub fn evaluate(&self, test_inputs: &[Vec<f64>], test_labels: &[u8]) -> (f64, usize, usize) {
        let mut correct = 0;
        let total = test_inputs.len();
        
        for (i, input) in test_inputs.iter().enumerate() {
            let prediction = self.forward(input);
            let predicted_digit = prediction
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap() as u8;
            
            if predicted_digit == test_labels[i] {
                correct += 1;
            }
        }
        
        let accuracy = correct as f64 / total as f64;
        (accuracy, correct, total)
    }
    
    pub fn save_to_file(&self, filename: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(filename, json)?;
        Ok(())
    }
    
    pub fn load_from_file(filename: &str) -> Result<Self> {
        let json = std::fs::read_to_string(filename)?;
        let network = serde_json::from_str(&json)?;
        Ok(network)
    }
}

impl MnistData {
    pub fn load_images(filename: &str) -> Result<Self> {
        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);
        
        // Read magic number (should be 2051 for images)
        let magic = reader.read_u32::<BigEndian>()?;
        if magic != 2051 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid magic number for images: {}", magic),
            ));
        }
        
        // Read dimensions
        let count = reader.read_u32::<BigEndian>()? as usize;
        let height = reader.read_u32::<BigEndian>()? as usize;
        let width = reader.read_u32::<BigEndian>()? as usize;
        
        // Read image data
        let mut images = Vec::with_capacity(count);
        for _ in 0..count {
            let mut image = vec![0u8; width * height];
            reader.read_exact(&mut image)?;
            images.push(image);
        }
        
        Ok(Self {
            images,
            count,
            width,
            height,
        })
    }
    
    pub fn load_labels(filename: &str) -> Result<Vec<u8>> {
        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);
        
        // Read magic number (should be 2049 for labels)
        let magic = reader.read_u32::<BigEndian>()?;
        if magic != 2049 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid magic number for labels: {}", magic),
            ));
        }
        
        // Read count
        let count = reader.read_u32::<BigEndian>()? as usize;
        
        // Read labels
        let mut labels = vec![0u8; count];
        reader.read_exact(&mut labels)?;
        
        Ok(labels)
    }
}

fn normalize_image(image: &[u8]) -> Vec<f64> {
    image.iter().map(|&pixel| pixel as f64 / 255.0).collect()
}

fn create_one_hot(label: u8, num_classes: usize) -> Vec<f64> {
    let mut one_hot = vec![0.0; num_classes];
    one_hot[label as usize] = 1.0;
    one_hot
}

fn data_augmentation(image: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut rng = thread_rng();
    
    // Random shift augmentation
    let shift_x = rng.gen_range(-3..=3);
    let shift_y = rng.gen_range(-3..=3);
    let mut shifted_pixels = vec![0u8; width * height];
        
    for y in 0..height {
        for x in 0..width {
            let new_x = (x as i32 + shift_x).clamp(0, (width - 1) as i32) as usize;
            let new_y = (y as i32 + shift_y).clamp(0, (height - 1) as i32) as usize;
            shifted_pixels[new_y * width + new_x] = image[y * width + x];
        }
    }
    shifted_pixels
}

fn train_neural_network(
    network: &mut NeuralNetwork,
    train_images: &MnistData,
    train_labels: &[u8],
    test_images: &MnistData,
    test_inputs: &[Vec<f64>],
    test_labels: &[u8],
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    momentum: f64,
    training_state: TrainingState,
    test_data_state: TestDataState,
) {
    let total_images = train_images.count;
    let total_batches = (total_images + batch_size - 1) / batch_size;
    let training_start = Instant::now();
    
    println!("Starting training session");
    println!("  {} training samples", total_images);
    println!("  {} test samples", test_inputs.len());
    println!("  {} epochs, batch size {}", epochs, batch_size);
    println!("  Learning rate: {}, Momentum: {}", learning_rate, momentum);
    
    // Update status to running
    {
        let mut state = training_state.lock().unwrap();
        state.status = "running".to_string();
        state.total_epochs = epochs;
        state.total_batches = total_batches;
        state.message = "Training started".to_string();
    }
    
    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        
        // Check if training was cancelled
        {
            let state = training_state.lock().unwrap();
            if state.status == "cancelled" {
                println!("Training cancelled by user");
                // Ensure the status is properly set before returning
                drop(state);
                let mut final_state = training_state.lock().unwrap();
                final_state.status = "cancelled".to_string();
                final_state.message = "Training was cancelled by user".to_string();
                return;
            }
        }
        
        // Shuffle training data indices
        let mut indices: Vec<usize> = (0..total_images).collect();
        indices.shuffle(&mut thread_rng());
        
        for batch in 0..total_batches {
            // Check if training was cancelled (more frequent check)
            {
                let state = training_state.lock().unwrap();
                if state.status == "cancelled" {
                    println!("Training cancelled by user during batch processing");
                    // Ensure the status is properly set before returning
                    drop(state);
                    let mut final_state = training_state.lock().unwrap();
                    final_state.status = "cancelled".to_string();
                    final_state.message = "Training was cancelled by user".to_string();
                    return;
                }
            }
            
            let start_idx = batch * batch_size;
            let end_idx = (start_idx + batch_size).min(total_images);
            
            // Prepare batch with data augmentation
            let mut batch_inputs = Vec::new();
            let mut batch_outputs = Vec::new();
            
            for &idx in &indices[start_idx..end_idx] {
                // Apply data augmentation to the image
                let augmented_image = data_augmentation(
                    &train_images.images[idx], 
                    train_images.width, 
                    train_images.height
                );
                
                // Normalize the augmented image
                let normalized_pixels = normalize_image(&augmented_image);
                
                // Create one-hot encoded label
                let expected_output = create_one_hot(train_labels[idx], 10);
                
                batch_inputs.push(normalized_pixels);
                batch_outputs.push(expected_output);
            }
            
            // Train on batch
            network.train_batch(&batch_inputs, &batch_outputs, learning_rate, momentum);
            
            // Update progress every 50 batches
            if batch % 50 == 0 {
                let elapsed = training_start.elapsed().as_secs_f64();
                let progress = ((epoch * total_batches + batch) as f64 / (epochs * total_batches) as f64) * 100.0;
                
                // Update status
                {
                    let mut state = training_state.lock().unwrap();
                    state.progress = progress;
                    state.current_epoch = epoch + 1;
                    state.current_batch = batch + 1;
                    state.current_loss = network.cost;
                    state.elapsed_time = elapsed;
                    state.message = format!("Epoch {}/{}, Batch {}/{}", epoch + 1, epochs, batch + 1, total_batches);
                }
                
                print!("  Epoch {}/{}, Batch {}/{}, Loss: {:.4}\r", 
                       epoch + 1, epochs, batch + 1, total_batches, network.cost);
                use std::io::{self, Write};
                io::stdout().flush().unwrap();
            }
        }
        
        let epoch_duration = epoch_start.elapsed();
        
        // Evaluate on test set
        let (accuracy, correct, total) = network.evaluate(test_inputs, test_labels);
        
        // Update status with epoch completion
        {
            let mut state = training_state.lock().unwrap();
            state.current_accuracy = accuracy;
            state.message = format!("Completed epoch {}/{}", epoch + 1, epochs);
        }
        
        println!("\nEpoch {}/{} completed in {:.2}s", epoch + 1, epochs, epoch_duration.as_secs_f64());
        println!("  Final loss: {:.4}", network.cost);
        println!("  Test accuracy: {:.2}% ({}/{} correct)", accuracy * 100.0, correct, total);
        println!();
    }

    // Stop if cancelled during training
    {
        let state = training_state.lock().unwrap();
        if state.status == "cancelled" {
            println!("Training cancelled by user");
            return;
        }
    }
    
    // Mark training as completed and collect test results
    let final_accuracy = {
        let (accuracy, _, _) = network.evaluate(test_inputs, test_labels);
        accuracy
    };
    
    // Collect detailed test results for storage
    let mut test_results = Vec::new();
    let mut total_correct = 0;
    
    for (i, (input, &actual_label)) in test_inputs.iter().zip(test_labels.iter()).enumerate() {
        let output = network.forward(input);
        
        // Find predicted class (index of maximum value)
        let predicted_class = output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap_or(0) as u8;
            
        let confidence = output[predicted_class as usize];
        let is_correct = predicted_class == actual_label;
        
        if is_correct {
            total_correct += 1;
        }
        
        test_results.push(TestResult {
            actual: actual_label,
            predicted: predicted_class,
            confidence,
            prediction_scores: output,
            is_correct,
            image: test_images.images[i].clone(),
        });
    }
    
    let test_data = TestData {
        results: test_results,
        accuracy: final_accuracy,
        total_correct,
        total_samples: test_inputs.len(),
        misclassified_count: test_inputs.len() - total_correct,
    };
    
    // Store test data
    {
        let mut state = test_data_state.lock().unwrap();
        *state = Some(test_data.clone());
    }
    
    // Save test data to disk
    if let Err(e) = save_test_data_to_file(&test_data) {
        println!("Failed to save test data to disk: {}", e);
    }

    {
        let mut state = training_state.lock().unwrap();
        state.status = "completed".to_string();
        state.progress = 100.0;
        state.current_accuracy = final_accuracy;
        state.elapsed_time = training_start.elapsed().as_secs_f64();
        state.message = format!("Training completed! Final accuracy: {:.2}%", final_accuracy * 100.0);
    }
}

// HTTP utilities
fn send_json_response(stream: &mut TcpStream, status: &str, body: &str) -> std::io::Result<()> {
    let response = format!(
        "HTTP/1.1 {}\r\n\
         Content-Type: application/json\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Methods: GET, POST, DELETE\r\n\
         Access-Control-Allow-Headers: Content-Type\r\n\
         Content-Length: {}\r\n\
         \r\n\
         {}",
        status,
        body.len(),
        body
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()
}

fn parse_json_body(request: &str) -> Option<String> {
    if let Some(body_start) = request.find("\r\n\r\n") {
        let body = &request[body_start + 4..];
        if !body.is_empty() {
            return Some(body.to_string());
        }
    }
    None
}

// HTTP handlers
fn handle_start_training(
    body: Option<String>,
    training_state: TrainingState,
    model_state: ModelState,
    test_data_state: TestDataState,
) -> std::result::Result<String, String> {
    // Parse request
    let req: TrainingRequest = if let Some(body) = body {
        serde_json::from_str(&body).map_err(|e| format!("Invalid JSON: {}", e))?
    } else {
        TrainingRequest::default()
    };
    
    // Check if training is already running
    {
        let state = training_state.lock().unwrap();
        if state.status == "running" {
            let response = TrainingResponse {
                status: "error".to_string(),
                message: "Training is already in progress".to_string(),
            };
            return Ok(serde_json::to_string(&response).unwrap());
        }
    }
    
    // Reset training state
    {
        let mut state = training_state.lock().unwrap();
        *state = TrainingStatus::default();
        state.status = "preparing".to_string();
        state.message = "Loading dataset...".to_string();
    }
    
    // Spawn training task
    let training_state_clone = training_state.clone();
    let model_state_clone = model_state.clone();
    thread::spawn(move || {
        // Load datasets
        let train_images = match MnistData::load_images("../train-images.idx3-ubyte") {
            Ok(data) => data,
            Err(e) => {
                let mut state = training_state_clone.lock().unwrap();
                state.status = "failed".to_string();
                state.message = format!("Failed to load training images: {}", e);
                return;
            }
        };
        
        let train_labels = match MnistData::load_labels("../train-labels.idx1-ubyte") {
            Ok(data) => data,
            Err(e) => {
                let mut state = training_state_clone.lock().unwrap();
                state.status = "failed".to_string();
                state.message = format!("Failed to load training labels: {}", e);
                return;
            }
        };
        
        let test_images = match MnistData::load_images("../t10k-images.idx3-ubyte") {
            Ok(data) => data,
            Err(e) => {
                let mut state = training_state_clone.lock().unwrap();
                state.status = "failed".to_string();
                state.message = format!("Failed to load test images: {}", e);
                return;
            }
        };
        
        let test_labels = match MnistData::load_labels("../t10k-labels.idx1-ubyte") {
            Ok(data) => data,
            Err(e) => {
                let mut state = training_state_clone.lock().unwrap();
                state.status = "failed".to_string();
                state.message = format!("Failed to load test labels: {}", e);
                return;
            }
        };
        
        // Prepare test data
        let mut test_inputs = Vec::new();
        for i in 0..test_images.count {
            test_inputs.push(normalize_image(&test_images.images[i]));
        }
        
        // Create neural network
        let mut network = NeuralNetwork::new(&[784, 256, 128, 10]);
        
        // Update status
        {
            let mut state = training_state_clone.lock().unwrap();
            state.message = "Starting training...".to_string();
        }
        
        // Start training
        train_neural_network(
            &mut network,
            &train_images,
            &train_labels,
            &test_images,
            &test_inputs,
            &test_labels,
            req.epochs.unwrap_or(10),
            req.batch_size.unwrap_or(32),
            req.learning_rate.unwrap_or(0.01),
            req.momentum.unwrap_or(0.9),
            training_state_clone,
            test_data_state.clone(),
        );

        {
            let mut state = training_state.lock().unwrap();
            if state.status == "cancelled" {
                state.status = "completed".to_string();
                state.message = "Training cancelled".to_string();
                return;
            }
        }
        
        // Store the trained model
        {
            let mut model = model_state_clone.lock().unwrap();
            *model = Some(network.clone());
        }
        
        // Save the trained model to disk
        if let Err(e) = save_model_to_file(&network) {
            println!("Failed to save model to disk: {}", e);
        }
    });
    
    let response = TrainingResponse {
        status: "preparing".to_string(),
        message: "Training started".to_string(),
    };
    
    Ok(serde_json::to_string(&response).unwrap())
}

fn handle_get_status(training_state: TrainingState) -> std::result::Result<String, String> {
    let state = training_state.lock().unwrap();
    Ok(serde_json::to_string(&*state).unwrap())
}

fn handle_cancel_training(training_state: TrainingState) -> std::result::Result<String, String> {
    let mut state = training_state.lock().unwrap();
    
    if state.status == "running" || state.status == "preparing" {
        state.status = "cancelled".to_string();
        state.message = "Training cancelled by user".to_string();
        Ok(serde_json::to_string(&serde_json::json!({
            "success": true,
            "message": "Training cancelled"
        })).unwrap())
    } else {
        Ok(serde_json::to_string(&serde_json::json!({
            "success": false,
            "message": format!("Cannot cancel training with status: {}", state.status)
        })).unwrap())
    }
}

fn handle_get_model(model_state: ModelState) -> std::result::Result<String, String> {
    let model = model_state.lock().unwrap();
    
    if let Some(ref network) = *model {
        Ok(serde_json::to_string(network).unwrap())
    } else {
        Ok(serde_json::to_string(&serde_json::json!({
            "error": "No trained model available",
            "message": "Train a model first using POST /api/train"
        })).unwrap())
    }
}

fn handle_get_test_data(test_data_state: TestDataState) -> std::result::Result<String, String> {
    let test_data = test_data_state.lock().unwrap();
    
    match test_data.as_ref() {
        Some(data) => {
            serde_json::to_string(data).map_err(|e| format!("Failed to serialize test data: {}", e))
        }
        None => {
            Ok(r#"{"error": "No test data available. Train a model first."}"#.to_string())
        }
    }
}

fn handle_request(
    request: &str,
    training_state: TrainingState,
    model_state: ModelState,
    test_data_state: TestDataState,
) -> (String, String) {
    let lines: Vec<&str> = request.lines().collect();
    if lines.is_empty() {
        return ("400 Bad Request".to_string(), r#"{"error": "Invalid request"}"#.to_string());
    }
    
    let request_line = lines[0];
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return ("400 Bad Request".to_string(), r#"{"error": "Invalid request line"}"#.to_string());
    }
    
    let method = parts[0];
    let path = parts[1];
    
    // Handle CORS preflight
    if method == "OPTIONS" {
        return ("200 OK".to_string(), "".to_string());
    }
    
    match (method, path) {
        ("POST", "/api/train") => {
            let body = parse_json_body(request);
            match handle_start_training(body, training_state, model_state, test_data_state) {
                Ok(response) => ("200 OK".to_string(), response),
                Err(error) => ("400 Bad Request".to_string(), format!(r#"{{"error": "{}"}}"#, error)),
            }
        }
        ("GET", "/api/status") => {
            match handle_get_status(training_state) {
                Ok(response) => ("200 OK".to_string(), response),
                Err(error) => ("500 Internal Server Error".to_string(), format!(r#"{{"error": "{}"}}"#, error)),
            }
        }
        ("DELETE", "/api/training") => {
            match handle_cancel_training(training_state) {
                Ok(response) => ("200 OK".to_string(), response),
                Err(error) => ("500 Internal Server Error".to_string(), format!(r#"{{"error": "{}"}}"#, error)),
            }
        }
        ("GET", "/api/model") => {
            match handle_get_model(model_state) {
                Ok(response) => ("200 OK".to_string(), response),
                Err(error) => ("500 Internal Server Error".to_string(), format!(r#"{{"error": "{}"}}"#, error)),
            }
        }
        ("GET", "/api/test-data") => {
            match handle_get_test_data(test_data_state) {
                Ok(response) => ("200 OK".to_string(), response),
                Err(error) => ("500 Internal Server Error".to_string(), format!(r#"{{"error": "{}"}}"#, error)),
            }
        }
        ("POST", "/api/continue-training") => {
            let body = parse_json_body(request);
            match handle_continue_training(body, training_state, model_state, test_data_state) {
                Ok(response) => ("200 OK".to_string(), response),
                Err(error) => ("400 Bad Request".to_string(), format!(r#"{{"error": "{}"}}"#, error)),
            }
        }
        ("DELETE", "/api/delete") => {
            match handle_delete_model(model_state, test_data_state) {
                Ok(response) => ("200 OK".to_string(), response),
                Err(error) => ("500 Internal Server Error".to_string(), format!(r#"{{"error": "{}"}}"#, error)),
            }
        }
        _ => ("404 Not Found".to_string(), r#"{"error": "Not found"}"#.to_string()),
    }
}

fn handle_client(mut stream: TcpStream, training_state: TrainingState, model_state: ModelState, test_data_state: TestDataState) {
    let mut buffer = [0; 4096];
    match stream.read(&mut buffer) {
        Ok(size) => {
            let request = String::from_utf8_lossy(&buffer[..size]);
            let (status, body) = handle_request(&request, training_state, model_state, test_data_state);
            
            let _ = send_json_response(&mut stream, &status, &body);
        }
        Err(e) => {
            eprintln!("Failed to read request: {}", e);
        }
    }
}

fn main() -> std::io::Result<()> {
    println!("MNIST Neural Network Training API Server");
    println!("============================================");
    
    // Initialize shared state
    let training_state: TrainingState = Arc::new(Mutex::new(TrainingStatus::default()));
    
    // Try to load existing model from file
    let initial_model = match load_model_from_file() {
        Ok(Some(model)) => {
            println!("Loaded existing model from disk");
            Some(model)
        }
        Ok(None) => {
            println!("No existing model found");
            None
        }
        Err(e) => {
            eprintln!("Error loading model: {}", e);
            None
        }
    };
    let model_state: ModelState = Arc::new(Mutex::new(initial_model));
    
    // Try to load existing test data from file
    let initial_test_data = match load_test_data_from_file() {
        Ok(Some(test_data)) => {
            println!("Loaded existing test data from disk");
            Some(test_data)
        }
        Ok(None) => {
            println!("No existing test data found");
            None
        }
        Err(e) => {
            eprintln!("Error loading test data: {}", e);
            None
        }
    };
    let test_data_state: TestDataState = Arc::new(Mutex::new(initial_test_data));
    
    // Load model and test data from files if they exist
    {
        let mut model = model_state.lock().unwrap();
        let loaded_model = load_model_from_file().unwrap_or(None);
        *model = loaded_model;
    }
    
    {
        let mut test_data = test_data_state.lock().unwrap();
        let loaded_test_data = load_test_data_from_file().unwrap_or(None);
        *test_data = loaded_test_data;
    }
    
    let listener = TcpListener::bind("0.0.0.0:3030")?;
    println!("Server starting on http://localhost:3030");
    println!("\nAvailable endpoints:");
    println!("  POST   /api/train            - Start training");
    println!("  POST   /api/continue-training - Continue training existing model");
    println!("  GET    /api/status           - Get training status");
    println!("  DELETE /api/training         - Cancel training");
    println!("  GET    /api/model            - Get trained model");
    println!("  DELETE /api/delete           - Delete trained model");
    println!("  GET    /api/test-data        - Get test results");
    println!("\nServer ready!");
    
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let training_state = training_state.clone();
                let model_state = model_state.clone();
                let test_data_state = test_data_state.clone();
                thread::spawn(move || {
                    handle_client(stream, training_state, model_state, test_data_state);
                });
            }
            Err(e) => {
                eprintln!("Failed to accept connection: {}", e);
            }
        }
    }
    
    Ok(())
}

fn handle_continue_training(
    body: Option<String>,
    training_state: TrainingState,
    model_state: ModelState,
    test_data_state: TestDataState,
) -> std::result::Result<String, String> {
    // Parse request
    let req: TrainingRequest = if let Some(body) = body {
        serde_json::from_str(&body).map_err(|e| format!("Invalid JSON: {}", e))?
    } else {
        TrainingRequest::default()
    };
    
    // Check if training is already running
    {
        let state = training_state.lock().unwrap();
        if state.status == "running" {
            let response = TrainingResponse {
                status: "error".to_string(),
                message: "Training is already in progress".to_string(),
            };
            return Ok(serde_json::to_string(&response).unwrap());
        }
    }
    
    // Check if model exists
    let existing_model = {
        let model = model_state.lock().unwrap();
        match model.as_ref() {
            Some(network) => network.clone(),
            None => {
                let response = TrainingResponse {
                    status: "error".to_string(),
                    message: "No existing model found. Please train a model first.".to_string(),
                };
                return Ok(serde_json::to_string(&response).unwrap());
            }
        }
    };
    
    // Reset training state
    {
        let mut state = training_state.lock().unwrap();
        *state = TrainingStatus::default();
        state.status = "preparing".to_string();
        state.message = "Loading dataset for continued training...".to_string();
    }
    
    // Spawn continue training task
    let training_state_clone = training_state.clone();
    let model_state_clone = model_state.clone();
    thread::spawn(move || {
        // Load datasets
        let train_images = match MnistData::load_images("../train-images.idx3-ubyte") {
            Ok(data) => data,
            Err(e) => {
                let mut state = training_state_clone.lock().unwrap();
                state.status = "failed".to_string();
                state.message = format!("Failed to load training images: {}", e);
                return;
            }
        };
        
        let train_labels = match MnistData::load_labels("../train-labels.idx1-ubyte") {
            Ok(data) => data,
            Err(e) => {
                let mut state = training_state_clone.lock().unwrap();
                state.status = "failed".to_string();
                state.message = format!("Failed to load training labels: {}", e);
                return;
            }
        };
        
        let test_images = match MnistData::load_images("../t10k-images.idx3-ubyte") {
            Ok(data) => data,
            Err(e) => {
                let mut state = training_state_clone.lock().unwrap();
                state.status = "failed".to_string();
                state.message = format!("Failed to load test images: {}", e);
                return;
            }
        };
        
        let test_labels = match MnistData::load_labels("../t10k-labels.idx1-ubyte") {
            Ok(data) => data,
            Err(e) => {
                let mut state = training_state_clone.lock().unwrap();
                state.status = "failed".to_string();
                state.message = format!("Failed to load test labels: {}", e);
                return;
            }
        };
        
        // Prepare test data
        let mut test_inputs = Vec::new();
        for i in 0..test_images.count {
            test_inputs.push(normalize_image(&test_images.images[i]));
        }
        
        // Use existing network for continued training
        let mut network = existing_model;
        
        // Update status
        {
            let mut state = training_state_clone.lock().unwrap();
            state.message = "Continuing training with existing model...".to_string();
        }
        
        // Continue training with existing network
        train_neural_network(
            &mut network,
            &train_images,
            &train_labels,
            &test_images,
            &test_inputs,
            &test_labels,
            req.epochs.unwrap_or(5), // Default fewer epochs for continued training
            req.batch_size.unwrap_or(32),
            req.learning_rate.unwrap_or(0.005), // Lower learning rate for fine-tuning
            req.momentum.unwrap_or(0.9),
            training_state_clone,
            test_data_state.clone(),
        );

        {
            let mut state = training_state.lock().unwrap();
            if state.status == "cancelled" {
                state.status = "completed".to_string();
                state.message = "Training cancelled".to_string();
                return;
            }
        }
        
        // Store the updated model
        {
            let mut model = model_state_clone.lock().unwrap();
            *model = Some(network.clone());
        }
        
        // Save the updated model to disk
        if let Err(e) = save_model_to_file(&network) {
            println!("Failed to save updated model to disk: {}", e);
        }
    });
    
    let response = TrainingResponse {
        status: "success".to_string(),
        message: "Continue training started".to_string(),
    };
    
    Ok(serde_json::to_string(&response).unwrap())
}

fn handle_delete_model(model_state: ModelState, test_data_state: TestDataState) -> std::result::Result<String, String> {
    // Clear the model
    {
        let mut model = model_state.lock().unwrap();
        *model = None;
    }
    
    // Clear test data
    {
        let mut test_data = test_data_state.lock().unwrap();
        *test_data = None;
    }
    
    // Delete model and test data files
    if let Err(e) = delete_model_files() {
        return Err(format!("Failed to delete model files: {}", e));
    }
    
    let response = serde_json::json!({
        "status": "success",
        "message": "Model and test data deleted successfully"
    });
    
    Ok(response.to_string())
}

// Model persistence functions
const MODEL_FILE_PATH: &str = "trained_model.json";
const TEST_DATA_FILE_PATH: &str = "test_data.json";

fn save_model_to_file(network: &NeuralNetwork) -> std::result::Result<(), String> {
    let json = serde_json::to_string_pretty(network)
        .map_err(|e| format!("Failed to serialize model: {}", e))?;
    
    std::fs::write(MODEL_FILE_PATH, json)
        .map_err(|e| format!("Failed to write model file: {}", e))?;
    
    println!("Model saved to {}", MODEL_FILE_PATH);
    Ok(())
}

fn load_model_from_file() -> std::result::Result<Option<NeuralNetwork>, String> {
    if !std::path::Path::new(MODEL_FILE_PATH).exists() {
        return Ok(None);
    }
    
    let json = std::fs::read_to_string(MODEL_FILE_PATH)
        .map_err(|e| format!("Failed to read model file: {}", e))?;
    
    let network: NeuralNetwork = serde_json::from_str(&json)
        .map_err(|e| format!("Failed to deserialize model: {}", e))?;
    
    println!("Model loaded from {}", MODEL_FILE_PATH);
    Ok(Some(network))
}

fn save_test_data_to_file(test_data: &TestData) -> std::result::Result<(), String> {
    let json = serde_json::to_string_pretty(test_data)
        .map_err(|e| format!("Failed to serialize test data: {}", e))?;
    
    std::fs::write(TEST_DATA_FILE_PATH, json)
        .map_err(|e| format!("Failed to write test data file: {}", e))?;
    
    println!("Test data saved to {}", TEST_DATA_FILE_PATH);
    Ok(())
}

fn load_test_data_from_file() -> std::result::Result<Option<TestData>, String> {
    if !std::path::Path::new(TEST_DATA_FILE_PATH).exists() {
        return Ok(None);
    }
    
    let json = std::fs::read_to_string(TEST_DATA_FILE_PATH)
        .map_err(|e| format!("Failed to read test data file: {}", e))?;
    
    let test_data: TestData = serde_json::from_str(&json)
        .map_err(|e| format!("Failed to deserialize test data: {}", e))?;
    
    println!("Test data loaded from {}", TEST_DATA_FILE_PATH);
    Ok(Some(test_data))
}

fn delete_model_files() -> std::result::Result<(), String> {
    if std::path::Path::new(MODEL_FILE_PATH).exists() {
        std::fs::remove_file(MODEL_FILE_PATH)
            .map_err(|e| format!("Failed to delete model file: {}", e))?;
        println!("Model file deleted");
    }
    
    if std::path::Path::new(TEST_DATA_FILE_PATH).exists() {
        std::fs::remove_file(TEST_DATA_FILE_PATH)
            .map_err(|e| format!("Failed to delete test data file: {}", e))?;
        println!("Test data file deleted");
    }
    
    Ok(())
}
