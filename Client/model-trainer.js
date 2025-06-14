// DOM Elements
const trainButton = document.getElementById('train-button');
const continueButton = document.getElementById('continue-button');
const deleteButton = document.getElementById('delete-button');
const cancelButton = document.getElementById('cancel-button');
const showRandomButton = document.getElementById('show-random-button');
const showMisclassifiedButton = document.getElementById('show-misclassified-button');
const trainingStatus = document.getElementById('training-status');
const trainingProgress = document.getElementById('training-progress');
const accuracyValue = document.getElementById('accuracy-value');
const correctCount = document.getElementById('correct-count');
const wrongCount = document.getElementById('wrong-count');
const exampleCanvas = document.getElementById('example-canvas');
const actualLabel = document.getElementById('actual-label');
const predictedLabel = document.getElementById('predicted-label');
const confidenceValue = document.getElementById('confidence-value');
const confidenceBarsContainer = document.getElementById('confidence-bars-container');

// Configuration inputs
const epochsInput = document.getElementById('epochs');
const batchSizeInput = document.getElementById('batch-size');
const learningRateInput = document.getElementById('learning-rate');

// Global variables
let neuralNetwork = null;
let testResults = [];
let misclassifiedExamples = [];
let trainingPollingInterval = null;
let serverCheckInterval = null;
let serverConnected = false;
let hasExistingModel = false;

// API configuration
let API_BASE_URL = 'http://localhost:3030/api';

// Function to update API base URL when server IP changes
window.updateAPIBaseURL = function(newURL) {
    API_BASE_URL = newURL;
    console.log('API Base URL updated to:', API_BASE_URL);
};

// Function to reconnect to server when IP changes
window.reconnectToServer = async function() {
    console.log('Reconnecting to server...');
    serverConnected = false;
    hasExistingModel = false;
    
    // Stop any ongoing polling
    stopTrainingStatusPolling();
    stopServerConnectivityCheck();
    
    // Update server status
    if (window.serverConfig) {
        window.serverConfig.updateStatus('connecting', 'Connecting...');
    }
    
    // Clear displays
    clearDisplays();
    
    // Restart connectivity check
    await checkModelAvailability();
    startServerConnectivityCheck();
};

// Initialize the page
async function init() {
    // Setup event listeners
    trainButton.addEventListener('click', startTraining);
    continueButton.addEventListener('click', continueTraining);
    deleteButton.addEventListener('click', deleteModel);
    cancelButton.addEventListener('click', cancelTraining);
    showRandomButton.addEventListener('click', showRandomExample);
    showMisclassifiedButton.addEventListener('click', showMisclassifiedExample);

    // Create an empty confidence bar display
    createConfidenceBars();
    
    // Display empty canvas
    const ctx = exampleCanvas.getContext('2d');
    ctx.fillStyle = '#2d2d2d';
    ctx.fillRect(0, 0, exampleCanvas.width, exampleCanvas.height);

    // Wait for server config to be ready
	if (window.serverConfig) {
		API_BASE_URL = window.serverConfig.getAPIBaseURL();
	}
    
    // Check if we have a trained model available
    await checkModelAvailability();

    // Start periodic server connectivity check
    startServerConnectivityCheck();
}

// Check if a trained model is available from the API
async function checkModelAvailability() {
    try {
        // First check if training is already in progress
        const statusResponse = await fetch(`${API_BASE_URL}/status`);
        if (statusResponse.ok) {
            const statusData = await statusResponse.json();
            
            // If training is in progress, start polling and update UI
            if (statusData.status === 'running') {
                serverConnected = true;
                updateServerStatus('connected');
                trainingStatus.textContent = `Training in progress: ${statusData.message}`;
                updateButtonStates(true); // Training in progress
                trainingProgress.value = statusData.progress;
                startTrainingStatusPolling();
                return; // Don't check for model since training is active
            }
        }
        
        // Check if a trained model is available
        const response = await fetch(`${API_BASE_URL}/model`);
        const data = await response.json();

        if (response.ok && !data.error) {
            // Model is available
            neuralNetwork = data;
            hasExistingModel = true;
            serverConnected = true;
            updateServerStatus('connected');
            trainingStatus.textContent = 'Model loaded! You can continue training or test it.';
            updateButtonStates(false); // Not training
            enableTestingButtons();
            await loadTestDataAndTest();
        } else {
            hasExistingModel = false;
            serverConnected = true;
            updateServerStatus('connected');
            trainingStatus.textContent = 'No trained model available. Click "Train New Model" to start.';
            updateButtonStates(false); // Not training
        }
    } catch (error) {
        hasExistingModel = false;
        serverConnected = false;
        trainingStatus.textContent = 'Server not available. Checking connection every 10 seconds...';
        updateButtonStates(false); // Not training, server disconnected
        console.error('Error checking model availability:', error);
    }
}

// Start training via API
async function startTraining() {
    try {
        // Check server connectivity first
        if (!serverConnected) {
            trainingStatus.textContent = 'Server not connected. Please wait for connection...';
            return;
        }

        // Update UI
        updateButtonStates(true); // Training started
        trainingStatus.textContent = 'Starting new training...';
        trainingProgress.value = 0;

        // Get training parameters
        const epochs = parseInt(epochsInput.value) || 5;
        const batchSize = parseInt(batchSizeInput.value) || 100;
        const learningRate = parseFloat(learningRateInput.value) || 0.01;

        // Start training via API
        const response = await fetch(`${API_BASE_URL}/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                epochs: epochs,
                batch_size: batchSize,
                learning_rate: learningRate,
                momentum: 0.9
            })
        });

        const data = await response.json();

        if (!response.ok || data.status === 'error') {
            if (data.message === 'Training is already in progress') {
                trainingStatus.textContent = 'Training is already in progress.';
                startTrainingStatusPolling();
                return;
            }
            throw new Error(data.message || 'Failed to start training');
        }

        trainingStatus.textContent = 'Training started on server...';
        // Start polling for training status
        startTrainingStatusPolling();
    } catch (error) {
        trainingStatus.textContent = `Error: ${error.message}`;
        updateButtonStates(false); // Reset to not training state
        console.error('Training error:', error);
    }
}

// Continue training with existing model
async function continueTraining() {
    try {
        // Check server connectivity first
        if (!serverConnected) {
            trainingStatus.textContent = 'Server not connected. Please wait for connection...';
            return;
        }

        if (!hasExistingModel) {
            trainingStatus.textContent = 'No existing model found to continue training.';
            return;
        }

        // Update UI
        updateButtonStates(true); // Training started
        trainingStatus.textContent = 'Continuing training with existing model...';
        trainingProgress.value = 0;

        // Get training parameters (use different defaults for continue training)
        const epochs = parseInt(epochsInput.value) || 3; // Fewer epochs for continue
        const batchSize = parseInt(batchSizeInput.value) || 100;
        const learningRate = parseFloat(learningRateInput.value) || 0.005; // Lower learning rate

        // Start continue training via API
        const response = await fetch(`${API_BASE_URL}/continue-training`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                epochs: epochs,
                batch_size: batchSize,
                learning_rate: learningRate,
                momentum: 0.9
            })
        });

        const data = await response.json();

        if (!response.ok || data.status === 'error') {
            if (data.message === 'Training is already in progress') {
                trainingStatus.textContent = 'Training is already in progress.';
                startTrainingStatusPolling();
                return;
            }
            throw new Error(data.message || 'Failed to start continue training');
        }

        trainingStatus.textContent = 'Continue training started on server...';

        // Start polling for training status
        startTrainingStatusPolling();
    } catch (error) {
        trainingStatus.textContent = `Error: ${error.message}`;
        updateButtonStates(false); // Reset to not training state
        console.error('Continue training error:', error);
    }
}

// Delete the current model
async function deleteModel() {
    try {
        // Check server connectivity first
        if (!serverConnected) {
            trainingStatus.textContent = 'Server not connected. Please wait for connection...';
            return;
        }

        if (!hasExistingModel) {
            trainingStatus.textContent = 'No model to delete.';
            return;
        }

        // Confirm deletion
        if (!confirm('Are you sure you want to delete the current model? This action cannot be undone.')) {
            return;
        }

        trainingStatus.textContent = 'Deleting model...';
        updateButtonStates(false); // Disable buttons during deletion

        // Delete model via API
        const response = await fetch(`${API_BASE_URL}/delete`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (!response.ok || data.status === 'error') {
            throw new Error(data.message || 'Failed to delete model');
        }

        // Update state
        hasExistingModel = false;
        neuralNetwork = null;
        testResults = [];
        misclassifiedExamples = [];

        trainingStatus.textContent = 'Model deleted successfully. You can now train a new model.';
        updateButtonStates(false); // Update button states
        disableTestingButtons();
        clearDisplays();

    } catch (error) {
        trainingStatus.textContent = `Error deleting model: ${error.message}`;
        updateButtonStates(false); // Reset button states
        console.error('Delete model error:', error);
    }
}

// Poll training status from the API
function startTrainingStatusPolling() {
    trainingPollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/status`);
            const status = await response.json();

            if (!response.ok) {
                throw new Error('Failed to get training status');
            }

            // Update progress
            trainingProgress.value = status.progress;

            // Update status message
            if (status.status === 'running') {
                trainingStatus.textContent = `${status.message} - Accuracy: ${(status.current_accuracy * 100).toFixed(2)}% - Loss: ${status.current_loss.toFixed(4)}`;
            } else {
                trainingStatus.textContent = status.message;
            }            // Check if training is complete
            if (status.status === 'completed') {
                stopTrainingStatusPolling();
                trainingStatus.textContent = `Training complete! ${status.message}`;
                hasExistingModel = true; // Model is now available
                updateButtonStates(false); // Training finished

                // Load the trained model and test it
                await loadTrainedModelAndTest();
            } else if (status.status === 'failed' || status.status === 'cancelled') {
                stopTrainingStatusPolling();
                trainingStatus.textContent = `Training ${status.status}: ${status.message}`;
                
                // Check if we still have a model available after failure/cancellation
                await checkModelAvailabilityAfterCancel();
            }
        } catch (error) {
            console.error('Error polling training status:', error);
            stopTrainingStatusPolling();

            // Check if this was a connection error
            if (error.message.includes('fetch') || error.message.includes('network')) {
                serverConnected = false;
                updateServerStatus('disconnected');
            } else {
                trainingStatus.textContent = 'Error getting training status';
                trainButton.disabled = false;
                cancelButton.disabled = true;
            }
        }
    }, 1000); // Poll every second
}

// Stop polling training status
function stopTrainingStatusPolling() {
    if (trainingPollingInterval) {
        clearInterval(trainingPollingInterval);
        trainingPollingInterval = null;
    }
}

// Load the trained model from API and run tests
async function loadTrainedModelAndTest() {
    try {
        // Get the trained model from API
        const response = await fetch(`${API_BASE_URL}/model`);
        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.message || 'Failed to load trained model');
        }

        neuralNetwork = data;
        trainingStatus.textContent = 'Model loaded successfully!';

        // Enable testing buttons
        enableTestingButtons();

        // Load test data and run tests
        await loadTestDataAndTest();

    } catch (error) {
        trainingStatus.textContent = `Error loading model: ${error.message}`;
        console.error('Error loading trained model:', error);
    }
}

// Fetch test data from server
async function fetchTestDataFromServer() {
    try {
        const response = await fetch(`${API_BASE_URL}/test-data`);

        if (!response.ok) {
            console.log('Server test data not available:', response.status);
            return null;
        }

        const data = await response.json();

        if (data.error) {
            console.log('Server test data error:', data.error);
            return null;
        }

        return data;

    } catch (error) {
        console.log('Error fetching test data from server:', error);
        return null;
    }
}

// Load test dataset and run tests
async function loadTestDataAndTest() {
    try {
        trainingStatus.textContent = 'Loading test results from server...';

        // Fetch test results from server
        const testDataFromServer = await fetchTestDataFromServer();

        if (testDataFromServer) {
            // Use server test results
            testResults = testDataFromServer.results.map(result => ({
                actual: result.actual,
                predicted: result.predicted,
                correct: result.is_correct,
                confidence: result.confidence,
                prediction: result.prediction_scores,
                image: result.image // Image data from server
            }));

            // Extract misclassified examples
            misclassifiedExamples = testResults.filter(result => !result.correct);

            // Update UI with results
            const accuracy = testDataFromServer.accuracy;
            accuracyValue.textContent = `${(accuracy * 100).toFixed(2)}%`;
            correctCount.textContent = testDataFromServer.total_correct;
            wrongCount.textContent = testDataFromServer.misclassified_count;

            trainingStatus.textContent = `Test results loaded! Accuracy: ${(accuracy * 100).toFixed(2)}%`;
            trainingProgress.value = 100;

            // Enable buttons
            showRandomButton.disabled = false;
            showMisclassifiedButton.disabled = misclassifiedExamples.length === 0;

            // Show a random example
            showRandomExample();
        } else {
            // No server data available
            trainingStatus.textContent = 'No test data available. Please train a model first.';
            showRandomButton.disabled = true;
            showMisclassifiedButton.disabled = true;
        }

    } catch (error) {
        trainingStatus.textContent = `Error loading test data: ${error.message}`;
        console.error('Error loading test data:', error);
    }
}

// Enable testing buttons
function enableTestingButtons() {
    showRandomButton.disabled = false;
    showMisclassifiedButton.disabled = false;
}

// Simulate forward pass for model loaded from API
function simulateForward(model, input) {
    let activations = input;

    // Forward pass through each layer
    for (let i = 0; i < model.layers.length; i++) {
        const layer = model.layers[i];
        const newActivations = new Array(layer.output_size);

        // Compute weighted sum + bias for each neuron
        for (let j = 0; j < layer.output_size; j++) {
            let sum = layer.biases[j];
            for (let k = 0; k < layer.input_size; k++) {
                sum += activations[k] * layer.weights[j][k];
            }

            // Apply activation function
            if (i === model.layers.length - 1) {
                // Output layer - sigmoid
                newActivations[j] = 1.0 / (1.0 + Math.exp(-sum));
            } else {
                // Hidden layer - ReLU
                newActivations[j] = Math.max(0, sum);
            }
        }

        activations = newActivations;
    }

    return activations;
}

// Display a random example from the test dataset
function showRandomExample() {
    if (!testResults.length) {
        trainingStatus.textContent = 'No test results available';
        return;
    }

    const randomIndex = Math.floor(Math.random() * testResults.length);
    displayExample(testResults[randomIndex]);
}

// Display a randomly selected misclassified example
function showMisclassifiedExample() {
    if (!misclassifiedExamples.length) {
        trainingStatus.textContent = 'No misclassified examples available';
        return;
    }

    const randomIndex = Math.floor(Math.random() * misclassifiedExamples.length);
    displayExample(misclassifiedExamples[randomIndex]);
}

// Display an example on the canvas
function displayExample(example) {
    const ctx = exampleCanvas.getContext('2d');
    ctx.fillStyle = '#2d2d2d';
    ctx.fillRect(0, 0, exampleCanvas.width, exampleCanvas.height);

    // Create a temporary 28x28 canvas for the image data
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    // Convert grayscale image data to RGBA format (4 bytes per pixel)
    const rgba = new Uint8ClampedArray(28 * 28 * 4);
    for (let i = 0; i < 28 * 28; i++) {
        const value = example.image[i]; // Direct value, no inversion needed
        rgba[i * 4] = value;     // R
        rgba[i * 4 + 1] = value; // G
        rgba[i * 4 + 2] = value; // B
        rgba[i * 4 + 3] = 255;   // A (fully opaque)
    }
    const imageData = new ImageData(rgba, 28, 28);

    // Put the image data on the temporary canvas
    tempCtx.putImageData(imageData, 0, 0);

    // Scale the image to fill the entire canvas using drawImage
    ctx.imageSmoothingEnabled = false; // Pixelated look for better visibility
    ctx.drawImage(tempCanvas, 0, 0, exampleCanvas.width, exampleCanvas.height);

    // Update labels
    actualLabel.textContent = example.actual;
    predictedLabel.textContent = example.predicted;
    confidenceValue.textContent = `${(example.confidence * 100).toFixed(2)}%`;
    // Highlight if it's correct or not
    if (example.correct) {
        actualLabel.style.color = '#66bb6a';
        predictedLabel.style.color = '#66bb6a';
    } else {
        actualLabel.style.color = '#ff6b6b';
        predictedLabel.style.color = '#ff6b6b';
    }

    // Update confidence bars
    updateConfidenceBars(example.prediction);
}

// Create the confidence bar elements
function createConfidenceBars() {
    confidenceBarsContainer.innerHTML = '';

    for (let i = 0; i < 10; i++) {
        const barWrapper = document.createElement('div');
        barWrapper.className = 'confidence-bar-wrapper';

        const digitLabel = document.createElement('div');
        digitLabel.className = 'digit-label';
        digitLabel.textContent = i;

        const barOuter = document.createElement('div');
        barOuter.className = 'confidence-bar-outer';

        const barInner = document.createElement('div');
        barInner.className = 'confidence-bar-inner';
        barInner.style.width = '0%';
        barOuter.appendChild(barInner);

        const percentage = document.createElement('div');
        percentage.className = 'confidence-percentage';
        percentage.textContent = '0.00%';

        barWrapper.appendChild(digitLabel);
        barWrapper.appendChild(barOuter);
        barWrapper.appendChild(percentage);

        confidenceBarsContainer.appendChild(barWrapper);
    }
}

// Update confidence bars with prediction values
function updateConfidenceBars(prediction) {
    const bars = confidenceBarsContainer.querySelectorAll('.confidence-bar-wrapper');

    for (let i = 0; i < 10; i++) {
        const confidence = prediction[i] * 100;
        const barInner = bars[i].querySelector('.confidence-bar-inner');
        const percentage = bars[i].querySelector('.confidence-percentage');

        barInner.style.width = `${confidence}%`;
        percentage.textContent = `${confidence.toFixed(2)}%`;

        // Highlight the highest confidence
        if (prediction[i] === Math.max(...prediction)) {
            barInner.style.backgroundColor = '#4285f4';
        } else {
            barInner.style.backgroundColor = '#a0c3ff';
        }
    }
}

// Cancel training via API
async function cancelTraining() {
    try {
        const response = await fetch(`${API_BASE_URL}/training`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (response.ok && data.success) {
            stopTrainingStatusPolling();
            trainingStatus.textContent = 'Training cancelled by user';
            
            // Check if we still have a model available after cancellation
            await checkModelAvailabilityAfterCancel();
        } else {
            trainingStatus.textContent = `Failed to cancel: ${data.message}`;
            updateButtonStates(false); // Reset button states on failure
        }

    } catch (error) {
        trainingStatus.textContent = `Error cancelling training: ${error.message}`;
        updateButtonStates(false); // Reset button states on error
        console.error('Cancel training error:', error);
    }
}

// Check model availability after training cancellation
async function checkModelAvailabilityAfterCancel() {
    try {
        const response = await fetch(`${API_BASE_URL}/model`);
        const data = await response.json();

        if (response.ok && !data.error) {
            // Model is still available
            hasExistingModel = true;
            trainingStatus.textContent = 'Training cancelled. Previous model is still available for continued training.';
            enableTestingButtons();
            // Load and test the existing model
            await loadTestDataAndTest();
        } else {
            // No model available
            hasExistingModel = false;
            trainingStatus.textContent = 'Training cancelled. No trained model available.';
            disableTestingButtons();
        }
        
        updateButtonStates(false); // Update button states based on hasExistingModel
    } catch (error) {
        console.error('Error checking model after cancellation:', error);
        hasExistingModel = false;
        trainingStatus.textContent = 'Training cancelled. Error checking model availability.';
        updateButtonStates(false);
    }
}

// Check server connectivity
async function checkServerConnectivity() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

        const response = await fetch(`${API_BASE_URL}/status`);

        clearTimeout(timeoutId);

        if (response.ok) {
            if (!serverConnected) {
                serverConnected = true;
                updateServerStatus('connected');
                // If server just came back online, check for model availability and training status
                await checkModelAvailability();
            } else {
                // Server was already connected, check if training status has changed
                const statusData = await response.json();
                
                // If training is running but we're not polling, start polling
                if (statusData.status === 'running' && !trainingPollingInterval) {
                    trainingStatus.textContent = `Training in progress: ${statusData.message}`;
                    updateButtonStates(true); // Training in progress
                    trainingProgress.value = statusData.progress;
                    startTrainingStatusPolling();
                }
            }
            return true;
        } else {
            throw new Error(`Server responded with status ${response.status}`);
        }
    } catch (error) {
        if (serverConnected) {
            serverConnected = false;
            updateServerStatus('disconnected');
        }
        return false;
    }
}

// Update UI based on server connection status
function updateServerStatus(status) {
    // Update the server config status indicator
    if (window.serverConfig) {
        if (status === 'connected') {
            window.serverConfig.updateStatus('connected', 'Connected');
        } else {
            window.serverConfig.updateStatus('disconnected', 'Disconnected');
        }
    }
    
    if (status === 'connected') {
        trainingStatus.textContent = 'Server connected! Checking for trained model...';
        trainButton.disabled = false;
    } else {
        trainingStatus.textContent = 'Server not available. Checking connection every 10 seconds...';
        trainButton.disabled = true;
        cancelButton.disabled = true;
        showRandomButton.disabled = true;
        showMisclassifiedButton.disabled = true;
    }
}

// Update button states based on training status and model availability
function updateButtonStates(isTraining) {
    if (!serverConnected) {
        // Server disconnected - disable all buttons
        trainButton.disabled = true;
        continueButton.disabled = true;
        deleteButton.disabled = true;
        cancelButton.disabled = true;
        return;
    }

    if (isTraining) {
        // Training in progress
        trainButton.disabled = true;
        continueButton.disabled = true;
        deleteButton.disabled = true;
        cancelButton.disabled = false;
    } else {
        // Not training
        trainButton.disabled = false;
        continueButton.disabled = !hasExistingModel;
        deleteButton.disabled = !hasExistingModel;
        cancelButton.disabled = true;
    }
}

// Start periodic server connectivity check
function startServerConnectivityCheck() {
    // Check immediately
    checkServerConnectivity();

    // Then check every 10 seconds
    serverCheckInterval = setInterval(async () => {
        await checkServerConnectivity();
    }, 10000); // 10 seconds
}

// Stop server connectivity check
function stopServerConnectivityCheck() {
    if (serverCheckInterval) {
        clearInterval(serverCheckInterval);
        serverCheckInterval = null;
    }
}

// Cleanup function to stop all intervals
function cleanup() {
    stopTrainingStatusPolling();
    stopServerConnectivityCheck();
}

// Initialize the page when loaded
window.addEventListener('DOMContentLoaded', init);

// Cleanup when page is unloaded
window.addEventListener('beforeunload', cleanup);

// Clear all displays when model is deleted
function clearDisplays() {
    // Clear canvas
    const ctx = exampleCanvas.getContext('2d');
    ctx.fillStyle = '#3d3d3d';
    ctx.fillRect(0, 0, exampleCanvas.width, exampleCanvas.height);

    // Clear labels
    actualLabel.textContent = '-';
    predictedLabel.textContent = '-';
    confidenceValue.textContent = '-';

    // Clear metrics
    accuracyValue.textContent = '-';
    correctCount.textContent = '-';
    wrongCount.textContent = '-';

    // Clear confidence bars
    createConfidenceBars();
}

// Disable testing buttons
function disableTestingButtons() {
    showRandomButton.disabled = true;
    showMisclassifiedButton.disabled = true;
}
