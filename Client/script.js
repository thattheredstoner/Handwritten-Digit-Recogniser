const canvas = document.getElementById("drawing-canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clear-btn");
const predictionResult = document.getElementById("prediction-result");
const confidenceLevel = document.getElementById("confidence-level");

// Variables to track drawing state
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Set canvas size and scaling factor
const actualSize = 28;   // Actual pixel size for data
// Dynamic scaling factor based on canvas display size
let displaySize, scaleFactor;

// Neural network
let neuralNetwork = null;

// Polling variables
let modelPollingInterval = null;

// Set canvas size
function resizeCanvas() {
	canvas.width = actualSize;
	canvas.height = actualSize;

	// Calculate size to fill the container
	const container = canvas.parentElement;
	const containerRect = container.getBoundingClientRect();
	
	// Use the smaller dimension to maintain square aspect ratio and fill container
	const containerSize = Math.min(containerRect.width, containerRect.height);
	
	// Set canvas to fill the container
	const canvasSize = containerSize;
	
	// Set explicit pixel dimensions to fill container
	canvas.style.width = canvasSize + 'px';
	canvas.style.height = canvasSize + 'px';
	
	// Update scaling factor based on explicit size
	displaySize = canvasSize;
	scaleFactor = actualSize / displaySize;

	// Clear and set default style when resized
	ctx.lineJoin = "round";
	ctx.lineCap = "round";
	ctx.lineWidth = 1.5; // Reduced lineWidth for scaled drawing
	ctx.strokeStyle = "#ffffff"; // White stroke for dark background
	
	// Disable image smoothing for crisp pixel rendering
	ctx.imageSmoothingEnabled = false;
	ctx.mozImageSmoothingEnabled = false;
	ctx.webkitImageSmoothingEnabled = false;
	ctx.msImageSmoothingEnabled = false;
}

// Initialize canvas
resizeCanvas();
// Re-calculate scaling when window resizes
window.addEventListener("resize", resizeCanvas);

// Drawing event listeners
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseout", stopDrawing);

// Touch support for mobile devices
canvas.addEventListener("touchstart", handleTouch);
canvas.addEventListener("touchmove", handleTouchMove);
canvas.addEventListener("touchend", stopDrawing);

// Control event listeners
clearBtn.addEventListener("click", clearCanvas);

// Initialize the application when the page loads
window.addEventListener('DOMContentLoaded', init);

// Clean up polling when page is unloaded
window.addEventListener('beforeunload', stopModelPolling);

// Convert display coordinates to canvas coordinates
function scaleCoordinates(x, y) {
	const rect = canvas.getBoundingClientRect();
	const displayX = x - rect.left;
	const displayY = y - rect.top;
	
	// Scale the coordinates to match the actual canvas size
	const canvasX = displayX * scaleFactor;
	const canvasY = displayY * scaleFactor;
	
	return [canvasX, canvasY];
}

// Drawing functions
function startDrawing(e) {
	isDrawing = true;
	[lastX, lastY] = scaleCoordinates(e.clientX, e.clientY);
}

function draw(e) {
	if (!isDrawing) return;
	
	const [x, y] = scaleCoordinates(e.clientX, e.clientY);

	ctx.beginPath();
	ctx.moveTo(lastX, lastY);
	ctx.lineTo(x, y);
	ctx.stroke();

	[lastX, lastY] = [x, y];

	if (neuralNetwork) {
		predictDrawing();
	}
}

function stopDrawing() {
	isDrawing = false;
}

// Touch functions for mobile
function handleTouch(e) {
	e.preventDefault();
	const touch = e.touches[0];
	
	isDrawing = true;
	[lastX, lastY] = scaleCoordinates(touch.clientX, touch.clientY);
}

function handleTouchMove(e) {
	if (!isDrawing) return;
	e.preventDefault();

	const touch = e.touches[0];
	const [x, y] = scaleCoordinates(touch.clientX, touch.clientY);

	ctx.beginPath();
	ctx.moveTo(lastX, lastY);
	ctx.lineTo(x, y);
	ctx.stroke();

	[lastX, lastY] = [x, y];

	if (neuralNetwork) {
		predictDrawing();
	}
}

// Control functions
function clearCanvas() {
	// Store current canvas style dimensions before clearing
	const currentWidth = canvas.style.width;
	const currentHeight = canvas.style.height;
	
	// Clear canvas
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	
	// Immediately restore explicit dimensions to prevent CSS recalculation
	if (currentWidth && currentHeight) {
		canvas.style.width = currentWidth;
		canvas.style.height = currentHeight;
	}
	
	// Clear prediction results
	predictionResult.textContent = '';
	confidenceLevel.textContent = '';
}

// Convert canvas drawing to MNIST format (28x28 grayscale)
function canvasToMNIST(sourceCanvas) {
	const imageWidth = 28;
	const imageHeight = 28;
	
	// Create a temporary canvas to resize and convert to grayscale
	const tempCanvas = document.createElement('canvas');
	tempCanvas.width = imageWidth;
	tempCanvas.height = imageHeight;
	const tempCtx = tempCanvas.getContext('2d');
		// Fill with black background first (since we're drawing white on dark)
	tempCtx.fillStyle = 'black';
	tempCtx.fillRect(0, 0, imageWidth, imageHeight);
	
	// Draw the source canvas onto the temporary canvas, effectively resizing it
	tempCtx.drawImage(sourceCanvas, 0, 0, imageWidth, imageHeight);
	
	// Get the resized image data
	const tempData = tempCtx.getImageData(0, 0, imageWidth, imageHeight).data;
		// Convert to grayscale and MNIST format (0-255 where 0 is black and 255 is white)
	const mnistData = new Uint8Array(imageWidth * imageHeight);
	for (let i = 0; i < mnistData.length; i++) {
		// Get RGBA values
		const r = tempData[i * 4];
		const g = tempData[i * 4 + 1];
		const b = tempData[i * 4 + 2];
		const a = tempData[i * 4 + 3];
		
		// Handle transparency - if pixel is transparent, treat as black background
		if (a === 0) {
			mnistData[i] = 0; // Black in MNIST format
		} else {
			// Convert to grayscale using luminance formula
			// No inversion needed since we're already drawing white on black
			const gray = 0.2126 * r + 0.7152 * g + 0.0722 * b;
			mnistData[i] = Math.min(255, Math.max(0, Math.round(gray)));
		}
	}
	
	return mnistData;
}

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
    
    // Stop any existing polling
    stopModelPolling();
    
    // Update server status
    if (window.serverConfig) {
        window.serverConfig.updateStatus('connecting', 'Connecting...');
    }
    
    // Try to reload the model
    await loadModel();
};

// Load trained model from server
async function loadModel() {
	try {
		const response = await fetch(`${API_BASE_URL}/model`);
		const data = await response.json();
		
		if (data.error) {
			// Server is reachable but no trained model exists
			predictionResult.innerHTML = '<div class="model-status">No trained model found. Please train a model first in the <a href="model-trainer.html">Model Trainer</a>.</div>';
			
			// Update server status to show connected
			if (window.serverConfig) {
				window.serverConfig.updateStatus('connected', 'Connected');
			}
			
			// Start polling for a trained model if not already polling
			startModelPolling();
			return false;
		}
				// Reconstruct the neural network from server data
		neuralNetwork = new NeuralNetwork([784, 128, 10]);
		neuralNetwork.fromJSON(data);
		predictionResult.innerHTML = '<div class="model-status">Model loaded successfully! Draw a number to test it.</div>';
		
		// Update server status
		if (window.serverConfig) {
			window.serverConfig.updateStatus('connected', 'Connected');
		}
		
		// Stop polling since we have a model
		stopModelPolling();
		
		return true;
	} catch (error) {
		console.error('Error loading model from server:', error);
		predictionResult.innerHTML = '<div class="model-status error">Error connecting to server. Make sure the server is running.</div>';
		
		// Update server status
		if (window.serverConfig) {
			window.serverConfig.updateStatus('disconnected', 'Connection failed');
		}
		
		return false;
	}
}

// Predict the drawn number
function predictDrawing() {
	if (!neuralNetwork) {
		predictionResult.innerHTML = '<div class="model-status error">No model loaded. Please train a model first.</div>';
		return;
	}	try {
		// Convert canvas drawing to MNIST format
		const mnistData = canvasToMNIST(canvas);
		
		// Normalize pixel values to 0-1 range
		const normalizedPixels = Array.from(mnistData).map(p => p / 255.0);
		
		// Get prediction from neural network
		const prediction = neuralNetwork.forward(normalizedPixels);
		
		// Find the digit with highest confidence
		let predictedDigit = 0;
		let maxConfidence = prediction[0];
		
		for (let i = 1; i < prediction.length; i++) {
			if (prediction[i] > maxConfidence) {
				maxConfidence = prediction[i];
				predictedDigit = i;
			}
		}
		
		// Display results
		const confidencePercentage = (maxConfidence * 100).toFixed(2);
		predictionResult.innerHTML = `
			<div class="prediction-main">
				<span class="predicted-digit">${predictedDigit}</span>
			</div>
		`;
		confidenceLevel.innerHTML = `
			<div class="confidence-info">
				<span class="confidence-label">Confidence:</span>
				<span class="confidence-value">${confidencePercentage}%</span>
			</div>
			<div class="all-predictions">
				${prediction.map((conf, digit) => `
					<div class="prediction-item ${digit === predictedDigit ? 'highest' : ''}">
						<span class="digit">${digit}</span>
						<span class="confidence">${(conf * 100).toFixed(1)}%</span>
					</div>
				`).join('')}
			</div>
		`;
		
	} catch (error) {
		console.error('Error during prediction:', error);
		predictionResult.innerHTML = '<div class="model-status error">Error during prediction. Please try again.</div>';	}
}

// Start polling for trained model every 10 seconds
function startModelPolling() {
	// Don't start if already polling
	if (modelPollingInterval) {
		return;
	}
	
	console.log('Starting model polling every 10 seconds...');
	modelPollingInterval = setInterval(async () => {
		console.log('Polling for trained model...');
		const modelLoaded = await loadModel();
		if (modelLoaded) {
			console.log('Model found and loaded, stopping polling');
			stopModelPolling();
		}
	}, 10000); // Poll every 10 seconds
}

// Stop polling for trained model
function stopModelPolling() {
	if (modelPollingInterval) {
		console.log('Stopping model polling');
		clearInterval(modelPollingInterval);
		modelPollingInterval = null;
	}
}

// Initialize the application
async function init() {
	// Wait for server config to be ready
	if (window.serverConfig) {
		API_BASE_URL = window.serverConfig.getAPIBaseURL();
	}
	
	// Load the trained model from server
	await loadModel();
}