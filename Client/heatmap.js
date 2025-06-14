// DOM Elements
const loadDataButton = document.getElementById('load-data-button');
const refreshButton = document.getElementById('refresh-button');
const statusMessage = document.getElementById('status-message');
const dataSummary = document.getElementById('data-summary');
const totalSamplesSpan = document.getElementById('total-samples');
const accuracySpan = document.getElementById('accuracy');
const correctSpan = document.getElementById('correct');
const misclassifiedSpan = document.getElementById('misclassified');
const heatmapGrid = document.getElementById('heatmap-grid');
const examplesGrid = document.getElementById('examples-grid');
const confidenceThreshold = document.getElementById('confidence-threshold');
const confidenceValue = document.getElementById('confidence-value');
const prevPageButton = document.getElementById('prev-page-button');
const nextPageButton = document.getElementById('next-page-button');
const pageInfo = document.getElementById('page-info');
const itemsPerPageSelect = document.getElementById('items-per-page');

// Global variables
let testData = null;
let filteredData = null;
let currentPage = 1;
let itemsPerPage = 100;

// Polling variables
let modelPollingInterval = null;

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
    
    // Reset the page state
    testData = null;
    currentPage = 1;
    filteredData = [];
    updateDataSummary();
    updatePagination();
    
    // Try to check for model and load test data
    await checkModelAndLoadData();
};

// Initialize the page
async function init() {
    // Setup event listeners
    loadDataButton.addEventListener('click', checkModelAndLoadData);
    refreshButton.addEventListener('click', checkModelAndLoadData);
    
    // Wait for server config to be ready
	if (window.serverConfig) {
		API_BASE_URL = window.serverConfig.getAPIBaseURL();
	}    
    
    // Filter controls
    const displayModeRadios = document.querySelectorAll('input[name="display-mode"]');
    displayModeRadios.forEach(radio => {
        radio.addEventListener('change', applyFilters);
    });
    
    // Initialize description
    updateHeatmapDescription();
    
    confidenceThreshold.addEventListener('input', (e) => {
        confidenceValue.textContent = Math.round(e.target.value * 100) + '%';
        applyFilters();
    });
    
    // Pagination controls
    prevPageButton.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            generateExamples();
        }
    });
    
    nextPageButton.addEventListener('click', () => {
        const totalPages = Math.ceil(filteredData.length / itemsPerPage);
        if (currentPage < totalPages) {
            currentPage++;
            generateExamples();
        }
    });
    
    itemsPerPageSelect.addEventListener('change', (e) => {
        itemsPerPage = parseInt(e.target.value);
        currentPage = 1;
        generateExamples();
    });
      // Check for model and load data on init
    await checkModelAndLoadData();
}

// Check if a trained model exists before loading test data
async function checkModelExists() {
    try {
        const response = await fetch(`${API_BASE_URL}/model`);
        const data = await response.json();
        
        // Return true if model exists, false if no model
        return !data.error;
    } catch (error) {
        console.error('Error checking for model:', error);
        return false;
    }
}

// Check for model and load data if available
async function checkModelAndLoadData() {
    try {
        statusMessage.textContent = 'Checking for trained model...';
        
        const modelExists = await checkModelExists();
        
        if (!modelExists) {
            // Server is reachable but no trained model exists
            statusMessage.textContent = 'No trained model found. Please train a model first in the Model Trainer, then click "Load Test Data".';
            
            // Update server status to show connected
            if (window.serverConfig) {
                window.serverConfig.updateStatus('connected', 'Connected');
            }
            
            // Start polling for a trained model
            startModelPolling();
            return false;
        }
        
        // Model exists, try to load test data
        await loadTestData();
        return true;
        
    } catch (error) {
        statusMessage.textContent = `Error connecting to server: ${error.message}`;
        console.error('Error checking model and loading data:', error);
        
        // Update server status
        if (window.serverConfig) {
            window.serverConfig.updateStatus('disconnected', 'Connection failed');
        }
        
        // Stop polling if server is unreachable
        stopModelPolling();
        
        return false;
    }
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
        const success = await checkModelAndLoadData();
        if (success) {
            console.log('Model found and data loaded, stopping polling');
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

// Load test data from the server
async function loadTestData() {
    try {
        statusMessage.textContent = 'Loading test data...';
        loadDataButton.disabled = true;
        refreshButton.disabled = true;
          const response = await fetch(`${API_BASE_URL}/test-data`);
        const data = await response.json();
        
        if (!response.ok || data.error) {
            throw new Error(data.message || data.error || 'Failed to load test data');
        }
          testData = data;
        statusMessage.textContent = 'Test data loaded successfully!';
        refreshButton.disabled = false;
        
        // Update server status
        if (window.serverConfig) {
            window.serverConfig.updateStatus('connected', 'Connected');
        }
        
        // Stop polling since we have data
        stopModelPolling();
        
        // Update summary
        updateDataSummary();
        
        // Apply initial filters and display
        applyFilters();
          } catch (error) {
        statusMessage.textContent = `Error: ${error.message}`;
        console.error('Error loading test data:', error);
        
        // Update server status
        if (window.serverConfig) {
            window.serverConfig.updateStatus('disconnected', 'Connection failed');
        }
        
        // Stop polling if server is unreachable
        stopModelPolling();
    } finally {
        loadDataButton.disabled = false;
        refreshButton.disabled = false;
    }
}

// Update data summary display
function updateDataSummary() {
    if (!testData) return;
    
    totalSamplesSpan.textContent = testData.total_samples.toLocaleString();
    accuracySpan.textContent = (testData.accuracy * 100).toFixed(2) + '%';
    correctSpan.textContent = testData.total_correct.toLocaleString();
    misclassifiedSpan.textContent = testData.misclassified_count.toLocaleString();
    
    dataSummary.classList.remove('hidden');
}

// Update heatmap description based on selected filter mode
function updateHeatmapDescription() {
    const displayMode = document.querySelector('input[name="display-mode"]:checked').value;
    const description = document.querySelector('.heatmap-description');
    
    if (displayMode === 'incorrect-by-ai-guess') {
        description.textContent = 'These heat maps show misclassified samples grouped by what the AI predicted them to be. Each map shows the pixel patterns of samples that the AI incorrectly classified as that digit.';
    } else {
        description.textContent = 'These heat maps show pixel-level overlaps from all sample images of each digit. Brighter colors indicate areas where multiple images have ink, revealing common patterns and stroke positions.';
    }
}

// Apply current filters to the data
function applyFilters() {
    if (!testData) return;
    
    const displayMode = document.querySelector('input[name="display-mode"]:checked').value;
    const minConfidence = parseFloat(confidenceThreshold.value);
    
    // Filter data based on current settings
    filteredData = testData.results.filter(result => {
        // Display mode filter
        if (displayMode === 'correct' && !result.is_correct) return false;
        if (displayMode === 'incorrect' && result.is_correct) return false;
        if (displayMode === 'incorrect-by-ai-guess' && result.is_correct) return false;
        
        // Confidence filter
        if (result.confidence < minConfidence) return false;
        
        return true;
    });
    
    // Reset pagination when filters change
    currentPage = 1;
      // Update displays
    updateHeatmapDescription();
    generateHeatMaps();
    generateExamples();
}

// Generate heat maps for each digit
function generateHeatMaps() {
    if (!filteredData) return;
    
    heatmapGrid.innerHTML = '';
    
    const displayMode = document.querySelector('input[name="display-mode"]:checked').value;
    
    // Group data differently based on display mode
    const digitGroups = {};
    for (let i = 0; i < 10; i++) {
        digitGroups[i] = [];
    }
    
    if (displayMode === 'incorrect-by-ai-guess') {
        // Group misclassified samples by AI's predicted digit
        filteredData.forEach(result => {
            if (!result.is_correct) {
                digitGroups[result.predicted].push(result);
            }
        });
    } else {
        // Group by actual digit (default behavior)
        filteredData.forEach(result => {
            digitGroups[result.actual].push(result);
        });
    }
    
    // Create heat map for each digit
    for (let digit = 0; digit < 10; digit++) {
        const digitData = digitGroups[digit];
        const heatmapDiv = createDigitHeatMap(digit, digitData, displayMode);
        heatmapGrid.appendChild(heatmapDiv);
    }
}

// Create heat map for a specific digit
function createDigitHeatMap(digit, digitData, displayMode = 'all') {
    const container = document.createElement('div');
    container.className = 'digit-heatmap';
    
    const title = document.createElement('h3');
    if (displayMode === 'incorrect-by-ai-guess') {
        title.textContent = `AI Predicted: ${digit}`;
    } else {
        title.textContent = `Digit ${digit}`;
    }
    container.appendChild(title);
    
    const stats = document.createElement('div');
    stats.className = 'heatmap-stats';
    
    const totalCount = digitData.length;
    const correctCount = digitData.filter(d => d.is_correct).length;
    const accuracy = totalCount > 0 ? (correctCount / totalCount * 100).toFixed(1) : 0;
    
    let statsContent = `
        <span>Samples: ${totalCount}</span>
        <span>Accuracy: ${accuracy}%</span>
        <span>Correct: ${correctCount}</span>
    `;
    
    // Add additional info for misclassified-by-AI-guess mode
    if (displayMode === 'incorrect-by-ai-guess' && totalCount > 0) {
        const actualDigits = [...new Set(digitData.map(d => d.actual))].sort((a, b) => a - b);
        statsContent += `<span>Actually: ${actualDigits.join(', ')}</span>`;
    }
    
    stats.innerHTML = statsContent;
    container.appendChild(stats);
    const canvas = document.createElement('canvas');
    canvas.className = 'heatmap-canvas';
    canvas.width = 300;
    canvas.height = 300;
    container.appendChild(canvas);
    // Draw heat map
    drawPredictionHeatMap(canvas, digitData);
    
    // Add interactivity
    addHeatMapInteractivity(canvas, digitData, digit);
    
    return container;
}

// Draw prediction heat map on canvas
function drawPredictionHeatMap(canvas, digitData) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, width, height);
    
    if (digitData.length === 0) {
        ctx.fillStyle = '#999999';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('No data', width / 2, height / 2);
        return;
    }
    
    // Create pixel heat map (28x28 grid)
    const imageSize = 28;
    const pixelCounts = new Array(imageSize * imageSize).fill(0);
    const maxPixelValue = 255;
    
    // Accumulate pixel values from all images
    digitData.forEach(result => {
        result.image.forEach((pixelValue, index) => {
            // Normalize pixel value (0-255) and add to heat map
            pixelCounts[index] += pixelValue / maxPixelValue;
        });
    });
    
    // Find max count for normalization
    const maxCount = Math.max(...pixelCounts);
    
    // Calculate display dimensions
    const pixelSize = Math.min(width, height) / imageSize;
    const offsetX = (width - (imageSize * pixelSize)) / 2;
    const offsetY = (height - (imageSize * pixelSize)) / 2;
    
    // Draw heat map
    for (let y = 0; y < imageSize; y++) {
        for (let x = 0; x < imageSize; x++) {
            const index = y * imageSize + x;
            const intensity = maxCount > 0 ? pixelCounts[index] / maxCount : 0;
            
            if (intensity > 0) {
                // Create heat map color (blue -> green -> yellow -> red)
                let r, g, b;
                if (intensity < 0.25) {
                    // Blue to Cyan
                    const t = intensity / 0.25;
                    r = 0;
                    g = Math.floor(t * 255);
                    b = 255;
                } else if (intensity < 0.5) {
                    // Cyan to Green
                    const t = (intensity - 0.25) / 0.25;
                    r = 0;
                    g = 255;
                    b = Math.floor((1 - t) * 255);
                } else if (intensity < 0.75) {
                    // Green to Yellow
                    const t = (intensity - 0.5) / 0.25;
                    r = Math.floor(t * 255);
                    g = 255;
                    b = 0;
                } else {
                    // Yellow to Red
                    const t = (intensity - 0.75) / 0.25;
                    r = 255;
                    g = Math.floor((1 - t) * 255);
                    b = 0;
                }
                
                ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            } else {
                ctx.fillStyle = '#1a1a1a';
            }
            
            const drawX = offsetX + x * pixelSize;
            const drawY = offsetY + y * pixelSize;
            ctx.fillRect(drawX, drawY, pixelSize, pixelSize);
        }
    }
    
    // Draw grid lines for better visibility
    ctx.strokeStyle = '#333333';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= imageSize; i++) {
        const pos = offsetX + i * pixelSize;
        ctx.beginPath();
        ctx.moveTo(pos, offsetY);
        ctx.lineTo(pos, offsetY + imageSize * pixelSize);
        ctx.stroke();
        
        const posY = offsetY + i * pixelSize;
        ctx.beginPath();
        ctx.moveTo(offsetX, posY);
        ctx.lineTo(offsetX + imageSize * pixelSize, posY);
        ctx.stroke();
    }
}

// Add hover handler to show detailed info for each heat map
function addHeatMapInteractivity(canvas, digitData, digit) {
    // Create tooltip element
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    canvas.parentElement.appendChild(tooltip);
    
    canvas.addEventListener('mousemove', (event) => {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // Convert to canvas coordinates
        const canvasX = (x / rect.width) * canvas.width;
        const canvasY = (y / rect.height) * canvas.height;
        
        // Calculate which pixel was hovered (28x28 grid)
        const imageSize = 28;
        const pixelSize = Math.min(canvas.width, canvas.height) / imageSize;
        const offsetX = (canvas.width - (imageSize * pixelSize)) / 2;
        const offsetY = (canvas.height - (imageSize * pixelSize)) / 2;
        
        const pixelX = Math.floor((canvasX - offsetX) / pixelSize);
        const pixelY = Math.floor((canvasY - offsetY) / pixelSize);

        // Calculate location for tooltip
        const tooltipX = event.clientX + window.scrollX;
        const tooltipY = event.clientY + window.scrollY;
        
        if (pixelX >= 0 && pixelX < imageSize && pixelY >= 0 && pixelY < imageSize) {
            showPixelTooltip(tooltip, tooltipX, tooltipY, digit, pixelX, pixelY, digitData);
        } else {
            hidePixelTooltip(tooltip);
        }
    });
    
    canvas.addEventListener('mouseleave', () => {
        hidePixelTooltip(tooltip);
    });
    
    canvas.style.cursor = 'crosshair';
}

// Show tooltip with detailed information about a specific pixel
function showPixelTooltip(tooltip, x, y, digit, pixelX, pixelY, digitData) {
    const pixelIndex = pixelY * 28 + pixelX;
    let totalValue = 0;
    let sampleCount = 0;
    
    digitData.forEach(result => {
        if (result.image[pixelIndex] > 0) {
            totalValue += result.image[pixelIndex];
            sampleCount++;
        }
    });
    
    const avgValue = sampleCount > 0 ? (totalValue / sampleCount).toFixed(1) : 0;
    const coverage = ((sampleCount / digitData.length) * 100).toFixed(1);
    
    tooltip.innerHTML = `
        <div class="heatmap-tooltip-title">Digit ${digit} - Pixel (${pixelX}, ${pixelY})</div>
        <div class="heatmap-tooltip-info">
            Coverage: ${coverage}% (${sampleCount}/${digitData.length})<br>
            Avg intensity: ${avgValue}/255
        </div>
    `;
    
    // Position tooltip with bounds checking
    const tooltipRect = tooltip.getBoundingClientRect();
    const viewportWidth = document.body.offsetWidth + window.scrollX;
    const viewportHeight = window.innerHeight + window.scrollY;

    let left = x + 10;
    let top = y + 10;

    // Check right boundary
    if (left + tooltipRect.width > viewportWidth) {
        left = x - tooltipRect.width - 10;
    }
    
    // Check bottom boundary
    if (top + tooltipRect.height > viewportHeight) {
        top = y - tooltipRect.height - 10;
    }
    
    // Check left boundary
    if (left - 10 < 0) {
        left = 10;
    }
    
    // Check top boundary
    if (top + 10 < 0) {
        top = 10;
    }
    
    tooltip.style.left = left + 'px';
    tooltip.style.top = top + 'px';
    tooltip.style.opacity = '1';
}

// Hide tooltip
function hidePixelTooltip(tooltip) {
    tooltip.style.opacity = '0';
}

// Generate example images with pagination
function generateExamples() {
    if (!filteredData) return;
    
    examplesGrid.innerHTML = '';
    
    // Calculate pagination
    const totalItems = filteredData.length;
    const totalPages = Math.ceil(totalItems / itemsPerPage);
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = Math.min(startIndex + itemsPerPage, totalItems);
    
    // Update pagination controls
    prevPageButton.disabled = currentPage <= 1;
    nextPageButton.disabled = currentPage >= totalPages;
    pageInfo.textContent = `Page ${currentPage} of ${totalPages} (${totalItems} items)`;
    
    // Get current page examples
    const examples = filteredData.slice(startIndex, endIndex);
    
    examples.forEach((result, index) => {
        const exampleDiv = createExampleItem(result, startIndex + index + 1);
        examplesGrid.appendChild(exampleDiv);
    });
    
    // Show loading message if no examples
    if (examples.length === 0) {
        const noData = document.createElement('div');
        noData.className = 'no-data-message';
        noData.textContent = 'No examples match the current filters.';
        examplesGrid.appendChild(noData);
    }
}

// Create an example item
function createExampleItem(result, index) {
    const container = document.createElement('div');
    container.className = 'example-item';
    
    const canvas = document.createElement('canvas');
    canvas.className = 'example-canvas';
    canvas.width = 100;
    canvas.height = 100;
    container.appendChild(canvas);
    
    // Draw image
    drawMNISTImage(canvas, result.image);
    
    const info = document.createElement('div');
    info.className = 'example-info';
    
    const actualClass = result.is_correct ? 'actual' : 'actual';
    const predictedClass = result.is_correct ? 'predicted' : 'predicted incorrect';
    
    info.innerHTML = `
        <div class="example-index">#${index}</div>
        <div>Actual: <span class="${actualClass}">${result.actual}</span></div>
        <div>Predicted: <span class="${predictedClass}">${result.predicted}</span></div>
        <div>Confidence: <span class="confidence">${(result.confidence * 100).toFixed(1)}%</span></div>
    `;
    
    container.appendChild(info);
    
    return container;
}

// Draw MNIST image on canvas
function drawMNISTImage(canvas, imageData) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, width, height);
    
    // Draw image (28x28 pixels)
    const imageSize = Math.sqrt(imageData.length); // Should be 28
    const pixelSize = Math.min(width, height) / imageSize;
    
    for (let y = 0; y < imageSize; y++) {
        for (let x = 0; x < imageSize; x++) {
            const pixelIndex = y * imageSize + x;
            const pixelValue = imageData[pixelIndex];
            
            // Convert to grayscale (MNIST is already grayscale)
            const grayValue = pixelValue;
            ctx.fillStyle = `rgb(${grayValue}, ${grayValue}, ${grayValue})`;
            
            const drawX = x * pixelSize;
            const drawY = y * pixelSize;
            ctx.fillRect(drawX, drawY, pixelSize, pixelSize);
        }
    }
}

// Initialize the page when loaded
window.addEventListener('DOMContentLoaded', init);

// Clean up polling when page is unloaded
window.addEventListener('beforeunload', stopModelPolling);
