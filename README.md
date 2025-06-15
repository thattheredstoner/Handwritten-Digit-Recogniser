# AI Number Recognition

A programming adventure project that implements handwritten digit recognition using the MNIST dataset. The project features a custom neural network built in Rust with a REST API backend and an interactive web frontend for training, testing, and visualizing the AI model.

**Development Time:** ~8 hours

## Features

- **Custom Neural Network Implementation** - Built from scratch in Rust without external ML libraries
- **MNIST Dataset Training** - Uses the standard 28x28 handwritten digit dataset  
- **REST API Backend** - Rust server with HTTP endpoints for model operations
- **Interactive Web Frontend** - Three comprehensive pages for complete model management:
  - **Drawing Tool** - Draw digits and get real-time predictions
  - **Model Trainer** - Configure and train neural network models
  - **Heat Map Analysis** - Visualize model statistics and performance data

## Project Structure

```
AI Number Recognition/
├── Server/                 # Rust backend
│   ├── src/
│   │   └── main.rs        # Neural network implementation & API server
│   ├── Cargo.toml         # Rust dependencies
│   ├── mnist-neural-network.exe # Compiled program
│   ├── trained_model.json # Saved model weights
│   └── test_data.json     # Test dataset cache
├── Client/                # Web frontend
│   ├── index.html         # Drawing tool page
│   ├── model-trainer.html # Training interface
│   ├── heatmap.html       # Analytics dashboard
│   ├── script.js          # Drawing canvas logic
│   ├── model-trainer.js   # Training controls
│   ├── heatmap.js         # Data visualization
│   ├── neural-network.js  # API communication
│   ├── server-config.js   # Server connection management
│   └── style.css          # UI styling
└── MNIST Dataset Files    # Binary dataset files
    ├── train-images.idx3-ubyte
    ├── train-labels.idx1-ubyte
    ├── t10k-images.idx3-ubyte
    └── t10k-labels.idx1-ubyte
```

## Technologies Used

### Backend (Rust)
- **Core Language:** Rust 2024 edition
- **Dependencies:**
  - `byteorder` - Binary data parsing for MNIST files
  - `rand` - Random number generation for weight initialization
  - `serde` + `serde_json` - JSON serialization for API responses
- **Features:**
  - Custom neural network with backpropagation
  - Momentum-based optimization
  - MNIST binary file parser
  - Multi-threaded HTTP server
  - Model persistence (save/load)

### Frontend (Web)
- **HTML5 Canvas** - Interactive drawing interface
- **Vanilla JavaScript** - No framework dependencies
- **CSS3** - Modern responsive design
- **REST API Integration** - Real-time communication with Rust backend

## Getting Started

### Prerequisites
- Rust (2024 edition or later)
- Web browser with JavaScript enabled
- MNIST dataset files (not included in project) - Note the location of the dataset files in the project structure relative to the compiled program

### Installation & Setup
**NOTE: Not suggested for running on public ports. This is a learning project, and so doesn't have adequate security measures in place.**

1. **Clone or download the project:**
   ```bash
   git clone <repository-url>
   cd "AI Number Recognition"
   ```

2. **Build and start the Rust server:**
   ```bash
   cd Server
   cargo build --release
   cargo run
   ```
   or use the pre-built executable:
   ```bash
   cd Server
   ./mnist-neural-network.exe
   ```
   The server will start on `localhost:3030`

3. **Open the web interface:**
   - Navigate to the `Client` folder
   - Open `index.html` in your web browser
   - Or serve the files using a local web server

### First Time Setup
1. Start with the **Model Trainer** page to train your first model
2. Configure training parameters (epochs, batch size, learning rate)
3. Click "Train New Model" and wait for completion
4. Switch to the **Drawing Tool** to test predictions
5. Use **Heat Map Analysis** to visualize model performance

## Usage Guide

### Drawing Tool (`index.html`)
- **Draw digits** on the canvas using your mouse
- **Get real-time predictions** as you draw
- **View confidence levels** for each prediction
- **Clear canvas** to try new digits

### Model Trainer (`model-trainer.html`)
- **Configure training parameters:**
  - Epochs: Number of training cycles (1-100)
  - Batch Size: Samples per training batch (10-1000) 
  - Learning Rate: Step size for weight updates (0.001-1.0)
- **Train new models** from scratch
- **Continue training** existing models
- **Monitor training progress** with real-time status updates
- **Delete models** to start fresh

### Heat Map Analysis (`heatmap.html`)
- **Load test data** to analyze model performance
- **View confusion matrices** and accuracy metrics
- **Visualize prediction confidence** across different digits
- **Monitor model statistics** and performance trends

## API Endpoints

The Rust server exposes several REST endpoints:

- `POST /train` - Start model training with parameters
- `POST /predict` - Get prediction for image data
- `GET /model-stats` - Retrieve model performance metrics
- `GET /test-data` - Load and return test dataset
- `DELETE /model` - Delete current trained model
- `POST /cancel-training` - Stop ongoing training process

## Technical Implementation

### Neural Network Architecture
- **Multi-layer perceptron** with configurable hidden layers
- **Sigmoid activation** functions
- **Backpropagation** algorithm for learning
- **Momentum optimization** for faster convergence and preventing local minima
- **Batch processing** for efficient training

### MNIST Data Processing
- **Binary file parsing** of IDX format files
- **Pixel normalization** (0-255 → 0-1 range)
- **One-hot encoding** for digit labels
- **Train/test split** handling

### Performance Features
- **Model persistence** - Save/load trained weights
- **Parallel processing** - Multi-threaded training
- **Memory efficient** - Streaming data processing
- **Real-time updates** - Live training progress

## Model Performance

Typical performance metrics with default settings:
- **Accuracy:** ~95-98%
- **Training Time:** 2-5 minutes (depending on parameters)
- **Prediction Speed:** <100ms per image

## Features Showcase

### Interactive Drawing
- Smooth canvas drawing with mouse and touch support
- Real-time prediction updates as you draw
- Confidence visualization for all 10 digits

### Advanced Training Controls
- Hyperparameter tuning interface
- Progress tracking with visual indicators
- Training cancellation and model management

### Data Visualization
- Performance heat maps and statistics
- Test data analysis and model insights
- Confusion matrix visualization

## License

This project is open source and available under the MIT License.
