# VLCID - Verbalized Learning Module for Confusing Intent Discrimination


## Features

- Multi-modal intent and sentiment classification
- Adaptive learning system with Learner, Optimizer, and Regularizer components
- Support for batch processing and incremental learning
- Automatic rule generation and refinement

## Requirements

- Python 3.7+
- OpenAI API compatible endpoints (supports Volcano Engine Doubao, Gemini, etc.)

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The main dependencies are:
- `openai>=1.0.0` - OpenAI API client
- `pandas>=1.0.0` - Data analysis (optional)
- `scikit-learn>=0.24.0` - Performance evaluation (optional)

## Configuration

### 1. API Configuration

Edit [learning_training_system.py](learning_training_system.py) (lines 14-24) to configure your API endpoints:

```python
# Learner & Optimizer API (e.g., Volcano Engine Doubao)
client = OpenAI(
    api_key="YOUR_API_KEY_HERE",
    base_url="YOUR_API_BASE_URL_HERE",
)

# Regularizer API (e.g., OpenAI proxy for Gemini-3)
regularizer_client = OpenAI(
    api_key="YOUR_REGULARIZER_API_KEY_HERE",
    base_url="YOUR_REGULARIZER_API_BASE_URL_HERE",
)
```

Replace:
- `YOUR_API_KEY_HERE` with your actual API key
- `YOUR_API_BASE_URL_HERE` with your API endpoint URL

### 2. Dataset Configuration

Prepare your training data in JSON format as `train均匀.json` in the project root directory.

### 3. Image Path Configuration

Edit [run_prod2.py](run_prod2.py) (line 25) to set your image directory path:

```python
image_path = "/your/path/to/images"  # Update this path
```

You can also adjust other parameters in [run_prod2.py](run_prod2.py):
- `train_json`: Training data filename
- `batch_size`: Batch size for error accumulation before calling Optimizer

## Quick Start

Once you have configured the API keys, dataset, and image path, simply run:

```bash
python run_prod2.py
```

## Training Output

During training, you will see:
- Current sample progress
- Predictions vs ground truth labels
- Optimizer and Regularizer call statistics
- Accuracy metrics
- Rule refinements

## Intent Labels

The system classifies 20 intent categories:
- Complain, Praise, Agree, Compromise, Query
- Joke, Oppose, Inform, Ask for help, Greet
- Taunt, Introduce, Guess, Leave, Advise
- Flaunt, Criticize, Thank, Comfort, Apologize

## Sentiment Labels

The system classifies 3 sentiment categories:
- Neutral
- Positive
- Negative
