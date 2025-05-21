# da6401_assignment3
A repository to build a transliteration system using recurrent neural networks .

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Prediction](#prediction)
- [Model Architecture](#model-architecture)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Visualization](#visualization-tools)
- [File Structure](#file-structure)

The Goal:
 (i) learn how to model sequence to sequence learning problems using Recurrent Neural Networks 
 (ii) compare different cells such as vanilla RNN, LSTM and GRU 
 (iii) understand how attention networks overcome the limitations of vanilla seq2seq models 
 (iv) visualise the interactions between different components in a RNN based model.

- [Assignment Link](https://wandb.ai/sivasankar1234/DA6401/reports/Assignment-3--VmlldzoxMjM4MjYzMg)  
- [Report Link](https://api.wandb.ai/links/ns24z274-iitm-ac-in/olmlpgsu)

## **Setup**

### **Clone the repository:**  
```bash
   git clone https://github.com/ak4off/da6401_assignment3.git
   cd da6401_assignment3
```
## Requirements
- Python 3.7+
- PyTorch 1.8+
- Weights & Biases (wandb)
- Other dependencies: numpy, matplotlib, seaborn, tqdm

### **Installation**
```bash
pip install wandb
pip install numpy
pip intall torch torchvision
pip install -r requirements.txt
wandb login
```
# Usage

## Training

## Basic Training
```bash
python train.py
```
## Hyperparameter Tuning
```bash
# For attention models
python wandb_sweep_attn.py

# For non-attention models 
python wandb_sweep_noAttn.py

```
### Key arguments:

```bash
--use_wandb          # Enable wandb logging
--use_attention      # Use attention mechanism  
--cell_type          # rnn/lstm/gru
--bidirectional      # Bidirectional encoder
--in_embed_dims      # Embedding dimension (128/256)
--hidden_layer_size  # Hidden layer size (128/256)
--n_layers          # Number of RNN layers (1/2)/
```


## Evaluation
Automatically evaluates on:
- Validation loss/accuracy
- Test loss/accuracy

## Prediction
Predictions saved in:
- `predictions_attention/` (attention models)
- `predictions_vanilla/` (non-attention)

## Model Architecture
### Encoder
- Embedding layer
- Configurable RNN (RNN/LSTM/GRU)
- Bidirectional option
- Dropout

### Decoder
- Embedding layer
- Configurable RNN
- Attention mechanism (optional)

## File Structure
```
./da6401_assignment3/
├── train.py                   # Main training
├── wandb_sweep*.py             # Hyperparameter sweeps
├── evaluate.py                 # Evaluation
├── predict.py                  # Prediction
├── the_data/                   # Data processing
├── model/                      # Model code
└── utils/                      # Utilities
```


# Complete File Structure Breakdown

## Root Directory Files
```
da6401_assignment3/
├── train.py                   # Main training script
├── wandb_sweep.py              # Combined hyperparameter sweep config
├── wandb_sweep_attn.py         # Attention-specific sweep config  
├── wandb_sweep_noAttn.py       # Non-attention sweep config
├── evaluate.py                 # Model evaluation functions
├── predict.py                  # Prediction generation script
```

## the_data/ Directory
```
the_data/
├── load_data.py        # Handles data loading and preprocessing
├── dataset.py          # Custom Dataset class implementation  
├── vocab.py            # Vocabulary and text processing
├── collate.py          # Batch collation functions
```

### Key Data Files
- `train.tsv`/`dev.tsv`/`test.tsv` # Training/validation/test data 

## model/ Directory
```
model/
├── encoder_decoder.py           # Base Seq2Seq model without attention
├── attention_encoder_decoder.py # Attention-enhanced Seq2Seq model
```

### Key Components
- **Encoder**: Bidirectional RNN/LSTM/GRU
- **Decoder**: With/without attention mechanism
- **Attention**: Implements Bahdanau-style attention

## utils/ Directory
```
utils/
├── decode.py           # Sequence decoding utilities  
├── wandb_logger.py     # Weights & Biases integration
├── visualize.py        # Attention visualization tools
```

### Visualization Features
- Attention heatmaps
- Interactive connectivity graphs
- PCA of hidden states
- W&B logging integration

## Key Implementation Files

### train.py
Main training script that:
- Handles model initialization
- Manages training loop
- Implements evaluation
- Handles wandb integration
- Supports both attention and non-attention models

### wandb_sweep*.py
Configuration files for hyperparameter tuning with:
- Bayesian optimization
- Parameter search spaces
- Attention/non-attention variants

### predict.py
Prediction utilities featuring:
- Batch prediction generation
- Attention visualization
- HTML output generation
- Results saving

### evaluate.py
Evaluation metrics including:
- Loss calculation
- Accuracy measurement
- Batch-wise evaluation

## Visualization Tools
- Attention heatmaps
- Interactive connectivity
- PCA of hidden states
- W&B automatic logging

## Citation

If you use the Dakshina dataset in your work, please cite:

```
Roark, B., et al. (2020). The Dakshina Dataset: A Multilingual, Multiscript Dataset for Research on Transliteration. LREC.
```
