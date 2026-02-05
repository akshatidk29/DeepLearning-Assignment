# Deep Learning Project: FCNN on Adult Income Dataset

## Author
Vandna

## Project Overview

This project implements a **3-layer Fully Connected Neural Network (FCNN)** ** using NumPy only**.  
No deep learning frameworks such as PyTorch or TensorFlow were used.

The model is trained on the **Adult Census Income Dataset** to predict whether a person's income is greater than **$50K per year**.

This project helps understand how neural networks work internally by manually coding:

- Forward Propagation  
- Backpropagation (Chain Rule)  
- Gradient Descent Optimization

## Project Structure
- `Assignment-1/Task-1/1.1/Q1.1.ipynb`: Main notebook containing the FCNN on Adult Income Dataset.
- `adult.data`: This file contain the dataset.

  ---

## Features Implemented

- He Initialization for weights  
- ReLU Activation Function  
- Sigmoid Output Layer  
- Binary Cross-Entropy Loss  
- Manual Backpropagation using Chain Rule  
- SGD (Stochastic Gradient Descent) Optimizer  
- Accuracy Evaluation  

---


## Setup Instructions

### 1. Create a Virtual Environment
Open a terminal in this directory (`DeepLearning-Project`) and run:
```bash
python -m venv venv
```

### 2. Activate the Virtual Environment
- **Windows**:
  ```powershell
  .\venv\Scripts\Activate
  ```
- **Linux/Mac**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install numpy pandas matplotlib jupyter
```

### 4. Run the Project
Start Jupyter Notebook:
```bash
jupyter notebook
```
Open `code.ipynb` and run all cells.
