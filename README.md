# Deep Learning Project: FCNN on Adult Income Dataset and The Importance of Pre-Processing

## Author
Vandna

## Project Structure
- `Assignment-1/Task-1/1.1/Q1.1.ipynb`: notebook containing the FCNN on Adult Income Dataset. (Question 1.1)
- `Assignment-1/Task-1/1.2/Q1.2.ipynb`: Notebook containing the importance of preprocessing (Question 1.2)
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

## Q 1.1 FCNN on Adult Income Dataset

This project implements a **3-layer Fully Connected Neural Network (FCNN)**  **using NumPy only**.  
No deep learning frameworks such as PyTorch or TensorFlow were used.

The model is trained on the **Adult Census Income Dataset** to predict whether a person's income is greater than **$50K per year**.

This project helps understand how neural networks work internally by manually coding:

- Forward Propagation  
- Backpropagation (Chain Rule)  
- Gradient Descent Optimization

---

## Question 1.2 – Importance of Pre-Processing

In Question 1.2, the same FCNN architecture implemented in Question 1.1 is trained twice:

- Once using raw (unscaled) data  
- Once using Min-Max scaled data  

The objective is to analyze how feature scaling affects gradient behavior, training stability, convergence speed, and final accuracy.

### Key Observations
- Training with raw data results in unstable gradients due to large differences in feature scales.
- Features such as Capital Gain dominate gradient updates.
- Min-Max scaling balances gradients, improves convergence, and increases accuracy.

This experiment highlights that preprocessing is essential for effective neural network training.


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
