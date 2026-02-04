# Deep Learning Project: Deep FCNN on Tiny ImageNet

## Author
Vansh-Pandey

## Project Structure
- `Assignment-1/Task-3/3.1/code.ipynb`: Main notebook containing the Deep FCNN implementation and experiments.
- `datasets/`: Folder containing the Tiny ImageNet dataset (10 classes).
- `plots/`: Generated plots from experiments (saved in notebook directory).
- `requirements.txt`: Python dependencies.

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
**Important for GPU**: Install PyTorch with CUDA support first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
*(Note: You can use `cu118` if your drivers are older, but `cu121` is recommended for RTX 40 series)*

Then install the rest:
```bash
pip install numpy matplotlib tqdm jupyter
```

### 4. Run the Project
Start Jupyter Notebook:
```bash
jupyter notebook
```
Open `code.ipynb` and run all cells.
