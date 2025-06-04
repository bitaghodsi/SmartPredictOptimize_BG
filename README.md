# Scheduling Loss Comparison

This project compares different loss functions for predicting task durations in scheduling problems, with the goal of minimizing total completion time (Sum of Completion Times, SCT). A simple neural network is trained using various loss functions, and their performance is evaluated in terms of SCT on held-out data.

## Problem Setting

Given input features about scheduled visits (e.g., appointment type, department, floor), the goal is to predict the actual duration of each visit and determine an ordering that minimizes the sum of completion times.

## Loss Functions Compared

- **Regret Loss (Sigmoid-based):** Smooth approximation of regret when predicting task durations out of order.
- **Rank Loss:** Pairwise loss penalizing misranked durations.
- **MSE Loss:** Standard regression loss.
- **SPO+ Loss:** Differentiable optimization-based loss designed to directly minimize scheduling cost.

## Evaluation Metric

All models are evaluated using **Relative SCT Error**:
\[
\text{Relative SCT Error} 
\]

## Project Structure

- `notebook.ipynb` – Full training and evaluation workflow
- `scheduled_visits_Bita.csv` – Input dataset (not included in repo)
- `README.md` – Project overview

## Requirements

- Python 3.8+
- PyTorch
- NumPy, Pandas, Matplotlib
- scikit-learn
- cvxpy, cvxpylayers

Install required packages with:
```bash
pip install -r requirements.txt
```

## How to Run

1. Open the notebook in Google Colab or Jupyter.
2. Upload the dataset.
3. Run all cells to train and compare models.
4. View the final plot to see how each loss function performs in terms of SCT.

## License

This project is for academic and educational use.
