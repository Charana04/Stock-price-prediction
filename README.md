
# Stock Price Prediction using Machine Learning

This repository contains a machine learning project focused on predicting stock prices using historical data. The project uses **Linear Regression** and **Random Forest Regression** models to forecast future stock prices based on key features like opening price, closing price, trading volume, and moving averages. It also includes data preprocessing, feature engineering, and evaluation of the models’ performance.

## Features

- **Data Loading and Preprocessing**: 
    - Loads historical stock data from a CSV file.
    - Cleans the data by handling missing values and removing anomalies.
    - Performs feature engineering (e.g., adding 7-day moving averages).
    
- **Machine Learning Models**:
    - Trains **Linear Regression** and **Random Forest Regressor** models.
    - Uses training and testing data splits for model evaluation.

- **Model Evaluation**:
    - Measures model performance using metrics like **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.
    - Provides **Actual vs Predicted** plots for visual performance comparison.
    - Analyzes **Feature Importance** for Random Forest to highlight influential predictors.

- **Visualization**:
    - **Correlation Heatmap** to show relationships between stock features.
    - **Actual vs Predicted** scatter plot for model performance visualization.

## Prerequisites

To run this project locally, you need Python 3.x and the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install the required libraries using:

```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/your-username/stock-price-prediction.git
    ```

2. Upload your dataset in CSV format or use the provided example (make sure the dataset has the columns like 'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume').

3. Run the Python script (`stock_price_prediction.py`), which will execute the data preprocessing, model training, and evaluation.

    ```bash
    python stock_price_prediction.py
    ```

4. View the output plots and evaluation metrics for both models.

## Project Structure

```
stock-price-prediction/
│
├── data/                     # Folder for storing the dataset(s)
├── notebooks/                # Jupyter notebooks (if applicable)
├── stock_price_prediction.py # Python script for the project
├── requirements.txt          # List of Python dependencies
├── README.md                 # This README file
└── LICENSE                   # License information (if any)
```

## Model Evaluation

After training the models, the following performance metrics are computed:

- **Mean Absolute Error (MAE)**: Measures the average of the absolute errors between predicted and actual stock prices.
- **Root Mean Squared Error (RMSE)**: Measures the square root of the average of squared errors, providing a clearer picture of model accuracy.
- **Feature Importance**: Random Forest identifies which features (e.g., open, high, low, volume) are most important for predicting stock prices.

## Visualizations

- **Actual vs Predicted** scatter plot shows the relationship between the predicted and actual stock prices.
- **Correlation Heatmap** visualizes the correlation between features to understand which ones are related to each other.

## Future Improvements

- Incorporating more advanced time-series models like **LSTM** (Long Short-Term Memory) for better accuracy.
- Hyperparameter tuning to optimize model performance.
- Adding more features (e.g., moving averages for different time windows, technical indicators like RSI, MACD, etc.).

