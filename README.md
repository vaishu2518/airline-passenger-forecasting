# ✈️ Airline Passenger Forecasting Using Python

## Project Overview
This project analyzes historical airline passenger data and forecasts future monthly passenger numbers using Python and Prophet.

The goal is to identify trends in passenger traffic, visualize historical patterns, and generate forecasts that demonstrate time-series analysis.

## Objectives
- Explore historical monthly airline passenger data.
- Analyze trends and patterns over time.
- Forecast passenger numbers for the next 24 months.
- Evaluate model performance using MAE and RMSE.
- Visualize actual versus predicted passenger numbers.

## Technologies Used
- Python
- Pandas
- Matplotlib
- Prophet
- Scikit-learn
- Plotly

## Dataset
The project uses a public airline passenger dataset containing 144 monthly observations from 1949 to 1960.

The passenger values are recorded in thousands.

## Project Workflow
1. Load the dataset.
2. Inspect data structure and missing values.
3. Visualize historical passenger trends.
4. Train a Prophet forecasting model.
5. Forecast the next 24 months.
6. Evaluate predictions on the final 24 historical months.
7. Compare actual and predicted passenger numbers.

## Model Evaluation
The current evaluation results are:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 31.17 thousand passengers |
| Root Mean Squared Error (RMSE) | 40.43 thousand passengers |

These results are from a test period consisting of the final 24 months of historical data. They provide an initial performance baseline; comparisons with simpler forecasting methods can help determine whether the model adds value.

## Visualizations
The project generates charts showing:
- Historical monthly passenger trends.
- Future passenger forecasts with uncertainty intervals.
- Actual versus predicted passenger numbers.

## How to Run
1. Clone or download this repository.
2. Install the required packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Run the program:

   ```bash
   python main.py
   ```

## Project Outcome
This project demonstrates practical experience with data loading, exploratory data analysis, time-series forecasting, model evaluation, and visualization.

## Future Improvements
- Compare Prophet with seasonal-naive and other forecasting methods.
- Experiment with different forecast settings.
- Add more evaluation metrics and clearer visual summaries.
- Develop an interactive dashboard.

## Author
BCA Data Science Student