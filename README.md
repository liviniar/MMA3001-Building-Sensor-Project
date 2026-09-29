# MMA3001 Building Sensor Project

## Project Overview

This project investigates one-hour-ahead indoor temperature
forecasting using environmental sensor data.

Three forecasting methods were compared:
- Persistence baseline
- Linear regression
- Decision-tree regression

## Dataset

Environmental sensor measurements were obtained from the
provided building sensor dataset.

The final analysis uses 29,557 observations from one sensor,
with temperature forecasts approximately one hour ahead.

Input variables:
- Current temperature
- Humidity
- Carbon dioxide
- Hour of day
- Day of week

## Methodology

The dataset was divided chronologically into training,
validation and testing periods.

Decision-tree depth was optimised using validation data.
The selected model was then retrained using the combined
training and validation datasets.

The test dataset was reserved for final evaluation.

## Final Test Results

| Model | MAE (°C) | RMSE (°C) | R² |
|---|---:|---:|---:|
| Persistence | 0.468 | 0.636 | 0.955 |
| Linear regression | 0.465 | 0.631 | 0.956 |
| Decision tree | 0.386 | 0.560 | 0.965 |

The selected depth-8 decision tree achieved a 17.5%
reduction in MAE compared with persistence.

## Project Structure

- data/ - Raw datasets (excluded from GitHub)
- notebooks/ - Data investigation and modelling
- src/ - Reusable Python functions
- tests/ - Automated tests
- results/ - Evaluation results and figures
- docs/ - Generated source-code documentation
- models/ - Saved trained model

## Testing

Automated tests are implemented using pytest.

Run:

    python -m pytest tests/ -v

## Dependencies

Install the required packages using:

    pip install -r requirements.txt

## Limitations

Current temperature dominates model feature importance.
Prediction errors tend to increase at higher temperatures.
Results are based on one environmental sensor, so
performance may differ for other sensors or buildings.
