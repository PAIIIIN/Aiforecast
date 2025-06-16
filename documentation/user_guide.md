# Earthquake Forecasting Application - User Guide

## Introduction

This user guide provides instructions for using the Earthquake Forecasting Application for the Almaty region. The application allows you to view earthquake forecasts, historical data, and risk assessments.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. Clone or download the project repository
2. Navigate to the app directory
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

To start the application, run the following command from the app directory:

```bash
streamlit run app.py
```

This will start the application and open it in your default web browser. If it doesn't open automatically, you can access it at http://localhost:8501.

## Using the Application

### Forecast Options

In the sidebar, you can configure the following options:

- **Use existing predictions**: Toggle to use pre-generated predictions or generate new ones
- **Months to forecast**: Slider to select how many months ahead to forecast (3-24 months)

### Main Sections

The application is divided into several sections:

#### 1. Dataset Information

This section provides an overview of the earthquake data used for forecasting, including:
- Total number of earthquakes in the dataset
- Time range of the data
- Region radius around Almaty

#### 2. Earthquake Forecasts

This section is divided into three tabs:

##### Forecast Table
- Displays a table of forecasted earthquakes by month
- Shows predicted earthquake count, maximum magnitude, and risk level for each month

##### Forecast Charts
- **Monthly Earthquake Count**: Historical and predicted earthquake counts over time
- **Maximum Magnitude**: Historical and predicted maximum magnitudes over time
- **Risk Level**: Visual representation of monthly risk levels

##### Historical Data
- **Spatial Distribution**: Map showing the locations of historical earthquakes
- **Magnitude Distribution**: Histogram of earthquake magnitudes
- **Depth Distribution**: Histogram of earthquake depths

#### 3. Model Information

This section provides details about:
- The forecasting methodology used
- Features used in the models
- Model limitations
- Recommendations based on risk levels

## Understanding the Forecasts

### Risk Levels

The application categorizes earthquake risk into four levels:

- **Low Risk** (Green): 0 earthquakes or maximum magnitude < 4.0
- **Moderate Risk** (Yellow): < 5 earthquakes and maximum magnitude < 5.0
- **High Risk** (Orange): < 10 earthquakes and maximum magnitude < 6.0
- **Very High Risk** (Red): ≥ 10 earthquakes or maximum magnitude ≥ 6.0

### Interpreting the Results

- Forecasts are more reliable for the near future and uncertainty increases with time
- The model predicts monthly aggregates, not individual earthquake events
- Risk levels are based on both the predicted number of earthquakes and their maximum magnitudes

## Troubleshooting

### Common Issues

- **Application doesn't start**: Ensure all dependencies are installed correctly
- **No data appears**: Check that the data files are in the correct location
- **Error generating forecasts**: Verify that the model files are in the models directory

### Support

For additional support or to report issues, please contact the development team.

## Limitations

- The model is based on historical patterns and may not account for unexpected geological changes
- Predictions are more reliable for the near future and uncertainty increases with time
- The model does not incorporate geological or tectonic information that might improve predictions

## Conclusion

This application provides a user-friendly interface for earthquake forecasting in the Almaty region. By following this guide, you can effectively use the application to view forecasts, explore historical data, and assess earthquake risks.
