# 🏭 PMF-IoT: Predictive Maintenance Forecasting Framework

## Overview

PMF-IoT is a Python-based framework designed for **Predictive Maintenance Forecasting** in IoT environments. It uses **ARIMA** (AutoRegressive Integrated Moving Average) models to analyze time-series data from sensors (Temperature and Vibration) and forecast potential equipment failure based on predefined thresholds. The system is exposed via a **Flask API**, allowing external applications to trigger model training, simulate data, and retrieve real-time failure predictions.



The project incorporates **Hyperparameter Optimization (Grid Search)** during training to automatically find the best ARIMA model orders $(p, d, q)$, ensuring robust performance without manual tuning.

## 🚀 Key Features

* **Multivariate Time Series Analysis:** Handles simultaneous forecasting for Temperature and Vibration sensors.
* **Automated Model Selection:** Implements a Grid Search to optimize ARIMA $(p, d, q)$ parameters based on AIC minimization.
* **Data Simulation:** Includes a robust `IoTSensorDataGenerator` to create synthetic historical data simulating gradual degradation and eventual failure.
* **Model Persistence:** Uses `joblib` to save and load trained ARIMA models, eliminating the need for retraining on every startup.
* **RESTful API:** Provides clear endpoints for system status, data generation, training, and prediction using Flask.
* **Alerting Mechanism:** Generates `CRITICAL` alerts when the forecasted sensor readings cross failure thresholds.

## 📂 Project Structure

. ├── data/ │ └── simulated_iot_data.csv # Generated historical data (ignored by git) ├── models/ │ ├── arima_temp_model.pkl # Saved Temperature Model (ignored by git) │ └── arima_vib_model.pkl # Saved Vibration Model (ignored by git) ├── api_server.py # Main Flask application and API endpoints ├── config.py # Global configuration, thresholds, and ARIMA ranges ├── data_simulator.py # Logic for generating synthetic sensor data └── model_pipeline.py # ML pipeline for ARIMA training, optimization, and prediction


## 🛠️ Setup and Installation

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/kingmaker007456/pmf-iot]
    cd PMF-IoT
    ```

2.  **Install dependencies:**
    This project primarily uses `flask`, `pandas`, `numpy`, `statsmodels`, and `joblib`.
    ```bash
    pip install pandas numpy statsmodels joblib flask
    ```

3.  **Run the application:**
    ```bash
    python api_server.py
    ```
    The server will start on `http://0.0.0.0:5000`.

## ⚙️ API Endpoints

The API base version is `/v1`.

| Endpoint | Method | Description | Response Code | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `/v1/status` | `GET` | Health check and current model status/ARIMA orders. | `200` | |
| `/v1/generate_data` | `POST` | Triggers the generation of a new `simulated_iot_data.csv` file. | `201` | **Must be run before `/v1/train`** |
| `/v1/train` | `POST` | Loads historical data, performs Grid Search, trains the optimal ARIMA models, and saves them. | `200` | Time-consuming operation. |
| `/v1/predict/<device_id>` | `GET` | Fetches the next **1 hour** (12 steps) forecast for the specified device. | `200` (Normal) or `202` (Critical Alert) | Requires models to be trained. |

### Example Workflow (cURL)

1.  **Generate Data:**
    ```bash
    curl -X POST http://localhost:5000/v1/generate_data
    ```

2.  **Train Models:**
    ```bash
    curl -X POST http://localhost:5000/v1/train
    ```

3.  **Get Prediction:**
    ```bash
    curl http://localhost:5000/v1/predict/IoT-Pump-001
    ```

## 🧠 Model Pipeline Details

The `model_pipeline.py` utilizes the following key components:

* **Training:** The `train_all` method triggers `_grid_search_arima`. The Grid Search iterates over the specified $p, d, q$ ranges (defined in `config.py`) and selects the order that minimizes the **AIC** (Akaike Information Criterion).
* **Prediction:** The `predict` method generates a forecast for the next 12 steps (1 hour, since `SAMPLE_FREQUENCY_MINUTES = 5`).
* **Alerting:** Prediction values are compared against:
    * Temperature Threshold: **45.0°C**
    * Vibration Threshold: **18.0**

## 📝 Configuration

All major settings can be modified in `config.py`:

| Variable | Description | Default Value | File |
| :--- | :--- | :--- | :--- |
| `SIMULATION_DAYS` | Length of the historical dataset in days. | `90` | `config.py` |
| `SAMPLE_FREQUENCY_MINUTES` | Data point interval. | `5` minutes | `config.py` |
| `FAILURE_THRESHOLD_TEMP` | Temperature limit for critical alert. | `45.0` | `config.py` |
| `P_RANGE, D_RANGE, Q_RANGE` | Search space for ARIMA Grid Search. | `range(0, 3)`, `range(0, 2)`, etc. | `config.py` |
| `PREDICTION_STEPS` | Number of future time steps to forecast. | `12` | `config.py` |

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests for bug fixes, new model implementations (e.g., Prophet, LSTM), or additional sensor support.