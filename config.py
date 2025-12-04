import os
from datetime import datetime

# --- General System Configuration ---
PROJECT_NAME = "PMF-IoT_Predictive_Maintenance_Framework"
MODEL_SAVE_DIR = "models"
DATA_SAVE_DIR = "data"
LOG_FILE = "pmf_iot.log"

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(DATA_SAVE_DIR, exist_ok=True)

# --- Data Simulation Configuration ---
START_DATE = datetime(2025, 1, 1, 0, 0, 0)
SIMULATION_DAYS = 90  # 90 days of historical data
SAMPLE_FREQUENCY_MINUTES = 5
FAILURE_THRESHOLD_TEMP = 45.0
FAILURE_THRESHOLD_VIB = 18.0
FAILURE_DAY_OFFSET = 70 # Failure starts manifesting around day 70

# --- ARIMA Model Configuration ---
# Initial (p, d, q) - These will now be overwritten by the best_order from training
ARIMA_ORDER_TEMP = (5, 1, 0)
ARIMA_ORDER_VIB = (3, 1, 1)

# Hyperparameter Search Space for Grid Search
P_RANGE = range(0, 3)
D_RANGE = range(0, 2)
Q_RANGE = range(0, 3)

PREDICTION_STEPS = 12  # Predict the next hour (12 * 5 minutes)
MODEL_TEMP_FILENAME = os.path.join(MODEL_SAVE_DIR, 'arima_temp_model.pkl')
MODEL_VIB_FILENAME = os.path.join(MODEL_SAVE_DIR, 'arima_vib_model.pkl')

# --- Flask API Configuration ---
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
API_VERSION = "v1"

# --- Logging Configuration (Basic Example) ---
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'standard',
            'filename': LOG_FILE,
            'maxBytes': 1024 * 1024 * 5,  # 5 MB
            'backupCount': 5,
            'level': 'WARNING',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
}
