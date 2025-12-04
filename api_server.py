from flask import Flask, jsonify, request
import logging.config
import pandas as pd
from datetime import datetime
from config import FLASK_HOST, FLASK_PORT, API_VERSION, LOGGING_CONFIG
from model_pipeline import PMFPipeline
from data_simulator import IoTSensorDataGenerator

# Initialize logging before creating the Flask app
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('PMF-IoT_API')

# Initialize Flask app and the ML Pipeline
app = Flask(__name__)
pipeline = PMFPipeline()
generator = IoTSensorDataGenerator() # Initialize the generator globally

# --- Application Startup: Load Model on Init ---
def load_initial_models():
    """Attempt to load trained models when the server starts."""
    if pipeline.load_models():
        logger.info("Successfully loaded previously trained models.")
    else:
        logger.warning("No pre-trained models found. Prediction endpoint will fail until /train is called.")

# Run the loader function
with app.app_context():
    load_initial_models()

# --- Helper Functions ---
def get_historical_data() -> pd.DataFrame:
    """A helper function to simulate fetching the latest historical data chunk."""
    data_path = 'data/simulated_iot_data.csv'
    try:
        # Use try-except to handle case where file might not exist yet
        df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
        return df
    except FileNotFoundError:
        logger.error(f"Historical data file not found at {data_path}. Please run /generate_data.")
        return pd.DataFrame()


# --- API Endpoints ---

@app.route(f'/{API_VERSION}/status', methods=['GET'])
def system_status():
    """Provides a health check and model status."""
    status = {
        "status": "Running",
        "api_version": API_VERSION,
        "model_loaded_temp": pipeline.temp_model is not None,
        "model_loaded_vib": pipeline.vib_model is not None,
        "temp_arima_order": str(pipeline.temp_order),
        "vib_arima_order": str(pipeline.vib_order),
        "current_timestamp": datetime.now().isoformat()
    }
    return jsonify(status), 200

@app.route(f'/{API_VERSION}/generate_data', methods=['POST'])
def generate_data_endpoint():
    """
    Triggers the generation of new synthetic historical data.
    """
    logger.info("API request received to generate new historical data.")
    try:
        # Generate and save a new dataset
        file_path = generator.generate_and_save(device_id="IoT-Pump-001")
        return jsonify({
            "message": "New historical data successfully generated and saved.",
            "file_path": file_path,
            "data_points": generator.n_samples
        }), 201 # Created
    except Exception as e:
        logger.error(f"Error during data generation: {e}")
        return jsonify({"message": f"Data generation failed: {e}"}), 500


@app.route(f'/{API_VERSION}/train', methods=['POST'])
def train_models():
    """
    Triggers a full model re-training using the latest historical data.
    """
    logger.info("API request received to re-train models.")
    historical_df = get_historical_data()
    
    if historical_df.empty:
        return jsonify({
            "message": "Training failed: Could not load historical data. Run /generate_data first."
        }), 500
    
    try:
        pipeline.train_all(historical_df)
        return jsonify({
            "message": "Model training successfully triggered and models saved.",
            "data_points_used": len(historical_df),
            "new_temp_order": str(pipeline.temp_order),
            "new_vib_order": str(pipeline.vib_order)
        }), 200
    except Exception as e:
        logger.error(f"Error during API-triggered training: {e}")
        return jsonify({"message": f"Model training failed due to internal error: {e}"}), 500


@app.route(f'/{API_VERSION}/predict/<device_id>', methods=['GET'])
def get_prediction(device_id: str):
    """
    Fetches the latest prediction for a specific device ID based on the currently loaded models.
    """
    if not pipeline.temp_model or not pipeline.vib_model:
        return jsonify({
            "error": "Models not ready.",
            "message": "Models must be trained first. Please use the /train endpoint."
        }), 503 # Service Unavailable
    
    try:
        prediction_output = pipeline.predict()
        prediction_output['device_id'] = device_id
        
        # Determine HTTP status code based on alert level
        status_code = 200
        if prediction_output['alert_status'] == 'CRITICAL':
            status_code = 202 # Accepted, but there is a critical alert
            
        return jsonify(prediction_output), status_code
    
    except RuntimeError as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed due to runtime error: {e}"}), 500
    except Exception as e:
        logger.error(f"Unhandled error in prediction: {e}")
        return jsonify({"error": "An unexpected server error occurred."}), 500


if __name__ == '__main__':
    logger.info(f"Starting Flask server on {FLASK_HOST}:{FLASK_PORT}...")
    # Setting use_reloader=False when using logging.config to prevent double logging
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True, use_reloader=False)
