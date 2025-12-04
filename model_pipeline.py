import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib
import warnings
import logging
from typing import Tuple, Dict, Optional, List
from itertools import product
from config import (
    ARIMA_ORDER_TEMP, ARIMA_ORDER_VIB, PREDICTION_STEPS,
    MODEL_TEMP_FILENAME, MODEL_VIB_FILENAME,
    FAILURE_THRESHOLD_TEMP, FAILURE_THRESHOLD_VIB,
    P_RANGE, D_RANGE, Q_RANGE
)

# Suppress warnings from statsmodels
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

class PMFPipeline:
    """
    Predictive Maintenance Forecasting Pipeline using ARIMA models.
    Manages training, saving, loading, and prediction for multiple sensors.
    """
    def __init__(self):
        self.temp_model: Optional[ARIMA] = None
        self.vib_model: Optional[ARIMA] = None
        # Store the orders actually used after training/loading
        self.temp_order: Tuple[int, int, int] = ARIMA_ORDER_TEMP
        self.vib_order: Tuple[int, int, int] = ARIMA_ORDER_VIB
        logger.info(f"Pipeline initialized with default orders. Best orders will be determined during training.")

    def preprocess_data(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Performs basic preprocessing (cleaning) and returns
        the time series for each sensor.
        """
        df = df.dropna()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        logger.info(f"Data preprocessed. Remaining samples: {len(df)}")
        return {
            'temperature': df['temperature'],
            'vibration': df['vibration']
        }

    def _grid_search_arima(self, data: pd.Series, p_range: range, d_range: range, q_range: range, sensor_name: str) -> Optional[Tuple[Tuple[int, int, int], ARIMA]]:
        """
        Performs a grid search to find the optimal ARIMA(p,d,q) order
        based on the lowest Akaike Information Criterion (AIC).
        """
        best_aic = float('inf')
        best_order = None
        best_model_fit = None
        
        orders = list(product(p_range, d_range, q_range))
        
        logger.info(f"Starting Grid Search for {sensor_name} across {len(orders)} combinations.")
        
        for order in orders:
            try:
                # Use a smaller subset of data for faster grid search
                subset_data = data[-500:] 
                model = ARIMA(subset_data, order=order)
                model_fit = model.fit(disp=False)
                
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = order
                    best_model_fit = model_fit
                    # logger.debug(f"New best for {sensor_name}: ARIMA{order} with AIC={best_aic:.2f}")
            except Exception:
                continue # Skip this order if it fails to converge
        
        if best_order:
            logger.info(f"Grid Search complete for {sensor_name}. Best ARIMA{best_order} with AIC={best_aic:.2f}")
            return (best_order, best_model_fit)
        else:
            logger.error(f"Grid Search failed to find a valid model for {sensor_name}.")
            return None

    def train_model(self, data: pd.Series, sensor_name: str, p_range: range = P_RANGE, d_range: range = D_RANGE, q_range: range = Q_RANGE) -> Optional[ARIMA]:
        """
        Finds the optimal order and trains the final ARIMA model on the full data.
        """
        # Step 1: Find best order
        grid_search_result = self._grid_search_arima(data, p_range, d_range, q_range, sensor_name)
        
        if not grid_search_result:
            return None
        
        best_order, _ = grid_search_result
        
        # Step 2: Train final model on ALL data with the best order
        logger.info(f"Training final ARIMA{best_order} for {sensor_name} on full dataset...")
        try:
            model = ARIMA(data, order=best_order)
            model_fit = model.fit(disp=False)
            logger.info(f"Final training complete for {sensor_name}.")
            return model_fit
        except Exception as e:
            logger.error(f"Failed to train final ARIMA model for {sensor_name} with order {best_order}: {e}")
            return None

    def train_all(self, df: pd.DataFrame):
        """Trains both temperature and vibration models."""
        processed_data = self.preprocess_data(df)

        self.temp_model = self.train_model(processed_data['temperature'], 'temperature')
        self.vib_model = self.train_model(processed_data['vibration'], 'vibration')
        
        if self.temp_model and self.vib_model:
            # Update the stored orders based on the trained models
            self.temp_order = self.temp_model.model.order
            self.vib_order = self.vib_model.model.order
            self.save_models()
        else:
            logger.error("One or both models failed to train and were not saved.")

    def save_models(self):
        """Saves the trained models and their orders."""
        if self.temp_model:
            # Save the fitted model
            joblib.dump(self.temp_model, MODEL_TEMP_FILENAME)
            logger.info(f"Temperature model saved to {MODEL_TEMP_FILENAME}")
        if self.vib_model:
            # Save the fitted model
            joblib.dump(self.vib_model, MODEL_VIB_FILENAME)
            logger.info(f"Vibration model saved to {MODEL_VIB_FILENAME}")

    def load_models(self) -> bool:
        """Loads the trained models from disk."""
        try:
            self.temp_model = joblib.load(MODEL_TEMP_FILENAME)
            self.vib_model = joblib.load(MODEL_VIB_FILENAME)
            # Update stored orders from the loaded model object
            self.temp_order = self.temp_model.model.order
            self.vib_order = self.vib_model.model.order
            logger.info(f"All models loaded successfully. TEMP order: {self.temp_order}, VIB order: {self.vib_order}")
            return True
        except FileNotFoundError:
            logger.warning("Model files not found. Models must be trained first.")
            return False
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def predict(self, steps: int = PREDICTION_STEPS) -> Dict:
        """
        Generates a multi-sensor prediction and checks for failure alerts.
        """
        if not self.temp_model or not self.vib_model:
            raise RuntimeError("Models are not loaded or trained. Cannot predict.")

        # --- Temperature Prediction ---
        temp_forecast: pd.Series = self.temp_model.forecast(steps=steps)
        temp_alert = (temp_forecast > FAILURE_THRESHOLD_TEMP).any()
        temp_alert_time: Optional[str] = None
        if temp_alert:
            idx = temp_forecast[temp_forecast > FAILURE_THRESHOLD_TEMP].index[0]
            temp_alert_time = idx.isoformat()

        # --- Vibration Prediction ---
        vib_forecast: pd.Series = self.vib_model.forecast(steps=steps)
        vib_alert = (vib_forecast > FAILURE_THRESHOLD_VIB).any()
        vib_alert_time: Optional[str] = None
        if vib_alert:
            idx = vib_forecast[vib_forecast > FAILURE_THRESHOLD_VIB].index[0]
            vib_alert_time = idx.isoformat()

        # --- Compile Results ---
        alert_status = 'CRITICAL' if temp_alert or vib_alert else 'NORMAL'
        
        result = {
            'prediction_timestamp': pd.Timestamp.now().isoformat(),
            'alert_status': alert_status,
            'forecast_steps': steps,
            'temperature': {
                'arima_order': self.temp_order,
                'forecast': temp_forecast.tolist(),
                'timestamps': temp_forecast.index.isoformat(timespec='minutes'), # Changed to minutes precision
                'alert': temp_alert,
                'threshold': FAILURE_THRESHOLD_TEMP,
                'alert_time': temp_alert_time
            },
            'vibration': {
                'arima_order': self.vib_order,
                'forecast': vib_forecast.tolist(),
                'timestamps': vib_forecast.index.isoformat(timespec='minutes'),
                'alert': vib_alert,
                'threshold': FAILURE_THRESHOLD_VIB,
                'alert_time': vib_alert_time
            }
        }
        logger.info(f"Prediction complete. Status: {alert_status}")
        return result

# --- Testing and Training Execution ---
if __name__ == '__main__':
    import logging.config
    from config import LOGGING_CONFIG
    logging.config.dictConfig(LOGGING_CONFIG)
    
    from data_simulator import IoTSensorDataGenerator
    
    # 1. Load or Generate Data
    data_path = 'data/simulated_iot_data.csv'
    try:
        data_df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
        logger.info(f"Loaded {len(data_df)} records for training.")
    except FileNotFoundError:
        logger.warning("Data not found. Generating new data for training...")
        generator = IoTSensorDataGenerator()
        data_df = generator.generate_data()

    # 2. Train and Save Models
    pipeline = PMFPipeline()
    pipeline.train_all(data_df)

    # 3. Test Prediction
    try:
        prediction_output = pipeline.predict()
        print("\n--- TEST PREDICTION OUTPUT ---")
        print(f"Overall Status: {prediction_output['alert_status']}")
        print(f"Temperature ARIMA Order: {prediction_output['temperature']['arima_order']}")
        print(f"Vibration ARIMA Order: {prediction_output['vibration']['arima_order']}")
        
        if prediction_output['temperature']['alert']:
            print(f"Temp Alert: Predicted to cross {FAILURE_THRESHOLD_TEMP}°C around {prediction_output['temperature']['alert_time']}")
        else:
             print("Temp Alert: None")

    except RuntimeError as e:
        print(f"Prediction failed: {e}")
