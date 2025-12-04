import os
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import logging
from config import START_DATE, SIMULATION_DAYS, SAMPLE_FREQUENCY_MINUTES, FAILURE_DAY_OFFSET, DATA_SAVE_DIR

# Set up logging for this module
logger = logging.getLogger(__name__)

class IoTSensorDataGenerator:
    """
    Generates synthetic multivariate time-series data for two sensors,
    simulating normal operation and equipment degradation/failure.
    """
    def __init__(self, start_date: datetime = START_DATE, simulation_days: int = SIMULATION_DAYS, sample_frequency_minutes: int = SAMPLE_FREQUENCY_MINUTES):
        self.start_date = start_date
        self.simulation_days = simulation_days
        self.sample_frequency_minutes = sample_frequency_minutes
        
        self.n_samples = int((simulation_days * 24 * 60) / sample_frequency_minutes)
        self.dates = [self.start_date + timedelta(minutes=i * self.sample_frequency_minutes)
                      for i in range(self.n_samples)]
        
        failure_offset_minutes = FAILURE_DAY_OFFSET * 24 * 60
        self.failure_sample_index = int(failure_offset_minutes / self.sample_frequency_minutes)
        
        logger.info(f"Generator initialized for {self.n_samples} samples.")

    def _generate_sensor_series(self, base_value: float, trend_slope: float, noise_scale: float, failure_multiplier: float, seasonality_amp: float) -> np.ndarray:
        """Helper to generate a single sensor time series."""
        series = np.full(self.n_samples, base_value)

        # 1. Base trend (linear degradation)
        linear_trend = np.linspace(0, trend_slope * self.simulation_days, self.n_samples)
        series += linear_trend

        # 2. Daily/Cyclical seasonality
        minutes_in_day = 24 * 60
        time_vector = np.array([d.hour * 60 + d.minute for d in self.dates])
        cycle = np.sin(2 * np.pi * time_vector / minutes_in_day) * seasonality_amp
        series += cycle

        # 3. Random Noise
        noise = np.random.normal(0, noise_scale, self.n_samples)
        series += noise

        # 4. Failure Mode Simulation (Exponential increase post-offset)
        failure_indices = np.arange(self.failure_sample_index, self.n_samples)
        failure_span = self.n_samples - self.failure_sample_index
        if failure_span > 0:
            # Ensure failure_increase starts from the base_value at the offset point
            failure_increase = (
                np.exp(np.linspace(0, np.log(failure_multiplier), failure_span)) * base_value
            )
            # Additive effect starts from 0 at the offset point
            additive_failure = failure_increase - base_value
            series[self.failure_sample_index:] += additive_failure
            
        return series

    def generate_data(self, device_id: str = "IoT-Pump-001") -> pd.DataFrame:
        """Generates the full DataFrame with Temperature and Vibration."""
        temp_series = self._generate_sensor_series(
            base_value=30.0, trend_slope=0.05, noise_scale=1.0,
            failure_multiplier=1.5, seasonality_amp=2.0
        )
        vib_series = self._generate_sensor_series(
            base_value=5.0, trend_slope=0.01, noise_scale=0.5,
            failure_multiplier=3.0, seasonality_amp=1.0
        )

        df = pd.DataFrame({
            'timestamp': self.dates,
            'device_id': device_id,
            'temperature': temp_series,
            'vibration': vib_series
        })
        df.set_index('timestamp', inplace=True)
        logger.info(f"Data generated for device {device_id}. Total rows: {len(df)}")
        return df

    def save_data(self, df: pd.DataFrame, filename: str = 'simulated_iot_data.csv') -> str:
        """Saves the generated DataFrame to a CSV file."""
        path = os.path.join(DATA_SAVE_DIR, filename)
        df.to_csv(path)
        logger.info(f"Data saved successfully to {path}")
        return path
    
    def generate_and_save(self, device_id: str = "IoT-Pump-001", filename: str = 'simulated_iot_data.csv') -> str:
        """Convenience method to generate and save data."""
        df = self.generate_data(device_id=device_id)
        return self.save_data(df, filename=filename)


if __name__ == '__main__':
    # Initialize logging
    import logging.config
    from config import LOGGING_CONFIG
    logging.config.dictConfig(LOGGING_CONFIG)
    
    logger.info("Starting Data Simulation Module.")
    
    # Generate and save the historical dataset
    generator = IoTSensorDataGenerator()
    data_path = generator.generate_and_save(device_id="IoT-Pump-001")

    # Load and print sample
    historical_df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
    print("\n--- Sample of Generated Data (Failure Approach) ---")
    print(historical_df.tail(5))
    logger.info(f"Total rows generated: {len(historical_df)}")
