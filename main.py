#!/usr/bin/env python3
"""
EcoTrack - Environmental Monitoring System
==================================================

This is the main entry point for the EcoTrack system, which collects, analyzes,
and visualizes environmental data from distributed sensor networks. The system
supports real-time monitoring, anomaly detection, trend prediction, and
community engagement through a web API.

Features:
    - Multi-sensor data collection and integration
    - Automated anomaly detection and alerting
    - Time-series analysis of environmental metrics
    - Predictive modeling for environmental trends
    - Interactive data visualization and dashboards
    - RESTful API for external integration

Usage:
    python main.py --mode collect --sensor-id SENSOR_ID
    python main.py --mode analyze --timeframe 24h
    python main.py --mode predict --timeframe 7d
    python main.py --mode serve

Author: Jeremy Mitts (jeremy.mitts@atlasschool.com)
GitHub: https://github.com/jerm014
LinkedIn: https://www.linkedin.com/in/jeremy-mitts/
Twitter: https://x.com/jermitts
Version: 1.2.0
License: MIT
"""

#-----------------------------------------------------------------------------
# IMPORTS
#-----------------------------------------------------------------------------

# Standard library imports
import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from abc import ABC, abstractmethod  # For abstract base classes
from typing import Dict, List, Optional, Any, Union  # Type hints

# Data processing and analysis libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# EcoTrack modules
from ecotrack.sensors import SensorNetwork, SensorTypes
from ecotrack.data import DataProcessor, TimeSeriesAnalyzer
from ecotrack.api import APIServer
from ecotrack.models import PredictionModel, AnomalyDetector
from ecotrack.utils import ConfigManager, Notifier
from ecotrack.visualization import Dashboard

#-----------------------------------------------------------------------------
# MAIN APPLICATION CLASS
#-----------------------------------------------------------------------------

class EcoTrackApp:
    """
    Main application class for the EcoTrack system.
    
    This class encapsulates the core functionality of the EcoTrack application,
    handling configuration, logging, and operation mode selection.
    
    Attributes:
        config_path (str): Path to the configuration file
        config (Dict): Application configuration dictionary
        logger (logging.Logger): Application logger
        args (argparse.Namespace): Command line arguments
    """
    
    # Default configuration paths
    DEFAULT_CONFIG_PATH = './config/settings.json'
    DEFAULT_LOG_PATH = './logs'
    
    def __init__(self, args: Optional[List[str]] = None):
        """
        Initialize the EcoTrack application.
        
        Args:
            args (List[str], optional): Command line arguments.
                If None, sys.argv is used.
        """
        # Parse command line arguments
        self.args = self._parse_arguments(args)
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Enable debug mode if requested
        if self.args.debug:
            self.logger.setLevel(logging.DEBUG)
            self.logger.debug("Debug mode enabled")
        
        # Load configuration
        self.config_path = self.args.config or os.environ.get(
            'ECOTRACK_CONFIG', self.DEFAULT_CONFIG_PATH)
        self.config = self._load_config()
        self.logger.info(f"Loaded configuration from {self.config_path}")
        
        # Create the appropriate operation handler based on the selected mode
        self.operation_handler = self._create_operation_handler()
    
    def _parse_arguments(self, args: Optional[List[str]]) -> argparse.Namespace:
        """
        Parse command line arguments for the EcoTrack application.
        
        Args:
            args (List[str], optional): Command line arguments.
                If None, sys.argv is used.
            
        Returns:
            argparse.Namespace: Parsed command line arguments
        """
        # Create argument parser with application description
        parser = argparse.ArgumentParser(
            description='EcoTrack Environmental Monitoring System')
        
        # Define command line arguments
        parser.add_argument(
            '--config', type=str, help='Path to configuration file')
        parser.add_argument(
            '--mode',
            type=str,
            choices=['collect', 'analyze', 'predict', 'serve'],
            default='collect',
            help='Operation mode')
        parser.add_argument(
            '--sensor-id', type=str, help='Specific sensor ID to operate on')
        parser.add_argument(
            '--timeframe',
            type=str,
            default='24h',
            help='Timeframe for analysis (e.g. 24h, 7d, 30d)')
        parser.add_argument(
            '--debug', action='store_true', help='Enable debug mode')
        
        # Parse and return arguments
        return parser.parse_args(args)
    
    def _setup_logging(self) -> logging.Logger:
        """
        Configure and set up the logging system for EcoTrack.
        
        Returns:
            logging.Logger: Configured logger instance for the application
        """
        # Get log path from environment variable or use default
        log_path = os.environ.get('ECOTRACK_LOGS', self.DEFAULT_LOG_PATH)
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        
        # Generate a timestamp for the log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"{log_path}/ecotrack_{timestamp}.log"
        
        # Configure logging with file and console output
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger('ecotrack')
    
    def _load_config(self) -> Dict:
        """
        Load application configuration from the specified path.
        
        Returns:
            Dict: Loaded and validated configuration dictionary
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            JSONDecodeError: If the configuration file contains invalid JSON
        """
        # Use ConfigManager to load and validate configuration
        config_manager = ConfigManager(self.config_path)
        return config_manager.load()
    
    def _create_operation_handler(self) -> 'OperationHandler':
        """
        Create the appropriate operation handler based on the command line args.
        
        Returns:
            OperationHandler: An instance of the appropriate handler subclass
        """
        # Select and create the appropriate handler based on operation mode
        if self.args.mode == 'collect':
            return CollectionHandler(self.config, self.logger, self.args.sensor_id)
        elif self.args.mode == 'analyze':
            return AnalysisHandler(
                self.config, self.logger, self.args.timeframe, self.args.sensor_id)
        elif self.args.mode == 'predict':
            return PredictionHandler(
                self.config, self.logger, self.args.timeframe, self.args.sensor_id)
        elif self.args.mode == 'serve':
            return APIHandler(self.config, self.logger)
        else:
            # This should not happen due to argparse choices
            self.logger.error(f"Unknown mode: {self.args.mode}")
            raise ValueError(f"Unknown operation mode: {self.args.mode}")
    
    def run(self) -> None:
        """
        Run the EcoTrack application with the configured operation handler.
        
        Executes the selected operation mode and handles any exceptions.
        
        Returns:
            None
        """
        try:
            # Log start of operation
            self.logger.info(f"Starting EcoTrack in {self.args.mode} mode")
            
            # Execute the selected operation
            self.operation_handler.execute()
            
            # Log completion
            self.logger.info("EcoTrack operation complete")
        except Exception as e:
            # Log any errors that occur during execution
            self.logger.error(f"Error during execution: {str(e)}")
            
            # Print detailed traceback in debug mode
            if self.args.debug:
                self.logger.exception("Detailed traceback:")
            
            # Exit with error code
            sys.exit(1)

#-----------------------------------------------------------------------------
# OPERATION HANDLER BASE CLASS
#-----------------------------------------------------------------------------

class OperationHandler(ABC):
    """
    Abstract base class for operation handlers.
    
    This class defines the interface for all operation handlers in the
    EcoTrack system. Each concrete implementation handles a specific mode.
    
    Attributes:
        config (Dict): Application configuration dictionary
        logger (logging.Logger): Application logger
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """
        Initialize the operation handler with configuration and logger.
        
        Args:
            config (Dict): Application configuration dictionary
            logger (logging.Logger): Application logger
        """
        self.config = config
        self.logger = logger
    
    @abstractmethod
    def execute(self) -> None:
        """
        Execute the operation.
        
        This method must be implemented by all concrete subclasses.
        
        Returns:
            None
        """
        pass  # Abstract method, implementation required in subclasses

#-----------------------------------------------------------------------------
# DATA COLLECTION HANDLER
#-----------------------------------------------------------------------------

class CollectionHandler(OperationHandler):
    """
    Handler for the data collection operation mode.
    
    This handler manages the process of collecting data from sensors,
    processing it, checking for anomalies, and storing results.
    
    Attributes:
        sensor_id (Optional[str]): ID of a specific sensor to collect from,
            or None for all
    """
    
    def __init__(
            self,
            config: Dict,
            logger: logging.Logger,
            sensor_id: Optional[str] = None):
        """
        Initialize the collection handler.
        
        Args:
            config (Dict): Application configuration dictionary
            logger (logging.Logger): Application logger
            sensor_id (Optional[str]): ID of a specific sensor to collect from,
                or None for all
        """
        super().__init__(config, logger)
        self.sensor_id = sensor_id
    
    def execute(self) -> None:
        """
        Execute the data collection operation.
        
        Collects data from sensors, processes it, checks for anomalies, and
        stores the results to disk.
        
        Returns:
            None
        """
        # Initialize the sensor network
        self.logger.info("Initializing sensor network...")
        sensor_network = SensorNetwork(self.config['sensors'])
        
        # Determine which sensors to collect from
        if self.sensor_id:
            self.logger.info(f"Collecting data from sensor {self.sensor_id}")
            sensors = [sensor_network.get_sensor(self.sensor_id)]
        else:
            self.logger.info(f"Collecting data from all sensors")
            sensors = sensor_network.get_all_sensors()
        
        # Initialize data processor
        data_processor = DataProcessor(self.config['data_processing'])
        
        # Process each sensor
        for sensor in sensors:
            try:
                # Read and process data from the sensor
                self.logger.info(
                    f"Reading from {sensor.type} sensor at {sensor.location}")
                raw_data = sensor.read()
                processed_data = data_processor.process(raw_data, sensor.type)
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"{self.config['data_dir']}/{sensor.id}_{timestamp}.json"
                
                # Store the processed data to disk
                with open(filename, 'w') as f:
                    json.dump(processed_data, f)
                
                self.logger.info(
                    f"Data from sensor {sensor.id} stored in {filename}")
                
                # Check for anomalies in the data
                anomaly_detector = AnomalyDetector(
                    self.config['anomaly_detection'])
                anomalies = anomaly_detector.detect(processed_data)
                
                # Send notifications for any detected anomalies
                if anomalies:
                    self.logger.warning(
                        f"Anomalies detected from sensor {sensor.id}: {anomalies}")
                    notifier = Notifier(self.config['notifications'])
                    notifier.send_alert(
                        subject=f"Anomaly detected from sensor {sensor.id}",
                        message=f"The following anomalies were detected: {anomalies}",
                        level="warning"
                    )
            except Exception as e:
                # Log errors but continue processing other sensors
                self.logger.error(
                    f"Error collecting data from sensor {sensor.id}: {str(e)}")

#-----------------------------------------------------------------------------
# TIMEFRAME UTILITY CLASS
#-----------------------------------------------------------------------------

class TimeFrame:
    """
    Utility class to handle timeframe parsing and conversion.
    
    This class provides methods to parse and convert timeframe strings 
    like '24h' or '7d' into datetime ranges and intervals.
    
    Attributes:
        timeframe_str (str): Original timeframe string (e.g., '24h', '7d')
        value (int): Numeric value extracted from the timeframe string
        unit (str): Time unit extracted from the timeframe string ('h' or 'd')
    """
    
    def __init__(self, timeframe_str: str):
        """
        Initialize a TimeFrame object from a timeframe string.
        
        Args:
            timeframe_str (str): Timeframe string in format like '24h', '7d'
            
        Raises:
            ValueError: If the timeframe string format is invalid
        """
        self.timeframe_str = timeframe_str
        # Parse the string into components
        self.value, self.unit = self._parse()
    
    def _parse(self) -> tuple:
        """
        Parse the timeframe string into value and unit components.
        
        Returns:
            tuple: (value, unit) tuple where value is an integer and
                unit is 'h' or 'd'
            
        Raises:
            ValueError: If the timeframe string format is invalid
        """
        # Parse based on the unit suffix
        if self.timeframe_str.endswith('h'):
            # Hours format
            return int(self.timeframe_str[:-1]), 'h'
        elif self.timeframe_str.endswith('d'):
            # Days format
            return int(self.timeframe_str[:-1]), 'd'
        else:
            # Invalid format
            raise ValueError(
                f"Invalid timeframe format: {self.timeframe_str}. "
                f"Expected format like '24h' or '7d'"
            )
    
    def get_time_range(self, end_time: Optional[datetime] = None) -> tuple:
        """
        Calculate the start and end times based on the timeframe.
        
        Args:
            end_time (datetime, optional): End time for the range.
                Defaults to current time.
            
        Returns:
            tuple: (start_time, end_time) tuple with datetime objects
        """
        # Use current time as end_time if not specified
        if end_time is None:
            end_time = datetime.now()
        
        # Calculate start_time based on the unit
        if self.unit == 'h':
            # Hours
            start_time = end_time - pd.Timedelta(hours=self.value)
        elif self.unit == 'd':
            # Days
            start_time = end_time - pd.Timedelta(days=self.value)
        
        return start_time, end_time
    
    def get_hours(self) -> int:
        """
        Convert the timeframe to hours.
        
        Returns:
            int: Number of hours represented by the timeframe
        """
        # Convert to hours based on the unit
        if self.unit == 'h':
            return self.value
        elif self.unit == 'd':
            return self.value * 24
        
    def __str__(self) -> str:
        """
        String representation of the timeframe.
        
        Returns:
            str: Human-readable description of the timeframe
        """
        # Create a human-readable string with proper pluralization
        if self.unit == 'h':
            unit_name = "hour" if self.value == 1 else "hours"
        else:
            unit_name = "day" if self.value == 1 else "days"
        
        return f"{self.value} {unit_name}"

#-----------------------------------------------------------------------------
# DATA ANALYSIS HANDLER
#-----------------------------------------------------------------------------

class AnalysisHandler(OperationHandler):
    """
    Handler for the data analysis operation mode.
    
    This handler manages the process of loading sensor data for a specified
    timeframe, performing time-series analysis, and generating visualizations.
    
    Attributes:
        timeframe (TimeFrame): Time period for analysis
        sensor_id (Optional[str]): ID of a specific sensor to analyze,
            or None for all
    """
    
    def __init__(
            self,
            config: Dict,
            logger: logging.Logger,
            timeframe_str: str,
            sensor_id: Optional[str] = None):
        """
        Initialize the analysis handler.
        
        Args:
            config (Dict): Application configuration dictionary
            logger (logging.Logger): Application logger
            timeframe_str (str): Time period for analysis in format '24h', '7d'
            sensor_id (Optional[str]): ID of a specific sensor to analyze,
                or None for all
        """
        super().__init__(config, logger)
        # Convert the timeframe string to a TimeFrame object
        self.timeframe = TimeFrame(timeframe_str)
        self.sensor_id = sensor_id
    
    def execute(self) -> None:
        """
        Execute the data analysis operation.
        
        Loads sensor data for the specified timeframe, performs analysis,
        generates visualizations, and saves results to disk.
        
        Returns:
            None
        """
        self.logger.info(f"Analyzing data for timeframe: {self.timeframe}")
        
        # Initialize analyzer with configuration
        analyzer = TimeSeriesAnalyzer(self.config['analysis'])
        
        # Calculate time range for data selection
        start_time, end_time = self.timeframe.get_time_range()
        
        # Get list of data files
        data_dir = self.config['data_dir']
        data_files = os.listdir(data_dir)
        
        # Filter files for specific sensor if requested
        if self.sensor_id:
            data_files = [f for f in data_files 
                         if f.startswith(f"{self.sensor_id}_")]
        
        # Check if any files are available
        if not data_files:
            self.logger.warning("No data files found for analysis")
            return
        
        # Load data from files within the specified timeframe
        all_data = self._load_data_files(
            data_files, data_dir, start_time, end_time)
        
        # Check if any data was found in the timeframe
        if not all_data:
            self.logger.warning("No data found within the specified timeframe")
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_data)
        
        # Perform time-series analysis
        results = analyzer.analyze(df)
        
        # Generate visualizations and save results
        self._save_analysis_results(results)
    
    def _load_data_files(
            self,
            data_files: List[str],
            data_dir: str,
            start_time: datetime,
            end_time: datetime) -> List[Dict]:
        """
        Load and filter data files based on the specified time range.
        
        Args:
            data_files (List[str]): List of data file names to process
            data_dir (str): Directory containing the data files
            start_time (datetime): Start of the time range to include
            end_time (datetime): End of the time range to include
            
        Returns:
            List[Dict]: List of data dictionaries within the time range
        """
        all_data = []
        # Process each data file
        for file in data_files:
            file_path = os.path.join(data_dir, file)
            try:
                # Read the data file
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Extract timestamp from filename (format: sensorID_YYYYMMDDHHMMSS)
                timestamp_str = file.split('_')[1].split('.')[0]
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                
                # Include data if timestamp is within range
                if start_time <= timestamp <= end_time:
                    data['timestamp'] = timestamp.isoformat()
                    all_data.append(data)
            except Exception as e:
                # Log errors but continue processing other files
                self.logger.error(
                    f"Error reading data file {file}: {str(e)}")
        
        return all_data
    
    def _save_analysis_results(self, results: Dict) -> None:
        """
        Generate visualizations and save analysis results to disk.
        
        Args:
            results (Dict): Analysis results from the TimeSeriesAnalyzer
            
        Returns:
            None
        """
        # Generate visualization charts
        dashboard = Dashboard(self.config['visualization'])
        figures = dashboard.create_charts(results)
        
        # Create output directory if it doesn't exist
        analysis_dir = os.path.join(self.config['output_dir'], 'analysis')
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
        
        # Generate timestamp for output files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save analysis results as JSON
        output_file = os.path.join(
            analysis_dir, f"analysis_{timestamp}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f)
        
        # Save visualization figures as images
        for i, fig in enumerate(figures):
            fig_path = os.path.join(
                analysis_dir, f"figure_{i}_{timestamp}.png")
            fig.savefig(fig_path)
        
        self.logger.info(f"Analysis complete. Results saved to {output_file}")

#-----------------------------------------------------------------------------
# PREDICTION HANDLER
#-----------------------------------------------------------------------------

class PredictionHandler(OperationHandler):
    """
    Handler for the trend prediction operation mode.
    
    This handler manages the process of training predictive models on historical
    data and generating forecasts for future environmental conditions.
    
    Attributes:
        timeframe (TimeFrame): Time period for prediction
        sensor_id (Optional[str]): ID of a specific sensor to predict for,
            or None for all
    """
    
    def __init__(
            self,
            config: Dict,
            logger: logging.Logger,
            timeframe_str: str,
            sensor_id: Optional[str] = None):
        """
        Initialize the prediction handler.
        
        Args:
            config (Dict): Application configuration dictionary
            logger (logging.Logger): Application logger
            timeframe_str (str): Time period for prediction (e.g. '24h', '7d')
            sensor_id (Optional[str]): ID of a specific sensor to predict for,
                or None for all
        """
        super().__init__(config, logger)
        # Convert timeframe string to TimeFrame object
        self.timeframe = TimeFrame(timeframe_str)
        self.sensor_id = sensor_id
    
    def execute(self) -> None:
        """
        Execute the trend prediction operation.
        
        Loads historical data, trains or loads predictive models, generates
        forecasts, and saves results to disk.
        
        Returns:
            None
        """
        # Calculate prediction horizon in hours
        prediction_horizon = self.timeframe.get_hours()
        self.logger.info(
            f"Generating predictions for next {prediction_horizon} hours")
        
        # Initialize prediction model with configuration
        prediction_model = PredictionModel(self.config['prediction'])
        
        # Load historical data for training
        training_data = self._load_training_data()
        
        # Train the model if needed or if retraining is configured
        if not prediction_model.is_trained() or self.config['prediction'].get(
                'retrain', False):
            self.logger.info("Training prediction model on historical data")
            prediction_model.train(training_data)
        
        # Prepare input data for prediction (most recent data)
        input_data = self._prepare_input_data()
        
        # Generate predictions for the specified horizon
        predictions = prediction_model.predict(input_data, prediction_horizon)
        
        # Save prediction results
        self._save_predictions(predictions)
    
    def _load_training_data(self) -> pd.DataFrame:
        """
        Load historical data for training the predictive model.
        
        This is a placeholder method - in a real implementation, this would
        load and prepare actual historical data from the system.
        
        Returns:
            pd.DataFrame: DataFrame containing historical data for training
        """
        # Note: This is a placeholder implementation
        self.logger.info("Loading historical data for model training")
        
        # Use a longer timeframe for training data (e.g., past 30 days)
        training_timeframe = TimeFrame('30d')
        start_time, end_time = training_timeframe.get_time_range()
        
        # In a real implementation, this would load actual historical data
        # based on the calculated time range and sensor ID
        
        # Return empty DataFrame as placeholder
        return pd.DataFrame()
    
    def _prepare_input_data(self) -> pd.DataFrame:
        """
        Prepare the most recent data as input for the prediction model.
        
        This is a placeholder method - in a real implementation, this would
        load and prepare the most recent data as input for prediction.
        
        Returns:
            pd.DataFrame: DataFrame with recent data for prediction input
        """
        # Note: This is a placeholder implementation
        self.logger.info("Preparing recent data as input for prediction")
        
        # In a real implementation, this would load the most recent data
        # from sensors for use as prediction input
        
        # Return empty DataFrame as placeholder
        return pd.DataFrame()
    
    def _save_predictions(self, predictions: Dict) -> None:
        """
        Save prediction results to disk.
        
        Args:
            predictions (Dict): Prediction results from the model
            
        Returns:
            None
        """
        # Create predictions directory if it doesn't exist
        prediction_dir = os.path.join(self.config['output_dir'], 'predictions')
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
        
        # Generate timestamp for output file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(
            prediction_dir, f"prediction_{timestamp}.json")
        
        # Save prediction results as JSON
        with open(output_file, 'w') as f:
            json.dump(predictions, f)
        
        self.logger.info(
            f"Prediction complete. Results saved to {output_file}")

#-----------------------------------------------------------------------------
# API SERVER HANDLER
#-----------------------------------------------------------------------------

class APIHandler(OperationHandler):
    """
    Handler for the API server operation mode.
    
    This handler manages the process of starting and running the RESTful API
    server that provides external access to EcoTrack data and functionality.
    """
    
    def execute(self) -> None:
        """
        Execute the API server operation.
        
        Initializes and starts the RESTful API server, which runs until
        interrupted.
        
        Returns:
            None
            
        Raises:
            RuntimeError: If the server fails to start or encounters an error
        """
        self.logger.info("Starting API server")
        
        try:
            # Extract API configuration
            api_config = self.config['api']
            host = api_config.get('host', '0.0.0.0')  # Default host
            port = api_config.get('port', 8000)       # Default port
            
            self.logger.info(f"API server will listen on {host}:{port}")
            
            # Initialize and start the API server
            api_server = APIServer(api_config)
            # This will typically block until the server is stopped
            api_server.start()
        except Exception as e:
            # Log error and re-raise as RuntimeError
            self.logger.error(f"Failed to start API server: {str(e)}")
            raise RuntimeError(f"API server startup failed: {str(e)}")

#-----------------------------------------------------------------------------
# MAIN ENTRY POINT
#-----------------------------------------------------------------------------

def main():
    """
    Main entry point for the EcoTrack application.
    
    Creates and runs an instance of the EcoTrackApp class.
    
    Returns:
        None
    """
    try:
        # Create and run the application
        app = EcoTrackApp()
        app.run()
    except KeyboardInterrupt:
        # Handle clean exit on Ctrl+C
        print("\nEcoTrack application terminated by user")
        sys.exit(0)
    except Exception as e:
        # Handle unexpected errors
        print(f"Error: {str(e)}")
        sys.exit(1)

# Run the application if this script is executed directly
if __name__ == "__main__":
    main()
