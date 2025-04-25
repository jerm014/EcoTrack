#!/usr/bin/env python3
"""
EcoTrack - Environmental Monitoring System
==================================================

This is the main entry point for the EcoTrack system, which collects, analyzes, and
visualizes environmental data from distributed sensor networks. The system supports
real-time monitoring, anomaly detection, trend prediction, and community engagement
through a web API.

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

# Standard library imports
import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime

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

# Global configuration
CONFIG_PATH = os.environ.get('ECOTRACK_CONFIG', './config/settings.json')
LOG_PATH = os.environ.get('ECOTRACK_LOGS', './logs')

def setup_logging():
    """
    Configure and set up the logging system for EcoTrack.
    
    Creates the log directory if it doesn't exist and configures logging to write to both
    a timestamped log file and stdout. The log format includes timestamp, logger name,
    level, and message.
    
    Returns:
        logging.Logger: Configured logger instance for the application
        
    Example:
        >>> logger = setup_logging()
        >>> logger.info("EcoTrack system initialized")
    """
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{LOG_PATH}/ecotrack_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('ecotrack')

def parse_arguments():
    """
    Parse command line arguments for the EcoTrack application.
    
    Defines and processes the following command line arguments:
        --config: Path to the configuration file
        --mode: Operation mode (collect, analyze, predict, serve)
        --sensor-id: ID of a specific sensor to operate on
        --timeframe: Time period for analysis or prediction
        --debug: Flag to enable debug logging
    
    Returns:
        argparse.Namespace: Parsed command line arguments
        
    Example:
        >>> args = parse_arguments()
        >>> if args.debug:
        ...     logger.setLevel(logging.DEBUG)
    """
    parser = argparse.ArgumentParser(description='EcoTrack Environmental Monitoring System')
    parser.add_argument('--config', type=str, default=CONFIG_PATH, help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['collect', 'analyze', 'predict', 'serve'], 
                      default='collect', help='Operation mode')
    parser.add_argument('--sensor-id', type=str, help='Specific sensor ID to operate on')
    parser.add_argument('--timeframe', type=str, default='24h', help='Timeframe for analysis (e.g. 24h, 7d, 30d)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    return parser.parse_args()

def load_config(config_path):
    """
    Load application configuration from the specified path.
    
    Uses the ConfigManager utility to parse and validate the configuration file.
    The configuration includes settings for sensors, data processing, analysis,
    visualization, API, and notification services.
    
    Args:
        config_path (str): Path to the configuration file (JSON format)
        
    Returns:
        dict: Loaded and validated configuration dictionary
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        JSONDecodeError: If the configuration file contains invalid JSON
        ValueError: If the configuration is invalid or missing required sections
        
    Example:
        >>> config = load_config('./config/settings.json')
        >>> sensor_configs = config['sensors']
    """
    config_manager = ConfigManager(config_path)
    return config_manager.load()

def collect_sensor_data(config, sensor_id=None):
    """
    Collect and process data from sensors in the network.
    
    This function connects to the sensor network, retrieves data from either a specific
    sensor or all available sensors, processes the raw data, saves it to disk, and
    performs anomaly detection. If anomalies are detected, alerts are sent through
    the notification system.
    
    Args:
        config (dict): Application configuration dictionary containing sensor settings,
                      data processing parameters, and notification options
        sensor_id (str, optional): ID of a specific sensor to collect data from.
                                  If None, collects from all sensors.
    
    Raises:
        Exception: Logs any errors during sensor data collection but doesn't propagate them
        
    Example:
        >>> collect_sensor_data(config, 'air-quality-001')  # Collect from specific sensor
        >>> collect_sensor_data(config)  # Collect from all sensors
    """
    logger.info("Initializing sensor network...")
    sensor_network = SensorNetwork(config['sensors'])
    
    if sensor_id:
        logger.info(f"Collecting data from sensor {sensor_id}")
        sensors = [sensor_network.get_sensor(sensor_id)]
    else:
        logger.info(f"Collecting data from all sensors")
        sensors = sensor_network.get_all_sensors()
    
    data_processor = DataProcessor(config['data_processing'])
    
    for sensor in sensors:
        try:
            logger.info(f"Reading from {sensor.type} sensor at {sensor.location}")
            raw_data = sensor.read()
            processed_data = data_processor.process(raw_data, sensor.type)
            
            # Store the processed data
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{config['data_dir']}/{sensor.id}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(processed_data, f)
            
            logger.info(f"Data from sensor {sensor.id} stored in {filename}")
            
            # Check for anomalies
            anomaly_detector = AnomalyDetector(config['anomaly_detection'])
            anomalies = anomaly_detector.detect(processed_data)
            
            if anomalies:
                logger.warning(f"Anomalies detected from sensor {sensor.id}: {anomalies}")
                notifier = Notifier(config['notifications'])
                notifier.send_alert(
                    subject=f"Anomaly detected from sensor {sensor.id}",
                    message=f"The following anomalies were detected: {anomalies}",
                    level="warning"
                )
        except Exception as e:
            logger.error(f"Error collecting data from sensor {sensor.id}: {str(e)}")

def analyze_data(config, timeframe, sensor_id=None):
    """
    Analyze environmental data over a specified timeframe.
    
    This function loads sensor data from the specified timeframe, performs time-series
    analysis, generates visualizations, and saves the results to the output directory.
    It can focus on data from a specific sensor or analyze data from all sensors.
    
    Args:
        config (dict): Application configuration dictionary
        timeframe (str): Time period for analysis in format '24h', '7d', etc.
        sensor_id (str, optional): ID of a specific sensor to analyze.
                                  If None, analyzes data from all sensors.
    
    Returns:
        None: Results are saved to disk in the specified output directory
        
    Raises:
        ValueError: If the timeframe format is invalid
        
    Example:
        >>> analyze_data(config, '24h', 'water-quality-003')  # Last 24 hours for specific sensor
        >>> analyze_data(config, '7d')  # Last 7 days for all sensors
    """
    logger.info(f"Analyzing data for timeframe: {timeframe}")
    
    analyzer = TimeSeriesAnalyzer(config['analysis'])
    
    # Determine time range from timeframe string
    end_time = datetime.now()
    if timeframe.endswith('h'):
        hours = int(timeframe[:-1])
        start_time = end_time - pd.Timedelta(hours=hours)
    elif timeframe.endswith('d'):
        days = int(timeframe[:-1])
        start_time = end_time - pd.Timedelta(days=days)
    else:
        logger.error(f"Invalid timeframe format: {timeframe}")
        return
    
    # Load the data
    data_dir = config['data_dir']
    data_files = os.listdir(data_dir)
    
    if sensor_id:
        data_files = [f for f in data_files if f.startswith(f"{sensor_id}_")]
    
    if not data_files:
        logger.warning("No data files found for analysis")
        return
    
    all_data = []
    for file in data_files:
        file_path = os.path.join(data_dir, file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract timestamp from filename
            timestamp_str = file.split('_')[1].split('.')[0]
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
            
            if start_time <= timestamp <= end_time:
                data['timestamp'] = timestamp.isoformat()
                all_data.append(data)
        except Exception as e:
            logger.error(f"Error reading data file {file}: {str(e)}")
    
    if not all_data:
        logger.warning("No data found within the specified timeframe")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_data)
    
    # Perform analysis
    results = analyzer.analyze(df)
    
    # Generate visualizations
    dashboard = Dashboard(config['visualization'])
    figures = dashboard.create_charts(results)
    
    # Save results and figures
    analysis_dir = os.path.join(config['output_dir'], 'analysis')
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    # Save analysis results
    output_file = os.path.join(analysis_dir, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f)
    
    # Save figures
    for i, fig in enumerate(figures):
        fig_path = os.path.join(analysis_dir, f"figure_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(fig_path)
    
    logger.info(f"Analysis complete. Results saved to {output_file}")

def predict_trends(config, timeframe, sensor_id=None):
    """
    Generate predictive models and forecasts for environmental trends.
    
    Uses machine learning models to predict future environmental metrics based on
    historical data. Predictions can be made for specific sensors or across the
    entire sensor network. Models are trained on historical data and can be saved
    for future use.
    
    Args:
        config (dict): Application configuration dictionary
        timeframe (str): Time period for prediction in format '24h', '7d', etc.,
                        indicating how far into the future to predict
        sensor_id (str, optional): ID of a specific sensor to generate predictions for.
                                  If None, makes predictions for all sensors.
    
    Returns:
        None: Prediction results are saved to disk in the specified output directory
        
    Raises:
        ValueError: If insufficient data is available for training the model
        ValueError: If the timeframe format is invalid
        
    Example:
        >>> predict_trends(config, '7d', 'air-quality-001')  # Predict next 7 days for specific sensor
        >>> predict_trends(config, '30d')  # Predict next 30 days for all sensors
    """
    logger.info(f"Generating predictions for timeframe: {timeframe}")
    
    # Parse prediction horizon
    prediction_horizon = None
    if timeframe.endswith('h'):
        prediction_horizon = int(timeframe[:-1])
    elif timeframe.endswith('d'):
        prediction_horizon = int(timeframe[:-1]) * 24  # Convert days to hours
    else:
        logger.error(f"Invalid timeframe format: {timeframe}")
        return
    
    # Initialize prediction model
    prediction_model = PredictionModel(config['prediction'])
    
    # Load historical data for training
    # (This is placeholder code - in a real implementation, we would load and prepare data)
    training_data = pd.DataFrame()  # Placeholder
    input_data = pd.DataFrame()  # Placeholder
    
    # Train the model if needed
    if not prediction_model.is_trained() or config['prediction'].get('retrain', False):
        logger.info("Training prediction model on historical data")
        prediction_model.train(training_data)
    
    # Generate predictions
    logger.info(f"Generating predictions for next {prediction_horizon} hours")
    predictions = prediction_model.predict(input_data, prediction_horizon)
    
    # Save predictions to disk
    prediction_dir = os.path.join(config['output_dir'], 'predictions')
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    
    output_file = os.path.join(prediction_dir, f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(predictions, f)
    
    logger.info(f"Prediction complete. Results saved to {output_file}")

def serve_api(config):
    """
    Start the RESTful API server for external integrations.
    
    This function initializes and runs the API server that provides access to
    EcoTrack data, analysis, and predictions. The server handles authentication,
    request routing, and data formatting for client applications.
    
    Args:
        config (dict): Application configuration dictionary containing API settings
                      such as host, port, authentication, and endpoint configurations
    
    Returns:
        None: This function runs until interrupted or the server is shut down
        
    Raises:
        RuntimeError: If the server fails to start or encounters a critical error
        
    Example:
        >>> serve_api(config)  # Start the API server
    """
    logger.info("Starting API server")
    
    try:
        api_server = APIServer(config['api'])
        logger.info(f"API server listening on {config['api'].get('host', '0.0.0.0')}:{config['api'].get('port', 8000)}")
        api_server.start()
    except Exception as e:
        logger.error(f"Failed to start API server: {str(e)}")
        raise RuntimeError(f"API server startup failed: {str(e)}")

def main():
    """
    Main entry point for the EcoTrack application.
    
    This function orchestrates the overall flow of the application:
    1. Parse command line arguments
    2. Set up logging
    3. Load configuration
    4. Execute the requested operation mode (collect, analyze, predict, or serve)
    
    The function handles the entire application lifecycle and exits appropriately
    based on success or failure of operations.
    
    Raises:
        SystemExit: With error code 1 if an invalid mode is specified
        
    Example:
        >>> if __name__ == "__main__":
        ...     main()
    """
    args = parse_arguments()
    
    global logger
    logger = setup_logging()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    if args.mode == 'collect':
        collect_sensor_data(config, args.sensor_id)
    elif args.mode == 'analyze':
        analyze_data(config, args.timeframe, args.sensor_id)
    elif args.mode == 'predict':
        predict_trends(config, args.timeframe, args.sensor_id)
    elif args.mode == 'serve':
        serve_api(config)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)
    
    logger.info("EcoTrack operation complete")

if __name__ == "__main__":
    main()
