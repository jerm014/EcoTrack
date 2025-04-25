#!/usr/bin/env python3

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

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
    parser = argparse.ArgumentParser(description='EcoTrack Environmental Monitoring System')
    parser.add_argument('--config', type=str, default=CONFIG_PATH, help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['collect', 'analyze', 'predict', 'serve'], 
                      default='collect', help='Operation mode')
    parser.add_argument('--sensor-id', type=str, help='Specific sensor ID to operate on')
    parser.add_argument('--timeframe', type=str, default='24h', help='Timeframe for analysis (e.g. 24h, 7d, 30d)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    return parser.parse_args()

def load_config(config_path):
    config_manager = ConfigManager(config_path)
    return config_manager.load()

def collect_sensor_data(config, sensor_id=None):
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
    logger.info(f"Generating predictions for timeframe: {timeframe}")
    
    # Similar to analyze_data but uses the prediction model
    prediction_model = PredictionModel(config['prediction'])
    
    # Load historical data (similar to analyze_data)
    # ...code to load and prepare data...
    
    # Train the model if needed
    prediction_model.train(training_data)
    
    # Generate predictions
    predictions = prediction_model.predict(input_data, prediction_horizon)
    
    # Save predictions
    # ...code to save predictions...
    
    logger.info("Prediction complete")

def serve_api(config):
    logger.info("Starting API server")
    api_server = APIServer(config['api'])
    api_server.start()

def main():
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
