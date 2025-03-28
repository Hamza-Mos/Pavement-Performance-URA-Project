"""
Temperature and Weather Data Aggregator - Hourly Interval
---------------------------------------------------------
This script processes 15-minute aggregated temperature and weather data,
further aggregates it to hourly intervals, and saves the results to CSV files.

The script:
1. Reads 15-minute aggregated data CSV files
2. Processes the data to create hourly aggregates
3. Preserves the full temperature range by using min of mins and max of maxes
4. Calculates statistics (mean, min, max, std dev, count)
5. Writes the aggregated data to output CSV files
6. Updates a status file to track the last processed timestamp

Usage:
    python temp_weather_hourly.py [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR]

Dependencies:
    - pandas
    - numpy
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import json
import glob
import math

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/temp_weather_hourly.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
STATUS_FILE = "../processed/aggregation_status.json"

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process 15-minute data to hourly intervals')
    parser.add_argument('--input-dir', type=str, default='./output', help='Directory containing 15-minute data files')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory for output files')
    return parser.parse_args()

def get_last_processed_time():
    """Read the last processed timestamp from the status file."""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                status = json.load(f)
                return pd.to_datetime(status.get('last_processed_hourly', '2000-01-01 00:00:00'))
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error reading status file: {e}")
    
    # Default to a date far in the past if no status file exists
    return pd.to_datetime('2000-01-01 00:00:00')

def update_last_processed_time(timestamp):
    """Update the status file with the latest processed timestamp."""
    status = {}
    
    # Read existing status if file exists
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                status = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Could not read status file, creating new one")
    
    # Update the hourly timestamp
    status['last_processed_hourly'] = timestamp.isoformat()
    
    # Write updated status back to file
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)
    
    logger.info(f"Updated last processed time to {timestamp}")

def read_15min_data(input_dir, data_type, last_processed_time):
    """
    Read 15-minute aggregated data files.
    
    Args:
        input_dir: Directory containing the input files
        data_type: Type of data ('temperature' or 'weather')
        last_processed_time: Timestamp of the last processed data (for logging only)
        
    Returns:
        DataFrame with the 15-minute data
    """
    logger.info(f"Reading {data_type} 15-minute data files")
    logger.info(f"Processing all data regardless of last processed time")
    
    # Find all CSV files for the data type
    file_pattern = os.path.join(input_dir, f"{data_type}_15min_*.csv")
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        logger.warning(f"No {data_type} 15-minute data files found")
        return pd.DataFrame()
    
    # Read and concatenate all files
    dfs = []
    for file in files:
        try:
            logger.info(f"Reading file: {file}")
            df = pd.read_csv(file)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Process all data regardless of timestamp
            if not df.empty:
                dfs.append(df)
                logger.info(f"Found {len(df)} rows of data in file")
        except Exception as e:
            logger.error(f"Error reading file {file}: {e}")
    
    if not dfs:
        logger.info(f"No {data_type} data to process")
        return pd.DataFrame()
    
    # Combine all data
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates (in case files overlap)
    combined_df = combined_df.drop_duplicates(subset=['timestamp', 'sensor_id'])
    
    # Sort by timestamp and sensor
    combined_df = combined_df.sort_values(['timestamp', 'sensor_id'])
    
    logger.info(f"Total {data_type} data to process: {len(combined_df)} rows")
    logger.info(f"Time range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    
    return combined_df

def calculate_aggregated_std_dev(groups):
    """
    Calculate the correct standard deviation when aggregating data.
    
    Parameters:
    groups - List of dictionaries with 'mean', 'std', and 'count' for each group
    
    Returns:
    Aggregated standard deviation
    """
    if not groups:
        return 0
        
    # 1. Calculate the overall combined mean (weighted by counts)
    total_count = sum(group['count'] for group in groups)
    if total_count == 0:
        return 0
        
    combined_mean = sum(group['mean'] * group['count'] for group in groups) / total_count
    
    # 2. Calculate combined variance using the formula:
    # combined_variance = (sum of (count * variance) + sum of (count * (group_mean - combined_mean)²)) / total_count
    
    # First part: sum of (count * variance)
    sum_weighted_variance = sum(group['count'] * (group['std']**2) for group in groups)
    
    # Second part: sum of (count * (group_mean - combined_mean)²)
    sum_weighted_mean_diff_squared = sum(group['count'] * (group['mean'] - combined_mean)**2 for group in groups)
    
    # Combined variance
    combined_variance = (sum_weighted_variance + sum_weighted_mean_diff_squared) / total_count
    
    # 3. Take square root to get standard deviation
    return math.sqrt(combined_variance)

def aggregate_to_hourly(df):
    """
    Aggregate 15-minute data to hourly intervals.
    
    Args:
        df: DataFrame with 15-minute aggregated data
        
    Returns:
        DataFrame with hourly aggregated data
    """
    logger.info("Aggregating data to hourly intervals")
    
    if df.empty:
        logger.warning("No data to aggregate")
        return pd.DataFrame()
    
    # Set timestamp as index
    df = df.set_index('timestamp')
    
    # Create a DataFrame to hold the aggregated results
    results = []
    
    # Process each sensor
    for sensor_id, sensor_data in df.groupby('sensor_id'):
        logger.info(f"Processing sensor: {sensor_id}")
        
        # Group by hour
        hourly_groups = sensor_data.groupby(pd.Grouper(freq='1H'))
        
        hourly_results = []
        for hour, hour_data in hourly_groups:
            if not hour_data.empty:
                # Create groups for std dev calculation
                groups = [
                    {'mean': row['mean'], 'std': row['std'], 'count': row['count']} 
                    for idx, row in hour_data.iterrows()
                ]
                
                hourly_results.append({
                    'timestamp': hour,
                    'mean': hour_data['mean'].mean(),
                    'min': hour_data['min'].min(),
                    'max': hour_data['max'].max(),
                    'std': calculate_aggregated_std_dev(groups),
                    'count': hour_data['count'].sum(),
                    'sensor_id': sensor_id
                })
        
        if hourly_results:
            results.append(pd.DataFrame(hourly_results))
    
    # Combine all results
    if not results:
        logger.warning("No data to aggregate")
        return pd.DataFrame()
    
    aggregated = pd.concat(results, ignore_index=True)
    
    # Sort by timestamp and sensor
    aggregated = aggregated.sort_values(['timestamp', 'sensor_id'])
    
    return aggregated

def save_aggregated_data(df, output_dir, data_type):
    """
    Save aggregated data to a CSV file.
    
    Args:
        df: DataFrame with aggregated data
        output_dir: Directory to save the output file
        data_type: Type of data ('temperature' or 'weather')
    """
    if df.empty:
        logger.warning(f"No {data_type} data to save")
        return
    
    # Filter out rows with null values or zero count to avoid propagating bad data
    valid_data = df.copy()
    
    # Replace empty strings with NaN
    for col in ['mean', 'min', 'max', 'std']:
        if col in valid_data.columns:
            valid_data[col] = pd.to_numeric(valid_data[col], errors='coerce')
    
    # Filter rows where essential values are not null and count > 0
    valid_data = valid_data.dropna(subset=['mean', 'min', 'max'])
    valid_data = valid_data[valid_data['count'] > 0]
    
    # More aggressive filtering: exclude rows where all values are exactly zero or very close to zero
    zero_threshold = 0.001  # Values below this are considered effectively zero
    valid_data = valid_data[
        (abs(valid_data['mean']) > zero_threshold) | 
        (abs(valid_data['min']) > zero_threshold) | 
        (abs(valid_data['max']) > zero_threshold)
    ]
    
    if valid_data.empty:
        logger.warning(f"No valid {data_type} data to save after filtering out null values and zeros")
        return
    
    logger.info(f"Filtered out {len(df) - len(valid_data)} rows with null values, zeros, or invalid data")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate filename with current date
    current_date = datetime.now().strftime('%Y%m%d')
    output_file = os.path.join(output_dir, f"{data_type}_hourly_{current_date}.csv")
    
    # Check if file exists
    file_exists = os.path.isfile(output_file)
    
    # Write to CSV
    mode = 'a' if file_exists else 'w'
    header = not file_exists
    
    valid_data.to_csv(output_file, mode=mode, index=False, header=header)
    
    logger.info(f"Saved {len(valid_data)} valid {data_type} data rows to {output_file}")

def main():
    """Main processing function."""
    args = parse_arguments()
    
    # Input and output directories
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # Get last processed time
    last_processed_time = get_last_processed_time()
    logger.info(f"Last processed time: {last_processed_time}")
    
    # Calculate the end time for processing
    # Process up to the last complete hour
    end_time = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    
    logger.info(f"Processing data up to: {end_time}")
    
    # Read and process temperature data
    temp_df = read_15min_data(input_dir, 'temperature', last_processed_time)
    
    # Filter data to process only complete hours
    if not temp_df.empty:
        temp_df = temp_df[temp_df['timestamp'] <= end_time]
        
        if not temp_df.empty:
            hourly_temp = aggregate_to_hourly(temp_df)
            # Save to hourly subdirectory
            save_aggregated_data(hourly_temp, os.path.join(output_dir, 'hourly'), 'temperature')
        else:
            logger.info("No complete hours to process for temperature data")
    
    # Read and process weather data
    weather_df = read_15min_data(input_dir, 'weather', last_processed_time)
    
    # Filter data to process only complete hours
    if not weather_df.empty:
        weather_df = weather_df[weather_df['timestamp'] <= end_time]
        
        if not weather_df.empty:
            hourly_weather = aggregate_to_hourly(weather_df)
            # Save to hourly subdirectory
            save_aggregated_data(hourly_weather, os.path.join(output_dir, 'hourly'), 'weather')
        else:
            logger.info("No complete hours to process for weather data")
    
    # Update the last processed time to the end of the processing window
    # Only update if we actually processed data
    if (not temp_df.empty and temp_df['timestamp'].max() >= last_processed_time) or \
       (not weather_df.empty and weather_df['timestamp'].max() >= last_processed_time):
        update_last_processed_time(end_time)
    
    logger.info("Processing complete")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")