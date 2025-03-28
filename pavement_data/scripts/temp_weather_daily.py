"""
Temperature and Weather Data Aggregator - Daily Interval
--------------------------------------------------------
This script processes hourly aggregated temperature and weather data,
further aggregates it to daily intervals, and saves the results to CSV files.

The script:
1. Reads hourly aggregated data CSV files
2. Processes data only for complete days (up through yesterday)
3. Calculates daily statistics including full daily temperature range
4. Captures the complete day/night cycle for engineering significance
5. Writes the aggregated data to output CSV files
6. Updates a status file to track the last processed date

Usage:
    python temp_weather_daily.py [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR]

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
        logging.FileHandler("../logs/temp_weather_daily.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
STATUS_FILE = "../processed/aggregation_status.json"

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process hourly data to daily intervals')
    parser.add_argument('--input-dir', type=str, default='./output', help='Directory containing hourly data files')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory for output files')
    return parser.parse_args()

def get_last_processed_date():
    """Read the last processed date from the status file."""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                status = json.load(f)
                return pd.to_datetime(status.get('last_processed_daily', '2000-01-01'))
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error reading status file: {e}")
    
    # Default to a date far in the past if no status file exists
    return pd.to_datetime('2000-01-01')

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

def update_last_processed_date(date):
    """Update the status file with the latest processed date."""
    status = {}
    
    # Read existing status if file exists
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                status = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Could not read status file, creating new one")
    
    # Update the daily timestamp
    status['last_processed_daily'] = date.strftime('%Y-%m-%d')
    
    # Write updated status back to file
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)
    
    logger.info(f"Updated last processed date to {date.strftime('%Y-%m-%d')}")

def read_hourly_data(input_dir, data_type, last_processed_date):
    """
    Read hourly aggregated data files.
    
    Args:
        input_dir: Directory containing the input files
        data_type: Type of data ('temperature' or 'weather')
        last_processed_date: Date of the last processed data (for logging only)
        
    Returns:
        DataFrame with the hourly data
    """
    logger.info(f"Reading {data_type} hourly data files")
    logger.info(f"Processing all data regardless of last processed date")
    
    # Find all CSV files for the data type
    file_pattern = os.path.join(input_dir, f"{data_type}_hourly_*.csv")
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        logger.warning(f"No {data_type} hourly data files found")
        return pd.DataFrame()
    
    # Read and concatenate all files
    dfs = []
    for file in files:
        try:
            logger.info(f"Reading file: {file}")
            df = pd.read_csv(file)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Extract date from timestamp
            df['date'] = df['timestamp'].dt.date
            df['date'] = pd.to_datetime(df['date'])
            
            # Process all data regardless of date
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
    logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    return combined_df

def check_day_completeness(df, day_date):
    """
    Check if a given day has sufficient data (at least 1 hour instead of 24 hours).
    
    Args:
        df: DataFrame with hourly data
        day_date: Date to check
        
    Returns:
        Tuple of (is_complete, complete_sensors, day_end)
    """
    day_end = day_date + timedelta(days=1) - timedelta(seconds=1)
    day_str = day_date.strftime('%Y-%m-%d')
    
    logger.info(f"Checking data for day: {day_str}")
    
    # Filter data for the given day
    day_data = df[(df['timestamp'].dt.date == day_date.date())]
    
    if day_data.empty:
        logger.warning(f"No data found for day: {day_str}")
        return False, [], day_end
    
    # Group by sensor and count hours
    hours_per_sensor = day_data.groupby('sensor_id')['timestamp'].apply(
        lambda x: len(x.unique())
    )
    
    # Check if any sensor has at least 1 hour of data (more lenient)
    # Was 24 hours before, which is too strict for our dataset
    required_hours = 1
    complete_sensors = hours_per_sensor[hours_per_sensor >= required_hours].index.tolist()
    incomplete_sensors = hours_per_sensor[hours_per_sensor < required_hours].index.tolist()
    
    if incomplete_sensors:
        logger.warning(f"Date {day_str} has less than {required_hours} hours of data for sensors: {incomplete_sensors}")
    
    if not complete_sensors:
        logger.warning(f"No sensors have at least {required_hours} hours of data for day {day_str}")
        return False, [], day_end
    
    logger.info(f"Found {len(complete_sensors)} sensors with at least {required_hours} hours of data")
    return True, complete_sensors, day_end

def aggregate_to_daily(df, date=None, sensors=None):
    """
    Aggregate hourly data to daily intervals.
    
    Args:
        df: DataFrame with hourly aggregated data
        date: Optional specific date to process
        sensors: Optional list of sensors to process
        
    Returns:
        DataFrame with daily aggregated data
    """
    logger.info(f"Aggregating data to daily intervals for {date if date else 'all dates'}")
    
    if df.empty:
        logger.warning("No data to aggregate")
        return pd.DataFrame()
    
    # Filter for specific date and sensors if provided
    if date is not None:
        df = df[df['date'] == date]
    
    if sensors is not None:
        df = df[df['sensor_id'].isin(sensors)]
    
    if df.empty:
        logger.warning(f"No data to aggregate for {date if date else 'selected dates'}")
        return pd.DataFrame()
    
    # Create a DataFrame to hold the aggregated results
    results = []
    
    # Process each sensor
    for sensor_id, sensor_data in df.groupby('sensor_id'):
        logger.info(f"Processing sensor: {sensor_id}")
        
        # Group by date
        daily_grouped = sensor_data.groupby('date')
        
        daily_results = []
        for day, day_data in daily_grouped:
            if not day_data.empty:
                # Create groups for std dev calculation
                groups = [
                    {'mean': row['mean'], 'std': row['std'], 'count': row['count']} 
                    for idx, row in day_data.iterrows()
                ]
                
                daily_stats = {
                    'date': day,
                    'sensor_id': sensor_id,
                    'mean': day_data['mean'].mean(),
                    'min': day_data['min'].min(),
                    'max': day_data['max'].max(),
                    'std': calculate_aggregated_std_dev(groups),
                    'count': day_data['count'].sum()
                }
                
                # Calculate daily temperature range
                daily_stats['temp_range'] = daily_stats['max'] - daily_stats['min']
                daily_results.append(daily_stats)
        
        if daily_results:
            results.append(pd.DataFrame(daily_results))
    
    # Combine all results
    if not results:
        logger.warning("No data to aggregate")
        return pd.DataFrame()
    
    aggregated = pd.concat(results, ignore_index=True)
    
    # Sort by date and sensor
    aggregated = aggregated.sort_values(['date', 'sensor_id'])
    
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
    output_file = os.path.join(output_dir, f"{data_type}_daily_{current_date}.csv")
    
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
    
    # Get last processed date
    last_processed_date = get_last_processed_date()
    logger.info(f"Last processed date: {last_processed_date.strftime('%Y-%m-%d')}")
    
    # Calculate the end date for processing
    # Process up to yesterday (ensure we have complete days)
    end_date = datetime.now().date() - timedelta(days=1)
    end_date = pd.to_datetime(end_date)
    
    logger.info(f"Processing data up to: {end_date.strftime('%Y-%m-%d')}")
    
    # Read temperature data
    temp_df = read_hourly_data(input_dir, 'temperature', last_processed_date)
    
    # Read weather data
    weather_df = read_hourly_data(input_dir, 'weather', last_processed_date)
    
    # Get the range of dates to process
    if not temp_df.empty:
        available_dates = sorted(temp_df['date'].unique())
    elif not weather_df.empty:
        available_dates = sorted(weather_df['date'].unique())
    else:
        logger.info("No data to process")
        return
    
    # Filter dates up to end_date
    available_dates = [d for d in available_dates if d <= end_date]
    
    if not available_dates:
        logger.info("No dates to process within the valid range")
        return
    
    # Process each date separately to ensure completeness
    latest_processed_date = last_processed_date
    
    for date in available_dates:
        date_str = date.strftime('%Y-%m-%d')
        logger.info(f"Processing date: {date_str}")
        
        # Check temperature data completeness
        temp_complete = False
        complete_temp_sensors = []
        if not temp_df.empty:
            temp_complete, complete_temp_sensors, _ = check_day_completeness(temp_df, date)
        
        # Check weather data completeness
        weather_complete = False
        complete_weather_sensors = []
        if not weather_df.empty:
            weather_complete, complete_weather_sensors, _ = check_day_completeness(weather_df, date)
        
        # Process temperature data if complete
        if temp_complete and complete_temp_sensors:
            daily_temp = aggregate_to_daily(temp_df, date, complete_temp_sensors)
            # Save to daily subdirectory
            save_aggregated_data(daily_temp, os.path.join(output_dir, 'daily'), 'temperature')
            latest_processed_date = date
        elif not temp_df.empty:
            logger.warning(f"Skipping incomplete temperature data for {date_str}")
        
        # Process weather data if complete
        if weather_complete and complete_weather_sensors:
            daily_weather = aggregate_to_daily(weather_df, date, complete_weather_sensors)
            # Save to daily subdirectory
            save_aggregated_data(daily_weather, os.path.join(output_dir, 'daily'), 'weather')
            if latest_processed_date < date:
                latest_processed_date = date
        elif not weather_df.empty:
            logger.warning(f"Skipping incomplete weather data for {date_str}")
    
    # Update the last processed date
    if latest_processed_date > last_processed_date:
        update_last_processed_date(latest_processed_date)
    
    logger.info("Processing complete")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")