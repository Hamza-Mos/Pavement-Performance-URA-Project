"""
Weather Station Data Processor
-----------------------------
This script processes weather station data from CSV files,
applies similar aggregation logic as the temperature data,
and saves results in the same format for later correlation with temperature data.

The script:
1. Reads weather station CSV files directly
2. Processes all quantitative weather metrics (temperature, humidity, etc.)
3. Aggregates data to 15-minute, hourly, daily and weekly intervals
4. Saves the aggregated data in the same format as temperature data
5. Updates the shared status file to track processing

Usage:
    python weather_station_processor.py [--input-file INPUT_FILE] [--output-dir OUTPUT_DIR]

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
import re
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/weather_station.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
# Use a separate status file for weather data
STATUS_FILE = "../processed_weather/weather_aggregation_status.json"

# Define quantitative weather metrics that we want to aggregate
# These should match column names from the weather station CSV
WEATHER_METRICS = [
    "temperature", "dew_point", "pressure_station", "pressure_sea", 
    "wind_speed", "wind_gust", "relative_humidity", "windchill", 
    "humidex", "visibility", "solar_radiation"
]

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process weather station data')
    parser.add_argument('--input-file', type=str, required=False, 
                        help='Path to the input weather station CSV file')
    parser.add_argument('--input-dir', type=str, required=False,
                        help='Directory containing weather station CSV files')
    parser.add_argument('--output-dir', type=str, default='../processed_weather',
                        help='Directory for output files')
    return parser.parse_args()

def get_last_processed_time():
    """Read the last processed timestamp from the status file."""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                status = json.load(f)
                return pd.to_datetime(status.get('last_processed_weather_station', '2000-01-01 00:00:00'))
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
    
    # Update the timestamp
    status['last_processed_weather_station'] = timestamp.isoformat()
    
    # Write updated status back to file
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)
    
    logger.info(f"Updated last processed time to {timestamp}")

def read_weather_csv(file_path):
    """
    Read a weather station CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with the weather data
    """
    logger.info(f"Reading weather station data from: {file_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Convert datetime column
        if 'date_time_local' in df.columns:
            # Fix date format - strip quotes and handle timezone
            df['date_time_local'] = df['date_time_local'].str.strip('"')
            # Remove any timezone info from string (EDT, EST, etc.)
            df['date_time_local'] = df['date_time_local'].str.replace(' EDT', '')
            df['date_time_local'] = df['date_time_local'].str.replace(' EST', '')
            
            # Use a more robust approach - split the timestamp and reformat
            def clean_timestamp(ts):
                # Extract just the date and time parts using regex
                match = re.match(r'"?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', ts)
                if match:
                    return match.group(1)
                return ts
            
            df['clean_timestamp'] = df['date_time_local'].apply(clean_timestamp)
            
            # Convert to datetime
            df['timestamp'] = pd.to_datetime(df['clean_timestamp'])
            df = df.drop('clean_timestamp', axis=1)
        elif 'unixtime' in df.columns:
            # Use unix time if available
            df['timestamp'] = pd.to_datetime(df['unixtime'], unit='s')
        else:
            # Try to find any column that might contain datetime information
            datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if datetime_cols:
                df['timestamp'] = pd.to_datetime(df[datetime_cols[0]])
            else:
                logger.error("No timestamp column found in the weather data")
                return pd.DataFrame()
        
        # Drop rows with invalid timestamps
        df = df.dropna(subset=['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Convert all numeric columns to float where possible
        for col in df.columns:
            if col != 'timestamp' and col != 'date_time_local' and col != 'unixtime':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    logger.warning(f"Could not convert column {col} to numeric")
        
        logger.info(f"Loaded {len(df)} rows of weather data")
        logger.info(f"Timespan: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error reading weather CSV file: {e}")
        return pd.DataFrame()

def process_all_weather_files(input_dir):
    """
    Process all weather CSV files in a directory.
    
    Args:
        input_dir: Directory containing weather CSV files
        
    Returns:
        Combined DataFrame with all weather data
    """
    logger.info(f"Processing all weather files in: {input_dir}")
    
    # Find all CSV files
    file_pattern = os.path.join(input_dir, "*.csv")
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        logger.warning(f"No CSV files found in {input_dir}")
        return pd.DataFrame()
    
    # Read and combine all files
    dfs = []
    for file in files:
        df = read_weather_csv(file)
        if not df.empty:
            dfs.append(df)
    
    if not dfs:
        logger.warning("No valid weather data found")
        return pd.DataFrame()
    
    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=['timestamp'])
    
    # Sort by timestamp
    combined_df = combined_df.sort_values('timestamp')
    
    logger.info(f"Combined {len(combined_df)} rows of weather data")
    
    return combined_df

def aggregate_to_15min(df):
    """
    Aggregate weather data to 15-minute intervals.
    
    Args:
        df: DataFrame with weather data
        
    Returns:
        DataFrame with aggregated data for each metric
    """
    logger.info("Aggregating weather data to 15-minute intervals")
    
    if df.empty:
        logger.warning("No data to aggregate")
        return pd.DataFrame()
    
    # Make sure we have a timestamp column
    if 'timestamp' not in df.columns:
        logger.error("No timestamp column found in the data")
        return pd.DataFrame()
    
    # Set timestamp as index
    df_indexed = df.set_index('timestamp')
    
    # Create a list to hold results for each metric
    results = []
    
    # Process each weather metric
    for metric in WEATHER_METRICS:
        if metric not in df.columns:
            logger.warning(f"Metric {metric} not found in the data, skipping")
            continue
        
        logger.info(f"Processing metric: {metric}")
        
        # Extract metric data
        metric_data = df_indexed[[metric]].copy()
        metric_data.columns = ['value']
        
        # Convert to numeric, coercing errors to NaN
        metric_data['value'] = pd.to_numeric(metric_data['value'], errors='coerce')
        
        # Remove NaN values
        metric_data = metric_data.dropna()
        
        if metric_data.empty:
            logger.warning(f"No valid numeric data for metric {metric}")
            continue
        
        # Resample to 15-minute intervals and calculate statistics
        resampled = metric_data.resample('15Min').agg({
            'value': ['mean', 'min', 'max', 'std', 'count']
        })
        
        # Flatten MultiIndex columns
        resampled.columns = ['mean', 'min', 'max', 'std', 'count']
        
        # Reset index to get timestamp as a column
        resampled = resampled.reset_index()
        
        # Filter out rows with zero counts or missing essential values
        resampled = resampled.dropna(subset=['mean', 'min', 'max'])
        resampled = resampled[resampled['count'] > 0]
        
        # Set std to 0.0 when count is 1 (can't calculate std dev for a single value)
        resampled.loc[resampled['count'] == 1, 'std'] = 0.0
        
        if resampled.empty:
            logger.warning(f"No valid aggregated data for metric {metric}")
            continue
        
        # Add metric name instead of sensor_id
        resampled['metric_name'] = metric
        
        # Add to results
        results.append(resampled)
    
    # Combine all results
    if not results:
        logger.warning("No data to aggregate")
        return pd.DataFrame()
    
    aggregated = pd.concat(results, ignore_index=True)
    
    # Sort by timestamp and metric
    aggregated = aggregated.sort_values(['timestamp', 'metric_name'])
    
    logger.info(f"Aggregated data: {len(aggregated)} rows with valid values across {len(set(aggregated['metric_name']))} metrics")
    
    return aggregated

def save_aggregated_data(df, output_dir, time_interval):
    """
    Save aggregated data to a CSV file.
    
    Args:
        df: DataFrame with aggregated data
        output_dir: Directory to save the output file
        time_interval: Time interval of the data ('15min', 'hourly', 'daily', 'weekly')
    """
    if df.empty:
        logger.warning(f"No weather data to save for {time_interval}")
        return
    
    # Create output directory if it doesn't exist
    interval_dir = os.path.join(output_dir, time_interval)
    os.makedirs(interval_dir, exist_ok=True)
    
    # Make a copy of the dataframe to avoid modifying the original
    df_to_save = df.copy()
    
    # Clean up the column names
    if 'sensor_id' in df_to_save.columns and 'metric_name' in df_to_save.columns:
        # If both exist, use metric_name and drop sensor_id
        df_to_save = df_to_save.drop('sensor_id', axis=1)
    elif 'sensor_id' in df_to_save.columns and 'metric_name' not in df_to_save.columns:
        # If only sensor_id exists, rename it to metric_name
        df_to_save = df_to_save.rename(columns={'sensor_id': 'metric_name'})
    
    # Fill empty or NaN std values with 0.0
    if 'std' in df_to_save.columns:
        # Convert std column to numeric, forcing NaN for non-numeric values
        df_to_save['std'] = pd.to_numeric(df_to_save['std'], errors='coerce')
        # Replace NaN with 0.0
        df_to_save['std'] = df_to_save['std'].fillna(0.0)
        # Also set std to 0.0 where count is 1 (can't calculate std for single value)
        if 'count' in df_to_save.columns:
            df_to_save.loc[df_to_save['count'] == 1, 'std'] = 0.0
    
    # Generate filename with current date
    current_date = datetime.now().strftime('%Y%m%d')
    output_file = os.path.join(interval_dir, f"weather_{time_interval}_{current_date}.csv")
    
    # Check if file exists and read existing data to avoid duplicates
    existing_df = pd.DataFrame()
    if os.path.isfile(output_file):
        try:
            existing_df = pd.read_csv(output_file)
        except Exception as e:
            logger.error(f"Error reading existing file {output_file}: {e}")
    
    # For weekly data, check for duplicates
    if time_interval == 'weekly' and not existing_df.empty:
        # Create a key for identifying duplicates
        if 'week_start' in df_to_save.columns and 'metric_name' in df_to_save.columns:
            # Remove duplicates, keeping only the new data
            if 'week_start' in existing_df.columns and 'metric_name' in existing_df.columns:
                # Convert dates to string for comparison
                existing_df['week_start'] = existing_df['week_start'].astype(str)
                df_to_save['week_start'] = df_to_save['week_start'].astype(str)
                
                # Create a merge key
                existing_df['merge_key'] = existing_df['week_start'] + '_' + existing_df['metric_name'].astype(str)
                df_to_save['merge_key'] = df_to_save['week_start'] + '_' + df_to_save['metric_name'].astype(str)
                
                # Only keep rows that aren't already in the existing file
                mask = ~df_to_save['merge_key'].isin(existing_df['merge_key'])
                df_to_save = df_to_save[mask].drop('merge_key', axis=1)
                
                # If all rows were duplicates, nothing new to save
                if df_to_save.empty:
                    logger.info(f"No new data to save for {time_interval}, all entries already exist in the file")
                    return
    
    # Write to CSV
    mode = 'a' if os.path.isfile(output_file) else 'w'
    header = not os.path.isfile(output_file)
    
    df_to_save.to_csv(output_file, mode=mode, index=False, header=header)
    
    logger.info(f"Successfully saved {len(df_to_save)} rows of {time_interval} weather data to {output_file}")

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
    # Handle missing std values by treating them as 0
    sum_weighted_variance = sum(group['count'] * ((group['std'] or 0)**2) for group in groups)
    
    # Second part: sum of (count * (group_mean - combined_mean)²)
    sum_weighted_mean_diff_squared = sum(group['count'] * (group['mean'] - combined_mean)**2 for group in groups)
    
    # Combined variance
    combined_variance = (sum_weighted_variance + sum_weighted_mean_diff_squared) / total_count
    
    # 3. Take square root to get standard deviation
    return math.sqrt(combined_variance)

def aggregate_to_hourly(df):
    """
    Aggregate 15-minute weather data to hourly intervals.
    
    Args:
        df: DataFrame with 15-minute aggregated data
        
    Returns:
        DataFrame with hourly aggregated data
    """
    logger.info("Aggregating weather data to hourly intervals")
    
    if df.empty:
        logger.warning("No data to aggregate to hourly")
        return pd.DataFrame()
    
    # Handle sensor_id vs metric_name column naming
    if 'sensor_id' in df.columns and 'metric_name' not in df.columns:
        df = df.rename(columns={'sensor_id': 'metric_name'})
    
    # Set timestamp as index
    df = df.set_index('timestamp')
    
    # Create a DataFrame to hold the aggregated results
    results = []
    
    # Process each metric separately
    for metric_name, metric_data in df.groupby('metric_name'):
        logger.info(f"Processing hourly aggregation for metric: {metric_name}")
        
        # Group by hour
        hourly_groups = metric_data.groupby(pd.Grouper(freq='1H'))
        
        hourly_results = []
        for hour, hour_data in hourly_groups:
            if not hour_data.empty:
                # Create groups for std dev calculation
                groups = [
                    {'mean': row['mean'], 'std': row['std'] if not pd.isna(row['std']) else 0.0, 'count': row['count']} 
                    for idx, row in hour_data.iterrows()
                ]
                
                # Calculate aggregated values
                count_sum = hour_data['count'].sum()
                
                hourly_result = {
                    'timestamp': hour,
                    'mean': hour_data['mean'].mean(),
                    'min': hour_data['min'].min(),
                    'max': hour_data['max'].max(),
                    'std': calculate_aggregated_std_dev(groups),
                    'count': count_sum,
                    'metric_name': metric_name
                }
                
                # Set std to 0.0 when count is 1
                if count_sum == 1:
                    hourly_result['std'] = 0.0
                
                hourly_results.append(hourly_result)
        
        if hourly_results:
            results.append(pd.DataFrame(hourly_results))
    
    # Combine all results
    if not results:
        logger.warning("No hourly data to aggregate")
        return pd.DataFrame()
    
    aggregated = pd.concat(results, ignore_index=True)
    
    # Sort by timestamp and metric
    aggregated = aggregated.sort_values(['timestamp', 'metric_name'])
    
    return aggregated

def aggregate_to_daily(df):
    """
    Aggregate hourly weather data to daily intervals.
    
    Args:
        df: DataFrame with hourly aggregated data
        
    Returns:
        DataFrame with daily aggregated data
    """
    logger.info("Aggregating weather data to daily intervals")
    
    if df.empty:
        logger.warning("No data to aggregate to daily")
        return pd.DataFrame()
    
    # Handle sensor_id vs metric_name column naming
    if 'sensor_id' in df.columns and 'metric_name' not in df.columns:
        df = df.rename(columns={'sensor_id': 'metric_name'})
    
    # Extract date from timestamp
    df['date'] = df['timestamp'].dt.date
    df['date'] = pd.to_datetime(df['date'])
    
    # Create a DataFrame to hold the aggregated results
    results = []
    
    # Process each metric
    for metric_name, metric_data in df.groupby('metric_name'):
        logger.info(f"Processing daily aggregation for metric: {metric_name}")
        
        # Group by date
        daily_grouped = metric_data.groupby('date')
        
        daily_results = []
        for day, day_data in daily_grouped:
            if not day_data.empty:
                # Create groups for std dev calculation
                groups = [
                    {'mean': row['mean'], 'std': row['std'] if not pd.isna(row['std']) else 0.0, 'count': row['count']} 
                    for idx, row in day_data.iterrows()
                ]
                
                # Calculate aggregated values
                count_sum = day_data['count'].sum()
                
                daily_stats = {
                    'date': day,
                    'metric_name': metric_name,
                    'mean': day_data['mean'].mean(),
                    'min': day_data['min'].min(),
                    'max': day_data['max'].max(),
                    'std': calculate_aggregated_std_dev(groups),
                    'count': count_sum
                }
                
                # Set std to 0.0 when count is 1
                if count_sum == 1:
                    daily_stats['std'] = 0.0
                
                # Calculate daily range
                daily_stats['temp_range'] = daily_stats['max'] - daily_stats['min']
                daily_results.append(daily_stats)
        
        if daily_results:
            results.append(pd.DataFrame(daily_results))
    
    # Combine all results
    if not results:
        logger.warning("No daily data to aggregate")
        return pd.DataFrame()
    
    aggregated = pd.concat(results, ignore_index=True)
    
    # Sort by date and metric
    aggregated = aggregated.sort_values(['date', 'metric_name'])
    
    return aggregated

def aggregate_to_weekly(df):
    """
    Aggregate daily weather data to weekly intervals.
    
    Args:
        df: DataFrame with daily aggregated data
        
    Returns:
        DataFrame with weekly aggregated data
    """
    logger.info("Aggregating weather data to weekly intervals")
    
    if df.empty:
        logger.warning("No data to aggregate to weekly")
        return pd.DataFrame()
    
    # Handle sensor_id vs metric_name column naming
    if 'sensor_id' in df.columns and 'metric_name' not in df.columns:
        df = df.rename(columns={'sensor_id': 'metric_name'})
    
    # Add week column (start of the week, Monday-based)
    df['week'] = df['date'].apply(lambda x: x - timedelta(days=x.weekday()))
    
    # Create a DataFrame to hold the aggregated results
    results = []
    
    # Process each metric by week
    for (week_start, metric_name), week_data in df.groupby(['week', 'metric_name']):
        logger.info(f"Processing weekly aggregation for metric: {metric_name}, week: {week_start}")
        
        if week_data.empty:
            continue
        
        # Calculate week end date
        week_end = week_start + timedelta(days=6)
        
        # Create groups for std dev calculation
        groups = [
            {'mean': row['mean'], 'std': row['std'] if not pd.isna(row['std']) else 0.0, 'count': row['count']} 
            for idx, row in week_data.iterrows()
        ]
        
        # Calculate aggregated values
        count_sum = week_data['count'].sum()
        
        weekly_stats = {
            'week_start': week_start,
            'week_end': week_end,
            'metric_name': metric_name,
            'mean': week_data['mean'].mean(),
            'min': week_data['min'].min(),
            'max': week_data['max'].max(),
            'std': calculate_aggregated_std_dev(groups),
            'count': count_sum
        }
        
        # Set std to 0.0 when count is 1
        if count_sum == 1:
            weekly_stats['std'] = 0.0
        
        # Calculate range for this metric
        weekly_stats['temp_range'] = weekly_stats['max'] - weekly_stats['min']
        
        results.append(weekly_stats)
    
    # Combine all results
    if not results:
        logger.warning("No weekly data to aggregate")
        return pd.DataFrame()
    
    aggregated = pd.DataFrame(results)
    
    # Sort by week start and metric
    aggregated = aggregated.sort_values(['week_start', 'metric_name'])
    
    return aggregated

def main():
    """Main processing function."""
    args = parse_arguments()
    
    # Get output directory
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, '15min'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'hourly'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'daily'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'weekly'), exist_ok=True)
    
    # Get last processed time (for logging)
    last_processed_time = get_last_processed_time()
    logger.info(f"Last processed time: {last_processed_time}")
    
    # Read weather data
    weather_df = pd.DataFrame()
    if args.input_file:
        weather_df = read_weather_csv(args.input_file)
    elif args.input_dir:
        weather_df = process_all_weather_files(args.input_dir)
    else:
        logger.error("No input file or directory specified")
        return
    
    if weather_df.empty:
        logger.error("No weather data to process")
        return
    
    # Process 15-minute aggregates
    weather_15min = aggregate_to_15min(weather_df)
    
    # Rename column to metric_name if needed (do it here once, before all aggregation steps)
    if 'sensor_id' in weather_15min.columns and 'metric_name' not in weather_15min.columns:
        weather_15min = weather_15min.rename(columns={'sensor_id': 'metric_name'})
    
    if not weather_15min.empty:
        save_aggregated_data(weather_15min, output_dir, '15min')
    
    # Process hourly aggregates - make sure we use the renamed DataFrame
    weather_hourly = aggregate_to_hourly(weather_15min)
    if not weather_hourly.empty:
        save_aggregated_data(weather_hourly, output_dir, 'hourly')
    
    # Process daily aggregates
    weather_daily = aggregate_to_daily(weather_hourly)
    if not weather_daily.empty:
        save_aggregated_data(weather_daily, output_dir, 'daily')
    
    # Process weekly aggregates
    weather_weekly = aggregate_to_weekly(weather_daily)
    if not weather_weekly.empty:
        save_aggregated_data(weather_weekly, output_dir, 'weekly')
    
    # Update the last processed time
    if 'timestamp' in weather_df.columns:
        latest_time = weather_df['timestamp'].max()
        update_last_processed_time(latest_time)
        logger.info(f"Updated last processed time to: {latest_time}")
    
    logger.info("Weather data processing complete")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred: {e}") 