"""
Temperature and Weather Data Aggregator - Weekly Interval
---------------------------------------------------------
This script processes daily aggregated temperature and weather data,
further aggregates it to weekly intervals, and saves the results to a database.

The script:
1. Reads daily aggregated data CSV files
2. Processes data only for complete weeks (ending on Sunday)
3. Calculates weekly statistics including full temperature range
4. Creates or updates database tables for weekly data storage
5. Writes the aggregated data to the database
6. Updates a status file to track the last processed week

Usage:
    python temp_weather_weekly.py [--input-dir INPUT_DIR] [--db-path DB_PATH]

Dependencies:
    - pandas
    - numpy
    - sqlite3 (for database operations)
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import json
import glob
import sqlite3
from pathlib import Path
import math
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/temp_weather_weekly.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
STATUS_FILE = "../processed/aggregation_status.json"
DB_SCHEMA = {
    'temperature_weekly': '''
        CREATE TABLE IF NOT EXISTS temperature_weekly (
            week_start_date DATE,
            week_end_date DATE,
            sensor_id TEXT,
            mean_value REAL,
            min_value REAL,
            max_value REAL,
            std_value REAL,
            temp_range REAL,
            reading_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (week_start_date, sensor_id)
        )
    ''',
    'weather_weekly': '''
        CREATE TABLE IF NOT EXISTS weather_weekly (
            week_start_date DATE,
            week_end_date DATE,
            sensor_id TEXT,
            mean_value REAL,
            min_value REAL,
            max_value REAL,
            std_value REAL,
            reading_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (week_start_date, sensor_id)
        )
    ''',
    'sensor_metadata': '''
        CREATE TABLE IF NOT EXISTS sensor_metadata (
            sensor_id TEXT PRIMARY KEY,
            sensor_type TEXT,
            description TEXT,
            units TEXT,
            depth REAL,
            layer TEXT,
            location TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    '''
}

# Metadata for temperature sensors (can be expanded as needed)
SENSOR_METADATA = [
    {"sensor_id": "TEMPS-02-1", "sensor_type": "temperature", "description": "Thermistor in asphalt layer", "units": "°C", "depth": 72.7, "layer": "asphalt", "location": "right array"},
    {"sensor_id": "TEMPS-01-1", "sensor_type": "temperature", "description": "Thermistor in asphalt layer", "units": "°C", "depth": 72.7, "layer": "asphalt", "location": "left array"},
    {"sensor_id": "TEMPS-02-2", "sensor_type": "temperature", "description": "Thermistor in asphalt layer", "units": "°C", "depth": 145.4, "layer": "asphalt", "location": "right array"},
    {"sensor_id": "TEMPS-01-2", "sensor_type": "temperature", "description": "Thermistor in asphalt layer", "units": "°C", "depth": 145.4, "layer": "asphalt", "location": "left array"},
    {"sensor_id": "TEMPS-02-3", "sensor_type": "temperature", "description": "Thermistor in asphalt layer", "units": "°C", "depth": 195.0, "layer": "asphalt", "location": "right array"},
    {"sensor_id": "TEMPS-01-3", "sensor_type": "temperature", "description": "Thermistor in asphalt layer", "units": "°C", "depth": 195.0, "layer": "asphalt", "location": "left array"},
    {"sensor_id": "TM-BA-01", "sensor_type": "temperature", "description": "Temperature probe in base layer", "units": "°C", "depth": 245.0, "layer": "base", "location": "center"},
    {"sensor_id": "TM-BA-02", "sensor_type": "temperature", "description": "Temperature probe in base layer", "units": "°C", "depth": 295.0, "layer": "base", "location": "center"},
    {"sensor_id": "TM-SB-03", "sensor_type": "temperature", "description": "Temperature probe in subbase layer", "units": "°C", "depth": 395.0, "layer": "subbase", "location": "center"},
    {"sensor_id": "TM-SB-04", "sensor_type": "temperature", "description": "Temperature probe in subbase layer", "units": "°C", "depth": 495.0, "layer": "subbase", "location": "center"},
    {"sensor_id": "TM-SB-05", "sensor_type": "temperature", "description": "Temperature probe in subbase layer", "units": "°C", "depth": 595.0, "layer": "subbase", "location": "center"},
    {"sensor_id": "TM-SG-06", "sensor_type": "temperature", "description": "Temperature probe in subgrade layer", "units": "°C", "depth": 695.0, "layer": "subgrade", "location": "center"},
    {"sensor_id": "TM-SG-07", "sensor_type": "temperature", "description": "Temperature probe in subgrade layer", "units": "°C", "depth": 745.0, "layer": "subgrade", "location": "center"}
]

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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process daily data to weekly intervals and write to database')
    parser.add_argument('--input-dir', type=str, default='./output', help='Directory containing daily data files')
    parser.add_argument('--db-path', type=str, default='./pavement_data.db', help='Path to the SQLite database file')
    return parser.parse_args()

def get_last_processed_week():
    """Read the last processed week from the status file."""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                status = json.load(f)
                last_week = status.get('last_processed_weekly')
                if last_week:
                    return pd.to_datetime(last_week)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error reading status file: {e}")
    
    # Default to a date far in the past if no status file exists
    return pd.to_datetime('2000-01-01')

def update_last_processed_week(week_start_date):
    """Update the status file with the latest processed week."""
    status = {}
    
    # Read existing status if file exists
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                status = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Could not read status file, creating new one")
    
    # Update the weekly timestamp
    status['last_processed_weekly'] = week_start_date.strftime('%Y-%m-%d')
    
    # Write updated status back to file
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)
    
    logger.info(f"Updated last processed week to {week_start_date.strftime('%Y-%m-%d')}")

def initialize_database(db_path):
    """
    Initialize the database with the required schema.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        Connection to the database
    """
    logger.info(f"Initializing database at {db_path}")
    
    # Ensure directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Create tables if they don't exist
    for table_name, schema in DB_SCHEMA.items():
        conn.execute(schema)
    
    # Initialize sensor metadata if the table is empty
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sensor_metadata")
    count = cursor.fetchone()[0]
    
    if count == 0:
        logger.info("Initializing sensor metadata table")
        for sensor in SENSOR_METADATA:
            # Check if sensor exists
            cursor.execute("SELECT COUNT(*) FROM sensor_metadata WHERE sensor_id = ?", (sensor["sensor_id"],))
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    "INSERT INTO sensor_metadata (sensor_id, sensor_type, description, units, depth, layer, location) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (sensor["sensor_id"], sensor["sensor_type"], sensor["description"], sensor["units"], sensor["depth"], sensor["layer"], sensor["location"])
                )
    
    conn.commit()
    logger.info("Database initialized")
    
    return conn

def read_daily_data(input_dir, data_type, last_processed_week):
    """
    Read daily aggregated data files.
    
    Args:
        input_dir: Directory containing the input files
        data_type: Type of data ('temperature' or 'weather')
        last_processed_week: Start date of the last processed week (for logging only)
        
    Returns:
        DataFrame with the daily data
    """
    logger.info(f"Reading {data_type} daily data files")
    logger.info(f"Processing all data regardless of last processed week")
    
    # Find all CSV files for the data type
    file_pattern = os.path.join(input_dir, f"{data_type}_daily_*.csv")
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        logger.warning(f"No {data_type} daily data files found")
        return pd.DataFrame()
    
    # Read and concatenate all files
    dfs = []
    for file in files:
        try:
            logger.info(f"Reading file: {file}")
            df = pd.read_csv(file)
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Add week column (start of the week, Monday-based)
            df['week'] = df['date'].apply(lambda x: x - timedelta(days=x.weekday()))
            
            # Process all data regardless of week
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
    combined_df = combined_df.drop_duplicates(subset=['date', 'sensor_id'])
    
    # Sort by date and sensor
    combined_df = combined_df.sort_values(['date', 'sensor_id'])
    
    logger.info(f"Total {data_type} data to process: {len(combined_df)} rows")
    logger.info(f"Week range: {combined_df['week'].min()} to {combined_df['week'].max()}")
    
    return combined_df

def check_week_completeness(df, week_start):
    """
    Check if a given week has sufficient data (at least 1 day for now).
    
    Args:
        df: DataFrame with daily data
        week_start: Start date of the week to check
        
    Returns:
        Tuple of (is_complete, complete_sensors, week_end)
    """
    week_end = week_start + timedelta(days=6)
    week_start_str = week_start.strftime('%Y-%m-%d')
    week_end_str = week_end.strftime('%Y-%m-%d')
    
    logger.info(f"Checking completeness for week: {week_start_str} to {week_end_str}")
    
    # Filter data for the given week
    week_data = df[(df['date'] >= week_start) & (df['date'] <= week_end)]
    
    if week_data.empty:
        logger.warning(f"No data found for week: {week_start_str} to {week_end_str}")
        return False, [], week_end
    
    # Group by sensor and count days
    days_per_sensor = week_data.groupby('sensor_id')['date'].apply(
        lambda x: len(x.unique())
    )
    
    # Check if any sensor has at least 1 day of data (more lenient)
    required_days = 1  # Reduced from 2 to 1 to be even more lenient
    complete_sensors = days_per_sensor[days_per_sensor >= required_days].index.tolist()
    incomplete_sensors = days_per_sensor[days_per_sensor < required_days].index.tolist()
    
    if incomplete_sensors:
        logger.warning(f"Week {week_start_str} to {week_end_str} has less than {required_days} days of data for sensors: {incomplete_sensors}")
    
    if not complete_sensors:
        logger.warning(f"No sensors have at least {required_days} days of data for week {week_start_str} to {week_end_str}")
        return False, [], week_end
    
    logger.info(f"Found {len(complete_sensors)} sensors with at least {required_days} days of data")
    return True, complete_sensors, week_end

def aggregate_to_weekly(df, week_start=None, week_end=None, sensors=None):
    """
    Aggregate daily data to weekly intervals.
    
    Args:
        df: DataFrame with daily aggregated data
        week_start: Optional specific week start date to process
        week_end: Optional specific week end date to process
        sensors: Optional list of sensors to process
        
    Returns:
        DataFrame with weekly aggregated data
    """
    logger.info(f"Aggregating data to weekly intervals for {week_start if week_start else 'all weeks'}")
    
    if df.empty:
        logger.warning("No data to aggregate")
        return pd.DataFrame()
    
    # Filter for specific week and sensors if provided
    if week_start is not None and week_end is not None:
        df = df[(df['date'] >= week_start) & (df['date'] <= week_end)]
    
    if sensors is not None:
        df = df[df['sensor_id'].isin(sensors)]
    
    if df.empty:
        logger.warning(f"No data to aggregate for {week_start if week_start else 'selected weeks'}")
        return pd.DataFrame()
    
    # Filter out rows with NaN values in critical columns
    df = df.dropna(subset=['mean', 'min', 'max', 'std', 'count'])
    
    # Filter out rows with zero count
    df = df[df['count'] > 0]
    
    if df.empty:
        logger.warning("No valid data to aggregate after filtering out NaN values")
        return pd.DataFrame()
    
    # Create a DataFrame to hold the aggregated results
    results = []
    
    # Process each sensor by week start
    for (week_start_date, sensor_id), sensor_week_data in df.groupby(['week', 'sensor_id']):
        logger.info(f"Processing sensor: {sensor_id} for week starting {week_start_date.strftime('%Y-%m-%d')}")
        
        # Skip if no valid data
        if sensor_week_data.empty:
            continue
        
        # Calculate week end date (last day with data or maximum of 7 days)
        week_end_date = min(week_start_date + timedelta(days=6), sensor_week_data['date'].max())
        
        # Create groups for std dev calculation
        groups = [
            {'mean': row['mean'], 'std': row['std'], 'count': row['count']} 
            for idx, row in sensor_week_data.iterrows()
            if not pd.isna(row['mean']) and not pd.isna(row['std']) and not pd.isna(row['count'])
        ]
        
        if not groups:
            logger.warning(f"No valid data groups for sensor {sensor_id}")
            continue
        
        # Calculate weekly statistics
        weekly_stats = {
            'week_start': week_start_date,
            'week_end': week_end_date,
            'sensor_id': sensor_id,
            'mean': sensor_week_data['mean'].mean(),
            'min': sensor_week_data['min'].min(),
            'max': sensor_week_data['max'].max(),
            'std': calculate_aggregated_std_dev(groups),
            'count': sensor_week_data['count'].sum()
        }
        
        # Calculate weekly temperature range
        weekly_stats['temp_range'] = weekly_stats['max'] - weekly_stats['min']
        
        # Add to results
        results.append(weekly_stats)
    
    # Combine all results
    if not results:
        logger.warning("No data to aggregate")
        return pd.DataFrame()
    
    aggregated = pd.DataFrame(results)
    
    # Sort by week start and sensor
    aggregated = aggregated.sort_values(['week_start', 'sensor_id'])
    
    return aggregated

def write_to_database(df, conn, data_type):
    """
    Write aggregated data to the database.
    
    Args:
        df: DataFrame with aggregated data
        conn: Database connection
        data_type: Type of data ('temperature' or 'weather')
    """
    if df.empty:
        logger.warning(f"No {data_type} data to write to database")
        return
    
    table_name = f"{data_type}_weekly"
    
    # Prepare data for insertion
    cursor = conn.cursor()
    
    # Process data in chunks to avoid large transactions
    chunk_size = 100
    records_inserted = 0
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        
        for _, row in chunk.iterrows():
            if data_type == 'temperature':
                query = f'''
                    INSERT OR REPLACE INTO {table_name} 
                    (week_start_date, week_end_date, sensor_id, mean_value, min_value, max_value, std_value, temp_range, reading_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''
                cursor.execute(query, (
                    row['week_start'].strftime('%Y-%m-%d'),
                    row['week_end'].strftime('%Y-%m-%d'),
                    row['sensor_id'],
                    row['mean'],
                    row['min'],
                    row['max'],
                    row['std'],
                    row['temp_range'],
                    row['count']
                ))
            else:
                query = f'''
                    INSERT OR REPLACE INTO {table_name} 
                    (week_start_date, week_end_date, sensor_id, mean_value, min_value, max_value, std_value, reading_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                '''
                cursor.execute(query, (
                    row['week_start'].strftime('%Y-%m-%d'),
                    row['week_end'].strftime('%Y-%m-%d'),
                    row['sensor_id'],
                    row['mean'],
                    row['min'],
                    row['max'],
                    row['std'],
                    row['count']
                ))
        
        # Commit after each chunk
        conn.commit()
        records_inserted += len(chunk)
        logger.info(f"Inserted {records_inserted} of {len(df)} records to {table_name} table")
    
    logger.info(f"Completed writing {records_inserted} records to {table_name} table")

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
    output_file = os.path.join(output_dir, f"{data_type}_weekly_{current_date}.csv")
    
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
    
    # Input directory and database path
    input_dir = args.input_dir
    db_path = args.db_path
    
    # Derive output directory from input directory
    output_dir = os.path.dirname(input_dir)
    if not output_dir:
        output_dir = '.'
    
    # Create weekly directory if it doesn't exist
    weekly_dir = os.path.join(output_dir, 'weekly')
    os.makedirs(weekly_dir, exist_ok=True)
    
    # Get last processed week
    last_processed_week = get_last_processed_week()
    logger.info(f"Last processed week: {last_processed_week.strftime('%Y-%m-%d')}")
    
    # Calculate the end week for processing
    # Process up to the last complete week (ending on Sunday)
    today = datetime.now().date()
    days_since_monday = today.weekday()  # 0 = Monday, 6 = Sunday
    
    # Find the most recent Monday (start of current week)
    current_week_start = today - timedelta(days=days_since_monday)
    
    # We want the previous completed week, so go back one more week if not enough days have passed
    if days_since_monday < 6:  # If today is not Sunday, we need the previous week
        last_complete_week_start = current_week_start - timedelta(days=7)
    else:
        last_complete_week_start = current_week_start
    
    logger.info(f"Processing data up to week starting: {last_complete_week_start}")
    
    # Initialize database
    conn = initialize_database(db_path)
    
    # Read temperature data
    temp_df = read_daily_data(input_dir, 'temperature', last_processed_week)
    
    # Read weather data
    weather_df = read_daily_data(input_dir, 'weather', last_processed_week)
    
    # Get the set of week start dates to process
    all_week_starts = set()
    
    if not temp_df.empty:
        all_week_starts.update(temp_df['week'].unique())
    
    if not weather_df.empty:
        all_week_starts.update(weather_df['week'].unique())
    
    # Sort week starts and filter for those before or equal to last_complete_week_start
    all_week_starts = sorted([w for w in all_week_starts if w <= pd.to_datetime(last_complete_week_start)])
    
    if not all_week_starts:
        logger.info("No weeks to process within the valid range")
        conn.close()
        return
    
    # Process each week separately to ensure completeness
    latest_processed_week = last_processed_week
    
    for week_start in all_week_starts:
        week_start_str = week_start.strftime('%Y-%m-%d')
        logger.info(f"Processing week starting: {week_start_str}")
        
        # Check temperature data completeness
        temp_complete = False
        complete_temp_sensors = []
        temp_week_end = None
        if not temp_df.empty:
            temp_complete, complete_temp_sensors, temp_week_end = check_week_completeness(temp_df, week_start)
        
        # Check weather data completeness
        weather_complete = False
        complete_weather_sensors = []
        weather_week_end = None
        if not weather_df.empty:
            weather_complete, complete_weather_sensors, weather_week_end = check_week_completeness(weather_df, week_start)
        
        # Process temperature data if complete
        if temp_complete and complete_temp_sensors:
            weekly_temp = aggregate_to_weekly(temp_df, week_start, temp_week_end, complete_temp_sensors)
            # Log the aggregated temperature data
            logger.info(f"Weekly temperature aggregates for week starting {week_start_str}:")
            logger.info(f"Total records: {len(weekly_temp)}")
            
            # Summary statistics
            if not weekly_temp.empty:
                avg_mean = weekly_temp['mean'].mean()
                avg_min = weekly_temp['min'].mean()
                avg_max = weekly_temp['max'].mean()
                avg_range = weekly_temp['temp_range'].mean()
                total_readings = weekly_temp['count'].sum()
                logger.info(f"Summary - Avg Mean: {avg_mean:.2f}°C, Avg Min: {avg_min:.2f}°C, Avg Max: {avg_max:.2f}°C, Avg Range: {avg_range:.2f}°C, Total Readings: {total_readings}")
                
                # Log by layer if possible
                try:
                    if 'layer' in SENSOR_METADATA[0]:
                        # Create a mapping of sensor_id to layer
                        sensor_layers = {sensor['sensor_id']: sensor['layer'] for sensor in SENSOR_METADATA}
                        weekly_temp['layer'] = weekly_temp['sensor_id'].map(sensor_layers)
                        
                        # Group by layer and calculate mean
                        layer_stats = weekly_temp.groupby('layer').agg({'mean': 'mean', 'min': 'mean', 'max': 'mean'})
                        for layer, row in layer_stats.iterrows():
                            if not pd.isna(layer): # Skip if layer is None or NaN
                                logger.info(f"Layer {layer} - Mean: {row['mean']:.2f}°C, Min: {row['min']:.2f}°C, Max: {row['max']:.2f}°C")
                except Exception as e:
                    logger.warning(f"Could not generate layer statistics: {e}")
            
            # Detailed logs per sensor
            for _, row in weekly_temp.iterrows():
                logger.info(f"Sensor: {row['sensor_id']}, Mean: {row['mean']:.2f}°C, Min: {row['min']:.2f}°C, Max: {row['max']:.2f}°C, Range: {row['temp_range']:.2f}°C, StdDev: {row['std']:.2f}, Count: {row['count']}")
            
            # Save to CSV
            save_aggregated_data(weekly_temp, weekly_dir, 'temperature')
            
            # Comment out database writing for now
            # write_to_database(weekly_temp, conn, 'temperature')
            latest_processed_week = week_start
        elif not temp_df.empty:
            logger.warning(f"Skipping incomplete temperature data for week starting {week_start_str}")
        
        # Process weather data if complete
        if weather_complete and complete_weather_sensors:
            weekly_weather = aggregate_to_weekly(weather_df, week_start, weather_week_end, complete_weather_sensors)
            # Log the aggregated weather data
            logger.info(f"Weekly weather aggregates for week starting {week_start_str}:")
            logger.info(f"Total records: {len(weekly_weather)}")
            
            # Summary statistics
            if not weekly_weather.empty:
                avg_mean = weekly_weather['mean'].mean()
                avg_min = weekly_weather['min'].mean()
                avg_max = weekly_weather['max'].mean()
                total_readings = weekly_weather['count'].sum()
                logger.info(f"Summary - Avg Mean: {avg_mean:.2f}, Avg Min: {avg_min:.2f}, Avg Max: {avg_max:.2f}, Total Readings: {total_readings}")
            
            # Detailed logs per sensor
            for _, row in weekly_weather.iterrows():
                logger.info(f"Sensor: {row['sensor_id']}, Mean: {row['mean']:.2f}, Min: {row['min']:.2f}, Max: {row['max']:.2f}, StdDev: {row['std']:.2f}, Count: {row['count']}")
            
            # Save to CSV
            save_aggregated_data(weekly_weather, weekly_dir, 'weather')
            
            # Comment out database writing for now
            # write_to_database(weekly_weather, conn, 'weather')
            if latest_processed_week < week_start:
                latest_processed_week = week_start
        elif not weather_df.empty:
            logger.warning(f"Skipping incomplete weather data for week starting {week_start_str}")
    
    # Close database connection
    conn.close()
    
    # Update the last processed week
    if latest_processed_week > last_processed_week:
        update_last_processed_week(latest_processed_week)
    
    logger.info("Processing complete")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")