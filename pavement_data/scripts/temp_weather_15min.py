"""
Temperature and Weather Data Aggregator - 15-Minute Interval
------------------------------------------------------------
This script processes raw temperature and weather data from TDMS files, 
aggregates it to 15-minute intervals, and saves the results to CSV files.

The script:
1. Reads the raw data from TDMS/Excel files
2. Identifies temperature sensors based on predefined sensor IDs
3. Processes the data in chunks to handle large files efficiently
4. Aggregates data for each sensor in 15-minute intervals
5. Calculates statistics (mean, min, max, std dev, count)
6. Writes the aggregated data to output CSV files
7. Updates a status file to track the last processed timestamp

Usage:
    python temp_weather_15min.py [--input-file INPUT_FILE] [--output-dir OUTPUT_DIR]

Dependencies:
    - pandas
    - numpy
    - nptdms (for reading TDMS files directly)
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import json
import re

import pandas as pd
import numpy as np

try:
    from nptdms import TdmsFile
    TDMS_AVAILABLE = True
except ImportError:
    TDMS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/temp_weather_15min.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
STATUS_FILE = "../processed/aggregation_status.json"
# List of temperature sensor IDs to look for in the data
TEMP_SENSORS = [
    "TEMPS-02-1", "TEMPS-01-1", "TEMPS-02-2", "TEMPS-01-2", 
    "TEMPS-02-3", "TEMPS-01-3", "TM-BA-01", "TM-BA-02", 
    "TM-SB-03", "TM-SB-04", "TM-SB-05", "TM-SG-06", "TM-SG-07"
]
# Add weather sensors if available
WEATHER_SENSORS = []  # Example: ["PRECIP", "HUMIDITY", "WIND"]

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process temperature and weather data to 15-minute intervals')
    parser.add_argument('--input-file', type=str, required=True, help='Path to the input TDMS/Excel file')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory for output files')
    return parser.parse_args()

def get_last_processed_time():
    """Read the last processed timestamp from the status file."""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                status = json.load(f)
                return pd.to_datetime(status.get('last_processed_15min', '2000-01-01 00:00:00'))
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
    
    # Update the 15-minute timestamp
    status['last_processed_15min'] = timestamp.isoformat()
    
    # Write updated status back to file
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)
    
    logger.info(f"Updated last processed time to {timestamp}")

def read_tdms_file(file_path):
    """
    Read a TDMS file and convert it to a pandas DataFrame.
    
    Args:
        file_path: Path to the TDMS file
        
    Returns:
        DataFrame with the TDMS data
    """
    if not TDMS_AVAILABLE:
        logger.error("nptdms package is not installed. Cannot read TDMS files directly.")
        raise ImportError("nptdms package is required to read TDMS files directly.")
    
    logger.info(f"Reading TDMS file: {file_path}")
    try:
        tdms_file = TdmsFile.read(file_path)
        
        # First, find the Time channel with the most data points
        time_channel = None
        max_length = 0
        for group in tdms_file.groups():
            for channel in group.channels():
                if channel.name == 'Time':
                    try:
                        time_data = channel[:]
                        if isinstance(time_data, np.ndarray) and len(time_data) > max_length:
                            time_channel = channel
                            max_length = len(time_data)
                    except Exception as e:
                        logger.warning(f"Could not process Time channel in group {group.name}: {e}")
        
        if time_channel is None:
            raise ValueError("No valid Time channel found in TDMS file")
        
        # Get the time data
        time_data = time_channel[:]
        if not isinstance(time_data, np.ndarray):
            time_data = np.array(time_data)
        time_data = time_data.astype(float)
        
        # Get the time range
        time_start = float(time_data.min())
        time_end = float(time_data.max())
        logger.info(f"Time range: {time_start} to {time_end}")
        
        # Initialize data dictionary with time
        data = {'Time': time_data}
        
        # Process other channels
        for group in tdms_file.groups():
            for channel in group.channels():
                if channel.name != 'Time':
                    try:
                        channel_data = channel[:]
                        if not isinstance(channel_data, np.ndarray):
                            channel_data = np.array(channel_data)
                        
                        # Skip empty channels
                        if len(channel_data) == 0:
                            logger.debug(f"Skipping empty channel {channel.name} in group {group.name}")
                            continue
                        
                        # Try to convert to float if possible
                        try:
                            channel_data = channel_data.astype(float)
                        except:
                            logger.warning(f"Could not convert channel {channel.name} to float, skipping")
                            continue
                        
                        # If lengths don't match, interpolate
                        if len(channel_data) != len(time_data):
                            logger.info(f"Interpolating channel {channel.name} from {len(channel_data)} to {len(time_data)} points")
                            # Create a time array for this channel
                            channel_time = np.linspace(time_start, time_end, len(channel_data))
                            # Interpolate the data to match the time channel
                            from scipy.interpolate import interp1d
                            f = interp1d(channel_time, channel_data, bounds_error=False, fill_value='extrapolate')
                            channel_data = f(time_data)
                        
                        data[channel.name] = channel_data
                            
                    except Exception as e:
                        logger.warning(f"Could not process channel {channel.name} in group {group.name}: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Process timestamp if available
        if 'Time' in df.columns:
            # Get the TestStart_Time if available
            test_start_time = None
            for group in tdms_file.groups():
                if hasattr(group, 'properties') and 'TestStart_Time' in group.properties:
                    test_start_time = pd.to_datetime(group.properties['TestStart_Time'])
                    break
            
            if test_start_time:
                # Add seconds from 'Time' column to the test start time
                df['timestamp'] = test_start_time + pd.to_timedelta(df['Time'], unit='s')
            else:
                # Try to extract date from filename (assuming format like *_MM_DD_YYYY_* or similar)
                filename = os.path.basename(file_path)
                date_match = None
                
                # Extract date pattern from filename
                # Try waterloo format: Waterloo_*_MM_DD_YYYY_HH_MM_SS
                waterloo_pattern = re.search(r'Waterloo_.*?_(\d{2})_(\d{2})_(\d{4})_', filename)
                # Try to match yyyy-mm-dd or yyyy_mm_dd format
                date_pattern_1 = re.search(r'(\d{4})[-_](\d{1,2})[-_](\d{1,2})', filename)
                # Try to match mm_dd_yyyy format
                date_pattern_2 = re.search(r'(\d{1,2})[-_](\d{1,2})[-_](\d{4})', filename)
                
                if waterloo_pattern:
                    # Waterloo format: MM_DD_YYYY
                    month, day, year = waterloo_pattern.groups()
                    date_match = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    logger.info(f"Extracted date {date_match} from Waterloo filename format: {filename}")
                elif date_pattern_1:
                    # Format: yyyy-mm-dd or yyyy_mm_dd
                    year, month, day = date_pattern_1.groups()
                    date_match = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                elif date_pattern_2:
                    # Format: mm_dd_yyyy
                    month, day, year = date_pattern_2.groups()
                    date_match = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                
                if date_match:
                    logger.info(f"Extracted date {date_match} from filename {filename}")
                    # Create a reference date at midnight of the extracted date
                    ref_date = pd.to_datetime(date_match)
                    # Add seconds from 'Time' column to the reference date
                    df['timestamp'] = ref_date + pd.to_timedelta(df['Time'], unit='s')
                else:
                    # As a fallback, use file modification time
                    logger.warning("Could not find TestStart_Time or extract date from filename, using file modification time")
                    file_mtime = pd.to_datetime(datetime.fromtimestamp(os.path.getmtime(file_path)).date())
                    df['timestamp'] = file_mtime + pd.to_timedelta(df['Time'], unit='s')
        else:
            logger.warning("No Time column found in TDMS file")
        
        # Log information about the DataFrame
        logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {', '.join(df.columns)}")
        
        # Log some basic statistics about the data
        if not df.empty:
            logger.info("Data statistics:")
            logger.info(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logger.info(f"Number of unique timestamps: {df['timestamp'].nunique()}")
            for col in df.columns:
                if col not in ['Time', 'timestamp']:
                    logger.info(f"{col}: {df[col].notna().sum()} non-null values")
            
        return df
        
    except Exception as e:
        logger.error(f"Error reading TDMS file: {e}")
        raise

def process_excel_file(file_path):
    """
    Process an Excel file containing TDMS data.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        DataFrame with the data
    """
    logger.info(f"Reading Excel file: {file_path}")
    try:
        df = pd.read_excel(file_path)
        
        # Ensure there's a proper timestamp column
        if 'Timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Timestamp'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'Time' in df.columns and 'Date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        elif 'Time' in df.columns:
            # Try to extract a test start time from the file
            # This might be in the first sheet or metadata
            try:
                metadata = pd.read_excel(file_path, sheet_name="Properties")
                test_start_time = metadata[metadata['Name'] == 'TestStart_Time']['Value'].iloc[0]
                test_start_time = pd.to_datetime(test_start_time)
                df['timestamp'] = test_start_time + pd.to_timedelta(df['Time'], unit='s')
            except:
                # Try to extract date from filename (assuming format like *_MM_DD_YYYY_* or similar)
                filename = os.path.basename(file_path)
                date_match = None
                
                # Extract date pattern from filename
                # Try waterloo format: Waterloo_*_MM_DD_YYYY_HH_MM_SS
                waterloo_pattern = re.search(r'Waterloo_.*?_(\d{2})_(\d{2})_(\d{4})_', filename)
                # Try to match yyyy-mm-dd or yyyy_mm_dd format
                date_pattern_1 = re.search(r'(\d{4})[-_](\d{1,2})[-_](\d{1,2})', filename)
                # Try to match mm_dd_yyyy format
                date_pattern_2 = re.search(r'(\d{1,2})[-_](\d{1,2})[-_](\d{4})', filename)
                
                if waterloo_pattern:
                    # Waterloo format: MM_DD_YYYY
                    month, day, year = waterloo_pattern.groups()
                    date_match = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    logger.info(f"Extracted date {date_match} from Waterloo filename format: {filename}")
                elif date_pattern_1:
                    # Format: yyyy-mm-dd or yyyy_mm_dd
                    year, month, day = date_pattern_1.groups()
                    date_match = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                elif date_pattern_2:
                    # Format: mm_dd_yyyy
                    month, day, year = date_pattern_2.groups()
                    date_match = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                
                if date_match:
                    logger.info(f"Extracted date {date_match} from filename {filename}")
                    # Create a reference date at midnight of the extracted date
                    ref_date = pd.to_datetime(date_match)
                    # Add seconds from 'Time' column to the reference date
                    df['timestamp'] = ref_date + pd.to_timedelta(df['Time'], unit='s')
                else:
                    # As a fallback, use file modification time
                    logger.warning("Could not find TestStart_Time or extract date from filename, using file modification time")
                    file_mtime = pd.to_datetime(datetime.fromtimestamp(os.path.getmtime(file_path)).date())
                    df['timestamp'] = file_mtime + pd.to_timedelta(df['Time'], unit='s')
        
        return df
    
    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")
        raise

def identify_sensors(df):
    """
    Identify temperature and weather sensors in the DataFrame.
    
    Args:
        df: DataFrame with sensor data
        
    Returns:
        Tuple of (temp_sensor_columns, weather_sensor_columns)
    """
    temp_sensor_columns = []
    weather_sensor_columns = []
    
    for column in df.columns:
        # Check if column is a known temperature sensor or contains temperature sensor ID
        if any(sensor in column for sensor in TEMP_SENSORS):
            temp_sensor_columns.append(column)
        
        # Check if column is a known weather sensor or contains weather sensor ID
        if any(sensor in column for sensor in WEATHER_SENSORS):
            weather_sensor_columns.append(column)
    
    logger.info(f"Found {len(temp_sensor_columns)} temperature sensors and "
               f"{len(weather_sensor_columns)} weather sensors")
    
    if not temp_sensor_columns and not weather_sensor_columns:
        logger.warning("No known temperature or weather sensors found in data")
        # Try to identify potential temperature columns by name
        for column in df.columns:
            if any(keyword in column.lower() for keyword in ['temp', 'temperature', 'tm', 'temps']):
                logger.info(f"Adding possible temperature sensor: {column}")
                temp_sensor_columns.append(column)
    
    return temp_sensor_columns, weather_sensor_columns

def aggregate_to_15min(df, sensor_columns):
    """
    Aggregate sensor data to 15-minute intervals.
    
    Args:
        df: DataFrame with sensor data
        sensor_columns: List of columns containing sensor data
        
    Returns:
        DataFrame with aggregated data
    """
    logger.info("Aggregating data to 15-minute intervals")
    
    # Ensure timestamp is set as index
    if df.index.name != 'timestamp':
        df = df.set_index('timestamp')
    
    # Create a DataFrame to hold the aggregated results
    results = []
    
    # Process each sensor
    for sensor in sensor_columns:
        logger.info(f"Processing sensor: {sensor}")
        
        # Extract sensor data
        sensor_data = df[[sensor]].copy()
        sensor_data.columns = ['value']
        
        # Convert to numeric, coercing errors to NaN
        sensor_data['value'] = pd.to_numeric(sensor_data['value'], errors='coerce')
        
        # Filter out values that are exactly zero or very close to zero (sensor not active)
        zero_threshold = 0.001  # Values below this are considered effectively zero
        if not sensor_data.empty:
            non_zero_values = sensor_data[abs(sensor_data['value']) > zero_threshold]
            if len(non_zero_values) > 0:
                # If there are non-zero values, use only those
                logger.info(f"Filtered out {len(sensor_data) - len(non_zero_values)} rows with near-zero values for sensor {sensor}")
                sensor_data = non_zero_values
        
        # Remove any rows with NaN values
        sensor_data = sensor_data.dropna()
        
        if sensor_data.empty:
            logger.warning(f"No valid numeric data for sensor {sensor}")
            continue
        
        # Resample to 15-minute intervals and calculate statistics
        resampled = sensor_data.resample('15Min').agg({
            'value': ['mean', 'min', 'max', 'std', 'count']
        })
        
        # Flatten MultiIndex columns
        resampled.columns = ['mean', 'min', 'max', 'std', 'count']
        
        # Reset index to get timestamp as a column
        resampled = resampled.reset_index()
        
        # Filter out rows with zero counts or missing essential values
        resampled = resampled.dropna(subset=['mean', 'min', 'max'])
        resampled = resampled[resampled['count'] > 0]
        
        # More aggressive filtering: exclude rows where values are very close to zero
        resampled = resampled[
            (abs(resampled['mean']) > zero_threshold) | 
            (abs(resampled['min']) > zero_threshold) | 
            (abs(resampled['max']) > zero_threshold)
        ]
        
        if resampled.empty:
            logger.warning(f"No valid aggregated data for sensor {sensor}")
            continue
        
        # Add sensor name
        resampled['sensor_id'] = sensor
        
        # Add to results
        results.append(resampled)
    
    # Combine all results
    if not results:
        logger.warning("No data to aggregate")
        return pd.DataFrame()
    
    aggregated = pd.concat(results, ignore_index=True)
    
    # Sort by timestamp and sensor
    aggregated = aggregated.sort_values(['timestamp', 'sensor_id'])
    
    logger.info(f"Aggregated data: {len(aggregated)} rows with valid values across {len(set(aggregated['sensor_id']))} sensors")
    
    return aggregated

def save_aggregated_data(df, output_path, data_type):
    """
    Save aggregated data to a CSV file.
    
    Args:
        df: DataFrame with aggregated data
        output_path: Directory to save the output file
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
    
    # Use the output directory as provided (should already include 15min)
    output_dir = output_path
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Using output directory: {output_dir}")
    
    # Generate filename with current date
    current_date = datetime.now().strftime('%Y%m%d')
    output_file = os.path.join(output_dir, f"{data_type}_15min_{current_date}.csv")
    
    # Check if file exists
    file_exists = os.path.isfile(output_file)
    logger.info(f"Output file {'exists' if file_exists else 'does not exist'}: {output_file}")
    
    # Write to CSV
    mode = 'a' if file_exists else 'w'
    header = not file_exists
    
    try:
        valid_data.to_csv(output_file, mode=mode, index=False, header=header)
        logger.info(f"Successfully saved {len(valid_data)} rows of valid {data_type} data to {output_file}")
    except Exception as e:
        logger.error(f"Error saving {data_type} data to {output_file}: {e}")
        raise

def main():
    """Main processing function."""
    args = parse_arguments()
    
    # Determine the input file
    input_file = args.input_file
    
    # Output directory
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)
    logger.info(f"Using output directory: {output_dir}")
    
    # Create base output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, '15min'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'hourly'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'daily'), exist_ok=True)
    
    # Get last processed time (for logging only)
    last_processed_time = get_last_processed_time()
    logger.info(f"Last processed time: {last_processed_time}")
    logger.info("Processing all data regardless of last processed time")
    
    # Read the data
    if input_file.lower().endswith('.tdms'):
        if TDMS_AVAILABLE:
            df = read_tdms_file(input_file)
        else:
            logger.error("Cannot read TDMS file directly. Please install nptdms or convert to Excel first.")
            return
    elif input_file.lower().endswith(('.xlsx', '.xls')):
        df = process_excel_file(input_file)
    else:
        logger.error(f"Unsupported file format: {input_file}")
        return
    
    if df.empty:
        logger.info("No data found in input file")
        return
    
    # Log the full time range of the data
    if 'timestamp' in df.columns:
        logger.info(f"Full data time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Calculate the end time for processing
    # Process up to 15 minutes ago to ensure we have complete intervals
    end_time = datetime.now() - timedelta(minutes=15)
    end_time = end_time.replace(minute=end_time.minute // 15 * 15, second=0, microsecond=0)
    
    logger.info(f"Processing data up to: {end_time}")
    
    # Identify the sensors
    temp_sensors, weather_sensors = identify_sensors(df)
    
    if not temp_sensors and not weather_sensors:
        logger.warning("No temperature or weather sensors found in the data")
        logger.info("Available columns: " + ", ".join(df.columns))
        return
    
    logger.info("Found temperature sensors: " + ", ".join(temp_sensors))
    logger.info("Found weather sensors: " + ", ".join(weather_sensors))
    
    # Process temperature data
    if temp_sensors:
        logger.info("Processing temperature data...")
        temp_data = aggregate_to_15min(df, temp_sensors)
        if not temp_data.empty:
            logger.info(f"Aggregated temperature data: {len(temp_data)} rows")
            # Save to the 15min subdirectory
            save_aggregated_data(temp_data, os.path.join(output_dir, '15min'), 'temperature')
        else:
            logger.warning("No temperature data after aggregation")
    
    # Process weather data
    if weather_sensors:
        logger.info("Processing weather data...")
        weather_data = aggregate_to_15min(df, weather_sensors)
        if not weather_data.empty:
            logger.info(f"Aggregated weather data: {len(weather_data)} rows")
            # Save to the 15min subdirectory
            save_aggregated_data(weather_data, os.path.join(output_dir, '15min'), 'weather')
        else:
            logger.warning("No weather data after aggregation")
    
    # Update the last processed time
    if 'timestamp' in df.columns:
        latest_time = df['timestamp'].max()
        update_last_processed_time(latest_time)
        logger.info(f"Updated last processed time to: {latest_time}")
    
    logger.info("Processing complete")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")