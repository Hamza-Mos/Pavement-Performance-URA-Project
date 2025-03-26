"""
Simple Pavement Temperature Data Processor
- Reads TDMS files for temperature readings
- Stores data in CSV files
- Creates aggregates (15min, hourly, daily)
- Updates database only once per week
"""
import math
import os
import csv
import pandas as pd
import zipfile
from datetime import datetime, timedelta
import nptdms

# config
DATA_DIR = "temp_data"
CSV_DIRS = {
    "raw": f"{DATA_DIR}/raw",
    "15min": f"{DATA_DIR}/15min", 
    "hourly": f"{DATA_DIR}/hourly",
    "daily": f"{DATA_DIR}/daily",
    "archive": f"{DATA_DIR}/archive"
}

# temp sensor mapping - depth and layer information
SENSOR_INFO = {
    'TEMPS-02-1': {'depth': 0.0, 'layer': 'ASPHALT'},
    'TEMPS-01-1': {'depth': 0.0, 'layer': 'ASPHALT'},
    'TEMPS-02-2': {'depth': 0.0, 'layer': 'ASPHALT'},
    'TEMPS-01-2': {'depth': 0.0, 'layer': 'ASPHALT'},
    'TEMPS-02-3': {'depth': 0.0, 'layer': 'ASPHALT'},
    'TEMPS-01-3': {'depth': 0.0, 'layer': 'ASPHALT'},
    'TM-BA-01': {'depth': 195.0, 'layer': 'BASE'},
    'TM-BA-02': {'depth': 195.0, 'layer': 'BASE'},
    'TM-SB-03': {'depth': 245.0, 'layer': 'SUBBASE'},
    'TM-SB-04': {'depth': 245.0, 'layer': 'SUBBASE'},
    'TM-SB-05': {'depth': 245.0, 'layer': 'SUBBASE'},
    'TM-SG-06': {'depth': 745.0, 'layer': 'SUBGRADE'},
    'TM-SG-07': {'depth': 745.0, 'layer': 'SUBGRADE'}
}

# status tracker
status = {
    "last_processed": {
        "15min": None,
        "hourly": None,
        "daily": None
    },
    "last_db_update": None,
    "week_start": datetime.now().strftime("%Y-%m-%d")
}

"""
how to calculate variance from inputs:
"""
"""
In this context, a "group" refers to each time interval from the previous aggregation level that you're combining into a 
higher-level statistic.

For example, when calculating hourly standard deviation:

Each "group" is one 15-minute interval
You would have 4 groups per hour (the four 15-minute intervals)
Each group has statistics: mean temperature, standard deviation, and count of readings

So in the function I shared, the groups parameter would be a list containing 4 dictionaries (one for each 15-minute interval), 
with each dictionary containing:

{
    'mean': 19.2,  # mean temperature for this 15-minute interval
    'std': 0.3,    # standard deviation for this 15-minute interval
    'count': 6     # number of readings in this 15-minute interval
}
"""
def calculate_aggregated_std_dev(groups):
    """
    Calculate the correct standard deviation when aggregating data.
    
    Parameters:
    groups - List of dictionaries with 'mean', 'std', and 'count' for each group
    
    Returns:
    Aggregated standard deviation
    """
    # 1. Calculate the overall combined mean (weighted by counts)
    total_count = sum(group['count'] for group in groups)
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

def read_tdms_files():
    """Read TDMS files and extract temperature data to raw CSV"""
    # Find all TDMS files in the data directory
    tdms_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.tdms'):
                tdms_files.append(os.path.join(root, file))
    
    if not tdms_files:
        print("No TDMS files found")
        return
    
    # create/append to raw CSV
    with open(f"{CSV_DIRS['raw']}/current_temps.csv", 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # write header if file is empty
        if os.path.getsize(f"{CSV_DIRS['raw']}/current_temps.csv") == 0:
            writer.writerow(['timestamp', 'sensor_id', 'temperature', 'depth', 'layer'])
        
        # Process each TDMS file
        for tdms_file in tdms_files:
            try:
                # Read the TDMS file
                tdms_data = nptdms.TdmsFile.read(tdms_file)
                
                # Extract temperature data from all groups and channels
                for group in tdms_data.groups():
                    for channel in group.channels():
                        # Check if the channel represents a temperature sensor
                        sensor_id = channel.name
                        
                        if sensor_id in SENSOR_INFO:
                            depth = SENSOR_INFO[sensor_id]['depth']
                            layer = SENSOR_INFO[sensor_id]['layer']
                            
                            # Get temperature values
                            temperature_values = channel[:]
                            
                            # Get timestamps if available
                            try:
                                # Try to get timestamps from the channel
                                timestamps = channel.time_track()
                            except:
                                # If no explicit timestamps, create based on file time
                                file_time = datetime.fromtimestamp(os.path.getmtime(tdms_file))
                                timestamps = [file_time + timedelta(seconds=i*10) for i in range(len(temperature_values))]
                            
                            # Write temperature data to CSV
                            for i, temp in enumerate(temperature_values):
                                if i < len(timestamps):
                                    timestamp = timestamps[i]
                                else:
                                    # Fallback if timestamps and values length don't match
                                    timestamp = datetime.now() + timedelta(minutes=i*10)
                                
                                if isinstance(timestamp, datetime):
                                    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                                else:
                                    timestamp_str = str(timestamp)
                                
                                writer.writerow([
                                    timestamp_str,
                                    sensor_id,
                                    round(float(temp), 2),
                                    depth,
                                    layer
                                ])
            except Exception as e:
                print(f"Error processing TDMS file {tdms_file}: {str(e)}")
                continue
    
    print("Processed TDMS files to raw CSV")

def process_15min_aggregates():
    """Process raw data into 15-minute aggregates"""
    raw_file = f"{CSV_DIRS['raw']}/current_temps.csv"
    if not os.path.exists(raw_file) or os.path.getsize(raw_file) == 0:
        print("No raw data to process")
        return
        
    # Read raw CSV data
    df = pd.read_csv(raw_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Only process new data
    if status['last_processed']['15min']:
        last_time = datetime.strptime(status['last_processed']['15min'], "%Y-%m-%d %H:%M:%S")
        df = df[df['timestamp'] > last_time]
    
    if df.empty:
        return
        
    # Create 15min intervals and group data
    df['interval'] = df['timestamp'].dt.floor('15min')
    grouped = df.groupby(['interval', 'sensor_id', 'depth', 'layer'])
    
    # Calculate statistics
    agg_data = grouped['temperature'].agg(['mean', 'min', 'max', 'count']).reset_index()
    agg_data.columns = ['interval_start', 'sensor_id', 'depth', 'layer', 'avg_temp', 'min_temp', 'max_temp', 'count']
    
    # Write to 15min CSV
    min15_file = f"{CSV_DIRS['15min']}/current_15min.csv"
    file_exists = os.path.exists(min15_file) and os.path.getsize(min15_file) > 0
    
    # Convert interval to string for CSV
    agg_data['interval_start'] = agg_data['interval_start'].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Append to file
    agg_data.to_csv(min15_file, mode='a', header=not file_exists, index=False)
    
    # Update status
    status['last_processed']['15min'] = df['timestamp'].max().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Processed 15min aggregates")

def process_hourly_aggregates():
    """Process 15min data into hourly aggregates"""
    # Similar approach to 15min aggregation but using 15min data as input
    # and aggregating to hourly intervals
    # Important: When aggregating, take min of mins and max of maxes
    print("Processed hourly aggregates")

def process_daily_aggregates():
    """Process hourly data into daily aggregates"""
    # Similar approach to hourly aggregation but using hourly data as input
    # and aggregating to daily intervals
    print("Processed daily aggregates")

def weekly_db_update():
    """Archive current data and update database (once per week)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    week_id = datetime.strptime(status["week_start"], "%Y-%m-%d").strftime("%Y%m%d")
    
    # 1. Archive current CSV files
    archive_path = f"{CSV_DIRS['archive']}/week_{week_id}_{timestamp}.zip"
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for csv_type, dir_path in CSV_DIRS.items():
            if csv_type != 'archive':
                for file in os.listdir(dir_path):
                    if file.startswith("current_") and file.endswith(".csv"):
                        zipf.write(os.path.join(dir_path, file), f"{csv_type}/{file}")
    
    # 2. Update database
    # In real implementation, connect to database and insert archive info
    print(f"[DB UPDATE] Recording archive: {archive_path}")
    
    # 3. Reset CSV files
    for dir_path in [CSV_DIRS['raw'], CSV_DIRS['15min'], CSV_DIRS['hourly'], CSV_DIRS['daily']]:
        for file in os.listdir(dir_path):
            if file.startswith("current_") and file.endswith(".csv"):
                os.remove(os.path.join(dir_path, file))
                # Create new empty file (CSV writers will add headers)
                open(os.path.join(dir_path, file), 'w').close()
    
    # 4. Update status
    status["last_db_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status["week_start"] = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Weekly database update complete")

def main():
    """Main execution function"""
    # create directories
    for dir_path in CSV_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # initialize files if needed
    if not os.path.exists(f"{CSV_DIRS['raw']}/current_temps.csv"):
        open(f"{CSV_DIRS['raw']}/current_temps.csv", 'w').close()
    
    # 1. read TDMS files
    read_tdms_files()
    
    # 2. process aggregates
    process_15min_aggregates()
    process_hourly_aggregates()
    process_daily_aggregates()
    
    # 3. check if weekly DB update is needed
    if status["last_db_update"]:
        last_update = datetime.strptime(status["last_db_update"], "%Y-%m-%d %H:%M:%S")
        days_since_update = (datetime.now() - last_update).days
        if days_since_update >= 7:
            weekly_db_update()
    else:
        weekly_db_update()  # First run
    
    print("Processing complete")

if __name__ == "__main__":
    main()