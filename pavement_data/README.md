# Pavement Data Processing and Visualization

This folder contains all the scripts and data for processing pavement temperature measurements and weather station data. The system helps you organize raw sensor data into useful time-based summaries.

## Folder Structure

- `raw/`: Where your raw data files go
  - Put TDMS files from temperature sensors here
  - Put CSV files from weather stations here
- `processed/`: Where temperature summaries are saved
  - `15min/`: 15-minute averages
  - `hourly/`: Hourly averages
  - `daily/`: Daily averages
  - `weekly/`: Weekly averages
- `processed_weather/`: Where weather data summaries are saved
  - Has the same structure as `processed/`
- `scripts/`: Contains all the processing scripts
- `plots/`: Where visualizations are saved
- `logs/`: Contains processing logs
- `database/`: For database files (if using database storage)

## How to Process Your Data

### For Temperature Data (TDMS Files)

1. Put your TDMS file in the `raw/` folder
2. Open a terminal in this folder
3. Run this command (change the filename to match yours):
   ```bash
   ./scripts/process_tdms.sh raw/your_temperature_file.tdms
   ```

This will:

- Read your temperature data
- Create summaries for different time periods
- Save everything in the `processed/` folder
- Create log files you can check

### For Weather Data (CSV Files)

1. Put your weather CSV file(s) in the `raw/` folder
2. Open a terminal in this folder
3. Run one of these commands:

   For a single file:

   ```bash
   ./scripts/process_weather.sh raw/your_weather_file.csv
   ```

   For all CSV files in a folder:

   ```bash
   ./scripts/process_weather.sh raw/weather_files/
   ```

### Create Plots

After processing your data, you can create plots:

```bash
cd scripts
python3 plot_all.py
```

This creates plots for all your data in the `plots/` folder.

## Using a Database Instead of Files

The scripts currently save data to CSV files. Here's how to use a database instead:

### 1. Weekly Data (temp_weather_weekly.py)

This script already has database code. To use it:

1. Open `scripts/temp_weather_weekly.py`
2. Find the `initialize_database()` function
3. Change the database connection code
4. Update the table schema if needed

Example for PostgreSQL:

```python
import psycopg2

# Add at the top of the file
DB_CONFIG = {
    'host': 'your_database_host',
    'database': 'your_database_name',
    'user': 'your_username',
    'password': 'your_password'
}

def initialize_database():
    """Connect to PostgreSQL database."""
    return psycopg2.connect(**DB_CONFIG)

def write_to_database(df, conn, data_type):
    """Write data to PostgreSQL."""
    cursor = conn.cursor()
    # Your INSERT statements here
    conn.commit()
```

### 2. Other Time Intervals

To save 15-minute, hourly, or daily data to the database:

1. Open the script you want to modify:

   - `temp_weather_15min.py`
   - `temp_weather_hourly.py`
   - `temp_weather_daily.py`

2. Add these functions (modify for your database):

   ```python
   def initialize_database():
       """Database connection setup."""
       # Your connection code here
       pass

   def write_to_database(df, data_type):
       """Save data to database."""
       conn = initialize_database()
       # Your INSERT statements here
       conn.close()
   ```

3. Replace the CSV saving code with database code:
   ```python
   # Find where it saves to CSV (usually in save_aggregated_data())
   # Replace or add:
   write_to_database(df, data_type)
   ```

## Troubleshooting

If you have problems:

1. Check the log files:

   - Look in the `logs/` folder
   - Find the log file matching your script name
   - Look for ERROR or WARNING messages

2. Common issues:

   - Scripts not executable? Run: `chmod +x scripts/*.sh`
   - Can't find files? Check your paths and current directory
   - Database errors? Check your connection settings
   - Missing data? Check your input file format

3. Script-specific logs:
   - Temperature processing: `logs/temp_weather_*.log`
   - Weather processing: `logs/weather_station.log`
   - Plotting: `logs/plotting.log`

## Need More Help?

- Check the main README in the project root folder
- Look at the example data files in `raw/examples/` (if provided)
- Make sure you have all required Python packages installed
