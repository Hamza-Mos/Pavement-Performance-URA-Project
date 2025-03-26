#!/bin/bash
# =============================================================
# Weather Station Data Processing Pipeline
# =============================================================
# This script processes weather station CSV files and runs the
# entire aggregation pipeline (15-min → hourly → daily → weekly)
#
# Usage: ./process_weather.sh /path/to/weather.csv
#    or: ./process_weather.sh /directory/with/weather/csv/files
# =============================================================

# Check if a file or directory was provided
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/weather.csv OR /path/to/weather/directory"
    exit 1
fi

# Get the input path
INPUT_PATH="$1"

# Set up directories
# Use absolute paths to avoid confusion
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WEATHER_DIR="$BASE_DIR/weather_data"
# Use a separate weather-specific output directory
WEATHER_OUTPUT_DIR="$BASE_DIR/processed_weather"
LOG_DIR="$BASE_DIR/logs"

# Create directories if they don't exist
mkdir -p "$WEATHER_DIR"
mkdir -p "$WEATHER_OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$WEATHER_OUTPUT_DIR/15min"
mkdir -p "$WEATHER_OUTPUT_DIR/hourly"
mkdir -p "$WEATHER_OUTPUT_DIR/daily"
mkdir -p "$WEATHER_OUTPUT_DIR/weekly"

# Handle input path appropriately
if [ -f "$INPUT_PATH" ]; then
    # Input is a file
    echo "Processing weather station CSV file: $INPUT_PATH"
    cp "$INPUT_PATH" "$WEATHER_DIR/"
    INPUT_ARG="--input-file $INPUT_PATH"
elif [ -d "$INPUT_PATH" ]; then
    # Input is a directory
    echo "Processing all weather station CSV files in directory: $INPUT_PATH"
    # Copy all CSV files to weather directory
    find "$INPUT_PATH" -name "*.csv" -exec cp {} "$WEATHER_DIR/" \;
    INPUT_ARG="--input-dir $WEATHER_DIR"
else
    echo "Error: $INPUT_PATH is neither a file nor a directory"
    exit 1
fi

# Set up logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/weather_process_$TIMESTAMP.log"

echo "Starting weather data processing at $(date)" > "$LOG_FILE"
echo "Input path: $INPUT_PATH" >> "$LOG_FILE"

# Run the weather data processing
echo "Running weather station data processing..."
echo "Running weather station data processing..." >> "$LOG_FILE"
cd "$SCRIPT_DIR" && python3 weather_station_processor.py $INPUT_ARG --output-dir "$WEATHER_OUTPUT_DIR" >> "$LOG_FILE" 2>&1

# Check if the previous command was successful
if [ $? -ne 0 ]; then
    echo "Error: Weather data processing failed. See log for details."
    echo "Error: Weather data processing failed. See log for details." >> "$LOG_FILE"
    exit 1
fi

echo "Processing completed successfully at $(date)"
echo "Processing completed successfully at $(date)" >> "$LOG_FILE"
echo "Results stored in:"
echo "  - 15-minute data: $WEATHER_OUTPUT_DIR/15min"
echo "  - Hourly data: $WEATHER_OUTPUT_DIR/hourly"
echo "  - Daily data: $WEATHER_OUTPUT_DIR/daily"
echo "  - Weekly data: $WEATHER_OUTPUT_DIR/weekly"
echo "Log file: $LOG_FILE"

exit 0 