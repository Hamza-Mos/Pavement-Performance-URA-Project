#!/bin/bash
# =============================================================
# Temperature & Weather Data Processing Pipeline
# =============================================================
# This script takes a TDMS file as input and runs the entire
# aggregation pipeline (15-min → hourly → daily → weekly → DB)
#
# Usage: ./process_tdms.sh /path/to/your/file.tdms
# =============================================================

# Check if a file was provided
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/your/file.tdms"
    exit 1
fi

# Get the input file
TDMS_FILE="$1"

# Check if the file exists
if [ ! -f "$TDMS_FILE" ]; then
    echo "Error: File not found: $TDMS_FILE"
    exit 1
fi

# Check if the file is a TDMS file
EXTENSION="${TDMS_FILE##*.}"
# Convert to lowercase using tr instead of bash operator
EXTENSION=$(echo "$EXTENSION" | tr '[:upper:]' '[:lower:]')
if [ "$EXTENSION" != "tdms" ]; then
    echo "Warning: File does not have .tdms extension. Continuing anyway..."
fi

# Set up directories
# Use absolute paths to avoid confusion
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/Users/hamza/Desktop/Pavement Performance URA Project/pavement_data"
RAW_DIR="$BASE_DIR/raw"
OUTPUT_DIR="$BASE_DIR/processed"
LOG_DIR="$BASE_DIR/logs"
DB_DIR="$BASE_DIR/database"

# Create directories if they don't exist
mkdir -p "$RAW_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$DB_DIR"
mkdir -p "$OUTPUT_DIR/15min"
mkdir -p "$OUTPUT_DIR/hourly"
mkdir -p "$OUTPUT_DIR/daily"
mkdir -p "$OUTPUT_DIR/weekly"

# Copy the TDMS file to the raw directory
echo "Copying TDMS file to raw directory..."
cp "$TDMS_FILE" "$RAW_DIR/"
FILENAME=$(basename "$TDMS_FILE")

# Set up logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/process_$TIMESTAMP.log"

echo "Starting processing at $(date)" > "$LOG_FILE"
echo "Input file: $TDMS_FILE" >> "$LOG_FILE"

# Run the 15-minute aggregation
echo "Running 15-minute aggregation..."
echo "Running 15-minute aggregation..." >> "$LOG_FILE"
cd "$SCRIPT_DIR" && python3 temp_weather_15min.py --input-file "$RAW_DIR/$FILENAME" --output-dir "$OUTPUT_DIR" >> "$LOG_FILE" 2>&1

# Check if the previous command was successful
if [ $? -ne 0 ]; then
    echo "Error: 15-minute aggregation failed. See log for details."
    echo "Error: 15-minute aggregation failed. See log for details." >> "$LOG_FILE"
    exit 1
fi

# Run the hourly aggregation
echo "Running hourly aggregation..."
echo "Running hourly aggregation..." >> "$LOG_FILE"
cd "$SCRIPT_DIR" && python3 temp_weather_hourly.py --input-dir "$OUTPUT_DIR/15min" --output-dir "$OUTPUT_DIR" >> "$LOG_FILE" 2>&1

# Check if the previous command was successful
if [ $? -ne 0 ]; then
    echo "Error: Hourly aggregation failed. See log for details."
    echo "Error: Hourly aggregation failed. See log for details." >> "$LOG_FILE"
    exit 1
fi

# Run the daily aggregation
echo "Running daily aggregation..."
echo "Running daily aggregation..." >> "$LOG_FILE"
cd "$SCRIPT_DIR" && python3 temp_weather_daily.py --input-dir "$OUTPUT_DIR/hourly" --output-dir "$OUTPUT_DIR" >> "$LOG_FILE" 2>&1

# Check if the previous command was successful
if [ $? -ne 0 ]; then
    echo "Error: Daily aggregation failed. See log for details."
    echo "Error: Daily aggregation failed. See log for details." >> "$LOG_FILE"
    exit 1
fi

# Run the weekly aggregation and database update
echo "Running weekly aggregation and database update..."
echo "Running weekly aggregation and database update..." >> "$LOG_FILE"
cd "$SCRIPT_DIR" && python3 temp_weather_weekly.py --input-dir "$OUTPUT_DIR/daily" --db-path "$DB_DIR/pavement_data.db" >> "$LOG_FILE" 2>&1

# Check if the previous command was successful
if [ $? -ne 0 ]; then
    echo "Error: Weekly aggregation failed. See log for details."
    echo "Error: Weekly aggregation failed. See log for details." >> "$LOG_FILE"
    exit 1
fi

echo "Processing completed successfully at $(date)"
echo "Processing completed successfully at $(date)" >> "$LOG_FILE"
echo "Results stored in:"
echo "  - 15-minute data: $OUTPUT_DIR/15min"
echo "  - Hourly data: $OUTPUT_DIR/hourly"
echo "  - Daily data: $OUTPUT_DIR/daily"
echo "  - Weekly data: $OUTPUT_DIR/weekly"
echo "  - Database: $DB_DIR/pavement_data.db"
echo "Log file: $LOG_FILE"

exit 0 