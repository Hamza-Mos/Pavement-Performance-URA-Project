# Pavement Performance Analysis

This repository contains analysis and modeling of pavement performance data, including weather conditions and their impact on pavement deterioration.

## Project Structure

- `pavement_data/`: Contains the main dataset
- Various Jupyter notebooks for different modeling approaches:
  - EBM (Explainable Boosting Machine) models
  - Random Forest models
  - Ensemble models combining EBM and RF
- Data processing scripts for weather and pavement data
- Database design and schema files

## Setup

1. Create a virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Data Processing:

### Temperature Data Processing

Process temperature data from TDMS files using the shell script wrapper:

```bash
./pavement_data/scripts/process_tdms.sh /path/to/your/file.tdms
```

This executes the full pipeline (15-min → hourly → daily → weekly) for temperature data, storing results in `processed/` directory.

Or run individual aggregation scripts manually:

```bash
# Process raw temperature data to 15-minute intervals
python pavement_data/scripts/temp_weather_15min.py --input-file /path/to/your/file.tdms --output-dir pavement_data/processed

# Process 15-minute data to hourly intervals
python pavement_data/scripts/temp_weather_hourly.py --input-dir pavement_data/processed/15min --output-dir pavement_data/processed

# Process hourly data to daily intervals
python pavement_data/scripts/temp_weather_daily.py --input-dir pavement_data/processed/hourly --output-dir pavement_data/processed

# Process daily data to weekly intervals
python pavement_data/scripts/temp_weather_weekly.py --input-dir pavement_data/processed/daily --db-path pavement_data/database/pavement_data.db
```

### Weather Station Data Processing

Process external weather station data from CSV files:

```bash
./pavement_data/scripts/process_weather.sh /path/to/weather.csv
# OR process all CSVs in a directory
./pavement_data/scripts/process_weather.sh /directory/with/weather/csv/files
```

This processes weather station data through the same aggregation pipeline, storing results in the `processed_weather/` directory.

Or run the weather station processor directly:

```bash
python pavement_data/scripts/weather_station_processor.py --input-file /path/to/weather.csv --output-dir pavement_data/processed_weather
```

## Requirements

See `requirements.txt` for the list of Python packages required for this project.

## Data Sources

- Weather data from various sources
- Pavement performance measurements
- Temperature and environmental conditions

## Models

The project includes several modeling approaches:

- Explainable Boosting Machine (EBM)
- Random Forest
- Ensemble models combining EBM and RF

## License

[Add your chosen license here]
