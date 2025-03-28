# Pavement Data Processing and Visualization

## Data Processing

### Process Temperature Data

```bash
# Process a TDMS file with temperature data
./scripts/process_tdms.sh /path/to/temperature_data.tdms
```

### Process Weather Data

```bash
# Process a weather CSV file
./scripts/process_weather.sh /path/to/weather_data.csv

# Or process all CSV files in a directory
./scripts/process_weather.sh /directory/with/weather/files
```

## Generate Plots

```bash
# Generate all plots (temperature and weather)
cd pavement_data/scripts
python3 plot_all.py
```

This will create plots for all sensors and metrics at all time intervals (15min, hourly, daily, weekly)
in the `plots/temperature/` and `plots/weather/` directories.

## Troubleshooting

If you encounter issues:

- Make sure scripts are executable: `chmod +x scripts/*.sh`
- Check logs in the `logs/` directory for error messages
