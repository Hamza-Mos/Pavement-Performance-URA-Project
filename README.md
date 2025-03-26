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

3. Run the data processing script:

```bash
python temp_weather_data_processing.py
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
