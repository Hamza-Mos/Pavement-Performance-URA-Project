"""
Plot Utilities for Pavement Temperature and Weather Data
-------------------------------------------------------
Common utility functions for plotting temperature and weather data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from datetime import datetime

# Set style for plots
plt.style.use('ggplot')

def load_data(file_path):
    """
    Load data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with data
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None
    
    try:
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert date to datetime if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        # Convert week_start and week_end to datetime if present
        if 'week_start' in df.columns:
            df['week_start'] = pd.to_datetime(df['week_start'])
        if 'week_end' in df.columns:
            df['week_end'] = pd.to_datetime(df['week_end'])
            
        return df
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def setup_plot_style(time_interval):
    """
    Set up common plot style based on time interval.
    
    Args:
        time_interval: Time interval ('15min', 'hourly', 'daily', or 'weekly')
        
    Returns:
        Figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set common style
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set date formatter based on time interval
    if time_interval == '15min':
        ax.xaxis.set_major_formatter(DateFormatter('%b %d, %H:%M'))
        plt.xticks(rotation=45)
    elif time_interval == 'hourly':
        ax.xaxis.set_major_formatter(DateFormatter('%b %d, %H:%M'))
        plt.xticks(rotation=45)
    elif time_interval == 'daily':
        ax.xaxis.set_major_formatter(DateFormatter('%b %d, %Y'))
        plt.xticks(rotation=30)
    elif time_interval == 'weekly':
        ax.xaxis.set_major_formatter(DateFormatter('%b %d, %Y'))
        plt.xticks(rotation=30)
    
    return fig, ax

def plot_temperature(df, sensor_id, output_dir, time_interval):
    """
    Plot temperature data for a specific sensor and save the plot.
    
    Args:
        df: DataFrame with temperature data
        sensor_id: ID of the sensor to plot
        output_dir: Directory to save the plot
        time_interval: Time interval ('15min', 'hourly', 'daily', or 'weekly')
    """
    # Filter data for the specific sensor
    sensor_data = df[df['sensor_id'] == sensor_id].copy()
    
    if sensor_data.empty:
        print(f"No data for sensor {sensor_id}")
        return
    
    # Get x-axis based on time interval
    if time_interval in ['15min', 'hourly']:
        x_column = 'timestamp'
    elif time_interval == 'daily':
        x_column = 'date'
    elif time_interval == 'weekly':
        x_column = 'week_start'
    
    if x_column not in sensor_data.columns:
        print(f"Error: {x_column} column not found in data")
        return
    
    # Create the plot
    fig, ax = setup_plot_style(time_interval)
    
    # Convert any string columns to numeric, coercing errors to NaN
    for col in ['mean', 'min', 'max', 'std']:
        if col in sensor_data.columns:
            sensor_data[col] = pd.to_numeric(sensor_data[col], errors='coerce')
    
    # Plot the lines
    ax.plot(sensor_data[x_column], sensor_data['mean'], label='Mean', linewidth=2, color='#1f77b4')
    ax.plot(sensor_data[x_column], sensor_data['min'], label='Min', linewidth=1, linestyle='--', color='#2ca02c')
    ax.plot(sensor_data[x_column], sensor_data['max'], label='Max', linewidth=1, linestyle='--', color='#d62728')
    
    # Fill between min and max to show range
    ax.fill_between(sensor_data[x_column], sensor_data['min'], sensor_data['max'], 
                    alpha=0.2, color='#1f77b4', label='Range')
    
    # Add titles and labels
    sensor_name = sensor_id.replace('_Temperature', '')
    title = f"Temperature Data: {sensor_name} ({time_interval})"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    
    # Add legend
    ax.legend(loc='best', frameon=True)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"{sensor_name}_{time_interval}.png"
    output_path = os.path.join(output_dir, plot_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved plot for {sensor_name} ({time_interval}) to {output_path}")

def plot_weather(df, metric_name, output_dir, time_interval):
    """
    Plot weather data for a specific metric and save the plot.
    
    Args:
        df: DataFrame with weather data
        metric_name: Name of the metric to plot
        output_dir: Directory to save the plot
        time_interval: Time interval ('15min', 'hourly', 'daily', or 'weekly')
    """
    # Filter data for the specific metric
    metric_data = df[df['metric_name'] == metric_name].copy()
    
    if metric_data.empty:
        print(f"No data for metric {metric_name}")
        return
    
    # Get x-axis based on time interval
    if time_interval in ['15min', 'hourly']:
        x_column = 'timestamp'
    elif time_interval == 'daily':
        x_column = 'date'
    elif time_interval == 'weekly':
        x_column = 'week_start'
    
    if x_column not in metric_data.columns:
        print(f"Error: {x_column} column not found in data")
        return
    
    # Create the plot
    fig, ax = setup_plot_style(time_interval)
    
    # Plot the lines
    ax.plot(metric_data[x_column], metric_data['mean'], label='Mean', linewidth=2, color='#1f77b4')
    ax.plot(metric_data[x_column], metric_data['min'], label='Min', linewidth=1, linestyle='--', color='#2ca02c')
    ax.plot(metric_data[x_column], metric_data['max'], label='Max', linewidth=1, linestyle='--', color='#d62728')
    
    # Fill between min and max to show range
    ax.fill_between(metric_data[x_column], metric_data['min'], metric_data['max'], 
                   alpha=0.2, color='#1f77b4', label='Range')
    
    # Add titles and labels
    title = f"Weather Data: {metric_name} ({time_interval})"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    
    # Set appropriate y-axis label based on metric
    if 'temperature' in metric_name.lower():
        y_label = 'Temperature (°C)'
    elif 'humidity' in metric_name.lower():
        y_label = 'Humidity (%)'
    elif 'pressure' in metric_name.lower():
        y_label = 'Pressure (kPa)'
    elif 'wind' in metric_name.lower() and 'speed' in metric_name.lower():
        y_label = 'Wind Speed (km/h)'
    elif 'wind' in metric_name.lower() and 'gust' in metric_name.lower():
        y_label = 'Wind Gust (km/h)'
    elif 'precipitation' in metric_name.lower():
        y_label = 'Precipitation (mm)'
    else:
        y_label = metric_name
    
    ax.set_ylabel(y_label, fontsize=12)
    
    # Add legend
    ax.legend(loc='best', frameon=True)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"{metric_name}_{time_interval}.png"
    output_path = os.path.join(output_dir, plot_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved plot for {metric_name} ({time_interval}) to {output_path}") 