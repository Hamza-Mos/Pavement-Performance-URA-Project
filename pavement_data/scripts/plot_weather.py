#!/usr/bin/env python3
"""
Weather Data Plotting Script
----------------------------
This script plots weather data from CSV files at different time intervals
(15min, hourly, daily, weekly) for each metric and saves the plots to a directory.
"""

import os
import sys
import argparse
import glob
from datetime import datetime
import traceback

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plot weather data')
    parser.add_argument('--input-dir', type=str, default='../processed_weather', help='Directory containing processed weather data files')
    parser.add_argument('--output-dir', type=str, default='../plots/weather', help='Directory for output plots')
    parser.add_argument('--interval', type=str, choices=['15min', 'hourly', 'daily', 'weekly', 'all'], default='all',
                        help='Time interval to plot (default: all)')
    return parser.parse_args()

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

def plot_weather_metric(df, metric_name, output_dir, time_interval):
    """
    Plot weather data for a specific metric and save the plot.
    
    Args:
        df: DataFrame with weather data
        metric_name: Name of the metric to plot
        output_dir: Directory to save the plot
        time_interval: Time interval ('15min', 'hourly', 'daily', or 'weekly')
    """
    try:
        # Filter data for the specific metric
        if isinstance(metric_name, str):
            metric_data = df[df['metric_name'].astype(str) == metric_name].copy()
        else:
            # Handle non-string metric names
            metric_data = df[df['metric_name'] == metric_name].copy()
        
        if metric_data.empty:
            print(f"No data for metric {metric_name}")
            return False
        
        # Get x-axis based on time interval
        if time_interval in ['15min', 'hourly']:
            x_column = 'timestamp'
        elif time_interval == 'daily':
            x_column = 'date'
        elif time_interval == 'weekly':
            x_column = 'week_start'
        
        if x_column not in metric_data.columns:
            print(f"Error: {x_column} column not found in data")
            return False
        
        # Convert all data columns to numeric
        for col in ['mean', 'min', 'max', 'std', 'count']:
            if col in metric_data.columns:
                metric_data[col] = pd.to_numeric(metric_data[col], errors='coerce')
        
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
        metric_str = str(metric_name).lower()
        if 'temperature' in metric_str:
            y_label = 'Temperature (°C)'
        elif 'humidity' in metric_str:
            y_label = 'Humidity (%)'
        elif 'pressure' in metric_str:
            y_label = 'Pressure (kPa)'
        elif 'wind' in metric_str and 'speed' in metric_str:
            y_label = 'Wind Speed (km/h)'
        elif 'wind' in metric_str and 'gust' in metric_str:
            y_label = 'Wind Gust (km/h)'
        elif 'precipitation' in metric_str:
            y_label = 'Precipitation (mm)'
        elif 'dew' in metric_str:
            y_label = 'Dew Point (°C)'
        else:
            y_label = str(metric_name)
        
        ax.set_ylabel(y_label, fontsize=12)
        
        # Add legend
        ax.legend(loc='best', frameon=True)
        
        # Tight layout for better spacing
        plt.tight_layout()
        
        # Save the plot
        # Clean up metric name for filename
        clean_metric = str(metric_name).replace(' ', '_').replace('/', '_').replace('\\', '_')
        plot_filename = f"{clean_metric}_{time_interval}.png"
        output_path = os.path.join(output_dir, plot_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved plot for {metric_name} ({time_interval}) to {output_path}")
        return True
    except Exception as e:
        print(f"Error plotting {metric_name}: {str(e)}")
        traceback.print_exc()
        return False

def plot_15min_data(input_dir, output_dir):
    """Plot 15-minute weather data."""
    print("\nProcessing 15-minute weather data...")
    
    # Find all 15-minute weather data files
    data_path = os.path.join(input_dir, '15min')
    file_pattern = os.path.join(data_path, 'weather_15min_*.csv')
    files = glob.glob(file_pattern)
    
    if not files:
        print("No 15-minute weather data files found.")
        return
    
    # Create output directory if it doesn't exist
    output_path = os.path.join(output_dir, '15min')
    os.makedirs(output_path, exist_ok=True)
    
    plots_created = 0
    
    # Process each file
    for file in sorted(files):
        print(f"Processing file: {os.path.basename(file)}")
        
        # Read the file
        try:
            df = pd.read_csv(file)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if df.empty:
                print(f"File {file} is empty.")
                continue
                
            # Get unique metrics
            if 'metric_name' not in df.columns:
                print(f"No metric_name column in {file}.")
                continue
                
            metrics = df['metric_name'].unique()
            
            # Plot data for each metric
            for metric in metrics:
                if plot_weather_metric(df, metric, output_path, '15min'):
                    plots_created += 1
        
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
    
    print(f"Created {plots_created} 15-minute plots in {output_path}")

def plot_hourly_data(input_dir, output_dir):
    """Plot hourly weather data."""
    print("\nProcessing hourly weather data...")
    
    # Find all hourly weather data files
    data_path = os.path.join(input_dir, 'hourly')
    file_pattern = os.path.join(data_path, 'weather_hourly_*.csv')
    files = glob.glob(file_pattern)
    
    if not files:
        print("No hourly weather data files found.")
        return
    
    # Create output directory if it doesn't exist
    output_path = os.path.join(output_dir, 'hourly')
    os.makedirs(output_path, exist_ok=True)
    
    plots_created = 0
    
    # Process each file
    for file in sorted(files):
        print(f"Processing file: {os.path.basename(file)}")
        
        # Read the file
        try:
            df = pd.read_csv(file)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if df.empty:
                print(f"File {file} is empty.")
                continue
                
            # Get unique metrics
            if 'metric_name' not in df.columns:
                print(f"No metric_name column in {file}.")
                continue
                
            metrics = df['metric_name'].unique()
            
            # Plot data for each metric
            for metric in metrics:
                if plot_weather_metric(df, metric, output_path, 'hourly'):
                    plots_created += 1
        
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
    
    print(f"Created {plots_created} hourly plots in {output_path}")

def plot_daily_data(input_dir, output_dir):
    """Plot daily weather data."""
    print("\nProcessing daily weather data...")
    
    # Find all daily weather data files
    data_path = os.path.join(input_dir, 'daily')
    file_pattern = os.path.join(data_path, 'weather_daily_*.csv')
    files = glob.glob(file_pattern)
    
    if not files:
        print("No daily weather data files found.")
        return
    
    # Create output directory if it doesn't exist
    output_path = os.path.join(output_dir, 'daily')
    os.makedirs(output_path, exist_ok=True)
    
    plots_created = 0
    
    # Process each file
    for file in sorted(files):
        print(f"Processing file: {os.path.basename(file)}")
        
        # Read the file
        try:
            df = pd.read_csv(file)
            
            # Convert date to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            if df.empty:
                print(f"File {file} is empty.")
                continue
                
            # Get unique metrics
            if 'metric_name' not in df.columns:
                print(f"No metric_name column in {file}.")
                continue
                
            metrics = df['metric_name'].unique()
            
            # Plot data for each metric
            for metric in metrics:
                if plot_weather_metric(df, metric, output_path, 'daily'):
                    plots_created += 1
        
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
    
    print(f"Created {plots_created} daily plots in {output_path}")

def plot_weekly_data(input_dir, output_dir):
    """Plot weekly weather data."""
    print("\nProcessing weekly weather data...")
    
    # Find all weekly weather data files
    data_path = os.path.join(input_dir, 'weekly')
    file_pattern = os.path.join(data_path, 'weather_weekly_*.csv')
    files = glob.glob(file_pattern)
    
    if not files:
        print("No weekly weather data files found.")
        return
    
    # Create output directory if it doesn't exist
    output_path = os.path.join(output_dir, 'weekly')
    os.makedirs(output_path, exist_ok=True)
    
    plots_created = 0
    
    # Process each file
    for file in sorted(files):
        print(f"Processing file: {os.path.basename(file)}")
        
        # Read the file
        try:
            df = pd.read_csv(file)
            
            # Convert week_start to datetime
            if 'week_start' in df.columns:
                df['week_start'] = pd.to_datetime(df['week_start'])
            if 'week_end' in df.columns:
                df['week_end'] = pd.to_datetime(df['week_end'])
            
            if df.empty:
                print(f"File {file} is empty.")
                continue
                
            # Get unique metrics
            if 'metric_name' not in df.columns:
                print(f"No metric_name column in {file}.")
                continue
                
            metrics = df['metric_name'].unique()
            
            # Plot data for each metric
            for metric in metrics:
                if plot_weather_metric(df, metric, output_path, 'weekly'):
                    plots_created += 1
        
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
    
    print(f"Created {plots_created} weekly plots in {output_path}")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Convert relative paths to absolute paths
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot data based on specified interval
    if args.interval == 'all' or args.interval == '15min':
        plot_15min_data(input_dir, output_dir)
        
    if args.interval == 'all' or args.interval == 'hourly':
        plot_hourly_data(input_dir, output_dir)
        
    if args.interval == 'all' or args.interval == 'daily':
        plot_daily_data(input_dir, output_dir)
        
    if args.interval == 'all' or args.interval == 'weekly':
        plot_weekly_data(input_dir, output_dir)
    
    print("\nPlotting complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc() 