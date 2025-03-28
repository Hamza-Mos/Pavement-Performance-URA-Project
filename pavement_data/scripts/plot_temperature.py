"""
Temperature Data Plotting Script
--------------------------------
This script plots temperature data from CSV files at different time intervals
(15min, hourly, daily, weekly) for each sensor and saves the plots to a directory.
"""

import os
import sys
import argparse
import glob
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

# Import utility functions
import plot_utils as utils

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plot temperature data')
    parser.add_argument('--input-dir', type=str, default='../processed', help='Directory containing processed data files')
    parser.add_argument('--output-dir', type=str, default='../plots/temperature', help='Directory for output plots')
    parser.add_argument('--interval', type=str, choices=['15min', 'hourly', 'daily', 'weekly', 'all'], default='all',
                        help='Time interval to plot (default: all)')
    return parser.parse_args()

def plot_15min_data(input_dir, output_dir):
    """Plot 15-minute temperature data."""
    print("\nProcessing 15-minute temperature data...")
    
    # Find all 15-minute temperature data files
    data_path = os.path.join(input_dir, '15min')
    file_pattern = os.path.join(data_path, 'temperature_15min_*.csv')
    files = glob.glob(file_pattern)
    
    if not files:
        print("No 15-minute temperature data files found.")
        return
    
    # Create output directory if it doesn't exist
    output_path = os.path.join(output_dir, '15min')
    os.makedirs(output_path, exist_ok=True)

    # Process each file
    for file in sorted(files):
        print(f"Processing file: {os.path.basename(file)}")
        df = utils.load_data(file)
        
        if df is None or df.empty:
            continue
            
        # Get unique sensors
        sensors = df['sensor_id'].unique()
        
        # Plot data for each sensor
        for sensor in sensors:
            utils.plot_temperature(df, sensor, output_path, '15min')
    
    print(f"Saved 15-minute plots to {output_path}")

def plot_hourly_data(input_dir, output_dir):
    """Plot hourly temperature data."""
    print("\nProcessing hourly temperature data...")
    
    # Find all hourly temperature data files
    data_path = os.path.join(input_dir, 'hourly')
    file_pattern = os.path.join(data_path, 'temperature_hourly_*.csv')
    files = glob.glob(file_pattern)
    
    if not files:
        print("No hourly temperature data files found.")
        return
    
    # Create output directory if it doesn't exist
    output_path = os.path.join(output_dir, 'hourly')
    os.makedirs(output_path, exist_ok=True)

    # Process each file
    for file in sorted(files):
        print(f"Processing file: {os.path.basename(file)}")
        df = utils.load_data(file)
        
        if df is None or df.empty:
            continue
            
        # Get unique sensors
        sensors = df['sensor_id'].unique()
        
        # Plot data for each sensor
        for sensor in sensors:
            utils.plot_temperature(df, sensor, output_path, 'hourly')
    
    print(f"Saved hourly plots to {output_path}")

def plot_daily_data(input_dir, output_dir):
    """Plot daily temperature data."""
    print("\nProcessing daily temperature data...")
    
    # Find all daily temperature data files
    data_path = os.path.join(input_dir, 'daily')
    file_pattern = os.path.join(data_path, 'temperature_daily_*.csv')
    files = glob.glob(file_pattern)
    
    if not files:
        print("No daily temperature data files found.")
        return
    
    # Create output directory if it doesn't exist
    output_path = os.path.join(output_dir, 'daily')
    os.makedirs(output_path, exist_ok=True)

    # Process each file
    for file in sorted(files):
        print(f"Processing file: {os.path.basename(file)}")
        df = utils.load_data(file)
        
        if df is None or df.empty:
            continue
            
        # Get unique sensors
        sensors = df['sensor_id'].unique()
        
        # Plot data for each sensor
        for sensor in sensors:
            utils.plot_temperature(df, sensor, output_path, 'daily')
    
    print(f"Saved daily plots to {output_path}")

def plot_weekly_data(input_dir, output_dir):
    """Plot weekly temperature data."""
    print("\nProcessing weekly temperature data...")
    
    # Find all weekly temperature data files
    data_path = os.path.join(input_dir, 'weekly')
    file_pattern = os.path.join(data_path, 'temperature_weekly_*.csv')
    files = glob.glob(file_pattern)
    
    if not files:
        print("No weekly temperature data files found.")
        return
    
    # Create output directory if it doesn't exist
    output_path = os.path.join(output_dir, 'weekly')
    os.makedirs(output_path, exist_ok=True)

    # Process each file
    for file in sorted(files):
        print(f"Processing file: {os.path.basename(file)}")
        df = utils.load_data(file)
        
        if df is None or df.empty:
            continue
            
        # Get unique sensors
        sensors = df['sensor_id'].unique()
        
        # Plot data for each sensor
        for sensor in sensors:
            utils.plot_temperature(df, sensor, output_path, 'weekly')
    
    print(f"Saved weekly plots to {output_path}")

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
        sys.exit(1) 