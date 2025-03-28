"""
Combined Plotting Script
-----------------------
This script runs both temperature and weather plotting scripts.
"""

import os
import sys
import subprocess

def main():
    """Run both temperature and weather plotting scripts."""
    print("===== Pavement Data Plotting Tool =====")
    print("Starting temperature plotting...")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run temperature plotting script
    temp_script = os.path.join(script_dir, "plot_temperature.py")
    try:
        subprocess.run([sys.executable, temp_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running temperature plotting script: {e}")
    
    print("\nStarting weather plotting...")
    
    # Run weather plotting script
    weather_script = os.path.join(script_dir, "plot_weather.py")
    try:
        subprocess.run([sys.executable, weather_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running weather plotting script: {e}")
    
    print("\n===== Plotting complete! =====")
    print("Temperature plots saved to: pavement_data/plots/temperature/")
    print("Weather plots saved to: pavement_data/plots/weather/")

if __name__ == "__main__":
    main() 