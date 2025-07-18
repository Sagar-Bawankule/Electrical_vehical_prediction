"""
Electric Vehicle Forecasting Model Runner
Execute this script to run the complete EV adoption forecasting pipeline.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing packages: {e}")
        return False
    return True

def run_main():
    """Run the main forecasting script"""
    print("\nStarting EV Forecasting Analysis...")
    try:
        subprocess.check_call([sys.executable, "main.py"])
        print("✓ Analysis completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running analysis: {e}")
        return False
    return True

if __name__ == "__main__":
    print("Electric Vehicle Adoption Forecasting System")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("main.py"):
        print("✗ main.py not found. Please run this script from the project directory.")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please install manually and try again.")
        sys.exit(1)
    
    # Run the main analysis
    if not run_main():
        print("Analysis failed. Please check the error messages above.")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("All tasks completed successfully!")
    print("Check the generated files:")
    print("- eda_plots.png (Exploratory Data Analysis)")
    print("- model_predictions.png (Model Performance)")
    print("- ev_forecast_plot.png (Future Forecasts)")
    print("- forecasting_report.txt (Summary Report)")
    print("- ev_forecasting_model.pkl (Trained Model)")
