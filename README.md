# Electric Vehicle Adoption Forecasting System 🚗⚡

A comprehensive machine learning project for analyzing and forecasting electric vehicle (EV) adoption trends across the United States using real-world data from 2017-2024.

## 📊 Project Overview

This project analyzes electric vehicle adoption patterns and creates forecasting models to predict future EV growth. It processes real CSV data containing EV statistics from all 50 US states and generates comprehensive analysis reports with visualizations.

## 🎯 Key Features

- **Data Processing**: Handles real-world EV data with automatic cleaning and preprocessing
- **Multiple ML Models**: Implements 4 different forecasting models (Linear Regression, Ridge, Random Forest, Gradient Boosting)
- **Comprehensive Analysis**: Generates detailed statistical analysis and growth projections
- **Rich Visualizations**: Creates multiple charts showing historical trends and future forecasts
- **Automated Pipeline**: Complete end-to-end analysis with a single command

## 📁 Project Structure

```
Electrical_vehical_prediction/
├── main.py                     # Main analysis script
├── data_loader.py              # Data loading and preprocessing
├── ev_forecasting_model.py     # Machine learning models
├── run.py                      # Automated pipeline runner
├── requirements.txt            # Python dependencies
├── 3ae033f50fa345051652.csv   # Real EV data (2017-2024)
├── .gitignore                  # Git ignore file
├── README.md                   # This file
└── Generated Files:
    ├── eda_plots.png           # Exploratory data analysis
    ├── model_predictions.png   # Model performance comparison
    ├── ev_forecast_plot.png    # Historical vs forecasted trends
    ├── forecasting_report.txt  # Detailed analysis report
    └── ev_forecasting_model.pkl # Trained model
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/electrical-vehicle-prediction.git
   cd electrical-vehicle-prediction
   ```

2. **Run the automated pipeline**
   ```bash
   python run.py
   ```
   
   OR

3. **Run manually**
   ```bash
   pip install -r requirements.txt
   python main.py
   ```

## 📈 Results

### Data Summary
- **Dataset**: 25,112 records from all 50 US states
- **Time Period**: 2017-2024
- **Total EVs Analyzed**: 6,195,315 vehicles
- **Vehicle Types**: Battery Electric Vehicles (BEV) and Plug-in Hybrid Electric Vehicles (PHEV)

### Model Performance
| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| **Linear Regression** | 1.0000 | 4.80 | 3.85 |
| Ridge Regression | 1.0000 | 4.80 | 3.86 |
| Random Forest | 0.9998 | 26.41 | 2.67 |
| Gradient Boosting | 0.9995 | 39.52 | 5.31 |

### 5-Year Forecasts (2025-2029)
- **2025**: 397,020 EVs (+11.9% annual growth projected)
- **2026**: 457,110 EVs
- **2027**: 526,210 EVs
- **2028**: 605,673 EVs
- **2029**: 697,054 EVs

## 📊 Generated Visualizations

The system automatically generates:

1. **Exploratory Data Analysis** (`eda_plots.png`)
   - EV adoption trends over time
   - Regional distribution analysis
   - Vehicle type breakdown
   - Feature correlation heatmap

2. **Model Performance** (`model_predictions.png`)
   - Actual vs predicted comparisons
   - R² scores for each model
   - Performance scatter plots

3. **Forecast Visualization** (`ev_forecast_plot.png`)
   - Historical data trends
   - Future projections
   - Growth trajectory analysis

## 🛠️ Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning models
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **joblib**: Model serialization

### Models Implemented
1. **Linear Regression**: Simple linear relationship modeling
2. **Ridge Regression**: Regularized linear regression
3. **Random Forest**: Ensemble method with multiple decision trees
4. **Gradient Boosting**: Sequential ensemble method

### Data Processing Pipeline
1. Load CSV data with automatic format detection
2. Clean numeric values (handle comma separators)
3. Extract temporal features (years)
4. Encode categorical variables
5. Create feature matrix and target variables
6. Train multiple models with cross-validation
7. Generate forecasts and visualizations

## 📝 Output Files

- `forecasting_report.txt`: Comprehensive analysis summary
- `ev_forecasting_model.pkl`: Trained model for future predictions
- `*.png`: Various visualization charts
- Console output: Real-time progress and results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Contact

- **Project Link**: [https://github.com/yourusername/electrical-vehicle-prediction](https://github.com/yourusername/electrical-vehicle-prediction)
- **Issues**: [https://github.com/yourusername/electrical-vehicle-prediction/issues](https://github.com/yourusername/electrical-vehicle-prediction/issues)

## 🙏 Acknowledgments

- Data source: Real EV adoption statistics from US states
- Built with Python and scikit-learn
- Visualization powered by matplotlib and seaborn

---

⚡ **Ready to forecast the electric future!** ⚡
