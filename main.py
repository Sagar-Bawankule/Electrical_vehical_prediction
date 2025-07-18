# e:\Electrical_vehical_prediction\main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import EVDataLoader
from ev_forecasting_model import EVForecastingModel

def main():
    print("=" * 60)
    print("Electric Vehicle Adoption Forecasting System")
    print("=" * 60)
    
    # Initialize data loader with the CSV file path
    csv_file_path = "3ae033f50fa345051652.csv"
    data_loader = EVDataLoader(csv_file_path)
    
    print("\n1. Loading and preprocessing data...")
    data = data_loader.load_data()
    data = data_loader.preprocess_data()
    
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    print("\nDataset Overview:")
    print(data.describe())
    
    print("\n2. Creating visualizations...")
    data_loader.visualize_data()
    
    print("\n3. Preparing features and target...")
    X, y = data_loader.get_features_target()
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    print("\n4. Training forecasting models...")
    model = EVForecastingModel()
    model_performance = model.train_models(X, y)
    
    print("\n5. Plotting model performance...")
    model.plot_predictions(model_performance)
    
    print("\n6. Creating future forecasts...")
    
    base_year = data['Year'].max()
    base_data = data[data['Year'] == base_year].copy()
    
    feature_cols = list(X.columns)
    base_features = base_data[feature_cols]
    
    forecasts = model.forecast_future(base_features, years_ahead=5)
    
    print("\nForecasting Results:")
    print("-" * 40)
    for forecast in forecasts:
        print(f"Year {forecast['Year']}: {forecast['Total_Predicted_EVs']:,.0f} total EVs")
        print(f"  Average per scenario: {forecast['Average_EVs_per_Region']:,.0f} EVs")
    
    create_forecast_plot(data, forecasts)
    
    print("\n7. Saving the trained model...")
    model.save_model('ev_forecasting_model.pkl')
    
    generate_summary_report(data, model_performance, forecasts)
    
    print("\n" + "=" * 60)
    print("Analysis completed successfully!")
    print("Check the generated files for detailed results.")
    print("=" * 60)

def create_forecast_plot(historical_data, forecasts):
    """Create visualization of historical data and forecasts"""
    plt.figure(figsize=(12, 8))
    
    # Historical data
    yearly_historical = historical_data.groupby('Year')['EV_Count'].sum()
    plt.plot(yearly_historical.index, yearly_historical.values, 
             marker='o', linewidth=2, label='Historical Data', color='blue')
    
    # Forecasted data
    forecast_years = [f['Year'] for f in forecasts]
    forecast_totals = [f['Total_Predicted_EVs'] for f in forecasts]
    
    plt.plot(forecast_years, forecast_totals, 
             marker='s', linewidth=2, linestyle='--', label='Forecasted Data', color='red')
    
    plt.xlabel('Year')
    plt.ylabel('Total EV Count')
    plt.title('Electric Vehicle Adoption: Historical vs Forecasted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add annotation for forecast start
    if len(yearly_historical) > 0:
        last_historical_year = yearly_historical.index[-1]
        plt.axvline(x=last_historical_year, color='gray', linestyle=':', alpha=0.7)
        plt.text(last_historical_year, plt.ylim()[1]*0.8, 'Forecast Start', 
                rotation=90, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig('ev_forecast_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_report(data, model_performance, forecasts):
    """Generate a summary report of the analysis"""
    report = []
    report.append("ELECTRIC VEHICLE ADOPTION FORECASTING REPORT")
    report.append("=" * 50)
    
    # Data summary
    report.append(f"\nDATA SUMMARY:")
    report.append(f"- Dataset size: {data.shape[0]} records")
    report.append(f"- Time period: {data['Year'].min()} - {data['Year'].max()}")
    report.append(f"- Total EVs in dataset: {data['EV_Count'].sum():,}")
    
    yearly_totals = data.groupby('Year')['EV_Count'].sum()
    if len(yearly_totals) > 1:
        growth_rate = ((yearly_totals.iloc[-1] / yearly_totals.iloc[0]) ** (1/(len(yearly_totals)-1)) - 1) * 100
        report.append(f"- Average annual growth: {growth_rate:.1f}%")
    
    # Model performance
    report.append(f"\nMODEL PERFORMANCE:")
    for name, perf in model_performance.items():
        report.append(f"- {name}:")
        report.append(f"  * R² Score: {perf['r2']:.4f}")
        report.append(f"  * RMSE: {perf['rmse']:.2f}")
        report.append(f"  * MAE: {perf['mae']:.2f}")
    
    # Forecasts
    report.append(f"\nFORECASTS (Next 5 Years):")
    for forecast in forecasts:
        report.append(f"- Year {forecast['Year']}: {forecast['Total_Predicted_EVs']:,.0f} EVs")
    
    # Growth projections
    if len(forecasts) >= 2:
        annual_growth = ((forecasts[-1]['Total_Predicted_EVs'] / forecasts[0]['Total_Predicted_EVs']) ** (1/len(forecasts)) - 1) * 100
        report.append(f"\nPROJECTED ANNUAL GROWTH RATE: {annual_growth:.1f}%")
    
    # Save report
    with open('forecasting_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print("\nSummary Report:")
    print('\n'.join(report))

if __name__ == "__main__":
    main()