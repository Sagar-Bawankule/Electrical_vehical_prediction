# e:\Electrical_vehical_prediction\ev_forecasting_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class EVForecastingModel:
    def __init__(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(),
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def train_models(self, X, y, test_size=0.2):
        """Train multiple models and select the best one"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model_performance = {}
        
        print("Training models...")
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            if name in ['linear_regression', 'ridge_regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            model_performance[name] = {
                'model': model,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred,
                'actual': y_test
            }
            
            print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
        
        # Select best model based on R² score
        best_name = max(model_performance.keys(), 
                       key=lambda x: model_performance[x]['r2'])
        
        self.best_model = model_performance[best_name]['model']
        self.best_model_name = best_name
        self.is_fitted = True
        
        print(f"\nBest model: {best_name}")
        print(f"Best R² score: {model_performance[best_name]['r2']:.4f}")
        
        return model_performance
    
    def predict(self, X):
        """Make predictions using the best model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Please train the model first.")
        
        if self.best_model_name in ['linear_regression', 'ridge_regression']:
            X_scaled = self.scaler.transform(X)
            return self.best_model.predict(X_scaled)
        else:
            return self.best_model.predict(X)
    
    def forecast_future(self, base_data, years_ahead=5):
        """Forecast EV adoption for future years"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Please train the model first.")
        
        forecasts = []
        current_year = base_data['Year'].max()
        
        for year_offset in range(1, years_ahead + 1):
            future_year = current_year + year_offset
            
            # Create future scenarios
            future_data = base_data.copy()
            future_data['Year'] = future_year
            
            # Adjust features based on trends
            # Assume charging stations grow with EV adoption
            growth_rate = 1.15  # 15% annual growth assumption
            future_data['Charging_Stations'] = future_data['Charging_Stations'] * (growth_rate ** year_offset)
            
            # GDP growth assumption
            future_data['GDP_Per_Capita'] = future_data['GDP_Per_Capita'] * (1.03 ** year_offset)
            
            # Make predictions
            predictions = self.predict(future_data)
            
            forecast_summary = {
                'Year': future_year,
                'Total_Predicted_EVs': predictions.sum(),
                'Average_EVs_per_Region': predictions.mean(),
                'Predictions_by_Scenario': predictions.tolist()
            }
            
            forecasts.append(forecast_summary)
        
        return forecasts
    
    def plot_predictions(self, model_performance):
        """Plot model predictions vs actual values"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (name, performance) in enumerate(model_performance.items()):
            if idx < 4:
                axes[idx].scatter(performance['actual'], performance['predictions'], alpha=0.6)
                axes[idx].plot([performance['actual'].min(), performance['actual'].max()],
                              [performance['actual'].min(), performance['actual'].max()], 'r--', lw=2)
                axes[idx].set_xlabel('Actual EV Count')
                axes[idx].set_ylabel('Predicted EV Count')
                axes[idx].set_title(f'{name}\nR² = {performance["r2"]:.4f}')
        
        plt.tight_layout()
        plt.savefig('e:/Electrical_vehical_prediction/model_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Please train the model first.")
        
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.scaler = model_data['scaler']
        self.is_fitted = True
        print(f"Model loaded from {filepath}")