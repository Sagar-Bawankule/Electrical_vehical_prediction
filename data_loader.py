# e:\Electrical_vehical_prediction\data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class EVDataLoader:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.data = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def load_data(self, file_path=None):
        """Load EV dataset from CSV file"""
        if file_path:
            self.file_path = file_path
        
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            
            # Convert the real CSV data format to our expected format
            self.data = self.convert_real_data_format()
            return self.data
        except (FileNotFoundError, TypeError):
            print("File not found or not specified. Creating sample data for demonstration.")
            return self.create_sample_data()
    
    def convert_real_data_format(self):
        """Convert the real CSV data format to the expected format for modeling"""
        print("Converting real data format...")
        
        # Clean numeric columns - remove commas and convert to numeric
        numeric_cols = ['Battery Electric Vehicles (BEVs)', 'Plug-In Hybrid Electric Vehicles (PHEVs)', 
                       'Electric Vehicle (EV) Total', 'Non-Electric Vehicle Total', 'Total Vehicles']
        
        for col in numeric_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Parse date column
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data['Year'] = self.data['Date'].dt.year
        
        # Create converted data
        data_converted = []
        
        for _, row in self.data.iterrows():
            # Skip rows with missing or invalid data
            if pd.isna(row['Battery Electric Vehicles (BEVs)']) and pd.isna(row['Plug-In Hybrid Electric Vehicles (PHEVs)']):
                continue
                
            bev_count = row['Battery Electric Vehicles (BEVs)'] if not pd.isna(row['Battery Electric Vehicles (BEVs)']) else 0
            phev_count = row['Plug-In Hybrid Electric Vehicles (PHEVs)'] if not pd.isna(row['Plug-In Hybrid Electric Vehicles (PHEVs)']) else 0
            
            # Create records for BEVs if they exist
            if bev_count > 0:
                data_converted.append({
                    'Year': row['Year'],
                    'Region': row['State'],
                    'Vehicle_Type': 'BEV',
                    'EV_Count': bev_count,
                    'Population': 1000000,  # Default value - could be enhanced with real population data
                    'Charging_Stations': max(1, int(bev_count * 0.1)),
                    'GDP_Per_Capita': 50000  # Default value - could be enhanced with real GDP data
                })
            
            # Create records for PHEVs if they exist
            if phev_count > 0:
                data_converted.append({
                    'Year': row['Year'],
                    'Region': row['State'],
                    'Vehicle_Type': 'PHEV',
                    'EV_Count': phev_count,
                    'Population': 1000000,  # Default value
                    'Charging_Stations': max(1, int(phev_count * 0.1)),
                    'GDP_Per_Capita': 50000  # Default value
                })
        
        if len(data_converted) == 0:
            print("No valid data found in CSV. Using sample data instead.")
            return self.create_sample_data()
        
        converted_df = pd.DataFrame(data_converted)
        print(f"Converted data shape: {converted_df.shape}")
        print(f"Year range: {converted_df['Year'].min()} - {converted_df['Year'].max()}")
        print(f"States: {converted_df['Region'].nunique()}")
        return converted_df
    
    def create_sample_data(self):
        """Create sample EV data for demonstration"""
        np.random.seed(42)
        years = list(range(2010, 2024))
        regions = ['North', 'South', 'East', 'West', 'Central']
        vehicle_types = ['BEV', 'PHEV', 'HEV']
        
        data = []
        base_adoption = 1000
        
        for year in years:
            growth_factor = 1.2 ** (year - 2010)
            for region in regions:
                region_factor = np.random.uniform(0.8, 1.2)
                for vehicle_type in vehicle_types:
                    type_factor = {'BEV': 1.5, 'PHEV': 1.2, 'HEV': 0.8}[vehicle_type]
                    
                    ev_count = int(base_adoption * growth_factor * region_factor * type_factor * 
                                 np.random.uniform(0.9, 1.1))
                    
                    data.append({
                        'Year': year,
                        'Region': region,
                        'Vehicle_Type': vehicle_type,
                        'EV_Count': ev_count,
                        'Population': np.random.randint(500000, 2000000),
                        'Charging_Stations': int(ev_count * 0.1 * np.random.uniform(0.8, 1.2)),
                        'GDP_Per_Capita': np.random.randint(30000, 80000)
                    })
        
        self.data = pd.DataFrame(data)
        print(f"Sample data created. Shape: {self.data.shape}")
        return self.data
    
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        self.data = self.data.fillna(self.data.mean(numeric_only=True))
        
        categorical_cols = ['Region', 'Vehicle_Type']
        for col in categorical_cols:
            if col in self.data.columns:
                le = LabelEncoder()
                self.data[f'{col}_encoded'] = le.fit_transform(self.data[col])
                self.label_encoders[col] = le
        
        return self.data
    
    def get_features_target(self, target_col='EV_Count'):
        """Prepare features and target for modeling"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        feature_cols = ['Year', 'Region_encoded', 'Vehicle_Type_encoded', 
                       'Population', 'Charging_Stations', 'GDP_Per_Capita']
        
        feature_cols = [col for col in feature_cols if col in self.data.columns]
        
        X = self.data[feature_cols]
        y = self.data[target_col]
        
        return X, y
    
    def visualize_data(self):
        """Create visualizations of the EV data"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # EV adoption over time
        yearly_data = self.data.groupby('Year')['EV_Count'].sum()
        axes[0, 0].plot(yearly_data.index, yearly_data.values, marker='o')
        axes[0, 0].set_title('Total EV Adoption Over Time')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Total EV Count')
        
        # EV adoption by region
        if 'Region' in self.data.columns:
            region_data = self.data.groupby('Region')['EV_Count'].sum()
            axes[0, 1].bar(region_data.index, region_data.values)
            axes[0, 1].set_title('EV Adoption by Region')
            axes[0, 1].set_xlabel('Region')
            axes[0, 1].set_ylabel('Total EV Count')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # EV adoption by vehicle type
        if 'Vehicle_Type' in self.data.columns:
            type_data = self.data.groupby('Vehicle_Type')['EV_Count'].sum()
            axes[1, 0].pie(type_data.values, labels=type_data.index, autopct='%1.1f%%')
            axes[1, 0].set_title('EV Adoption by Vehicle Type')
        
        # Correlation heatmap
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig('e:/Electrical_vehical_prediction/eda_plots.png', dpi=300, bbox_inches='tight')
        plt.show()