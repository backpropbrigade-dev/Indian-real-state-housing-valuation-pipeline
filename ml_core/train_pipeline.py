import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_production_pipeline(data_path, model_output_path):
    print("🚀 Initializing ValuReal AI Model Training Pipeline...")
    start_time = time.time()
    
    # 1. Load the dataset safely
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
        
    df = pd.read_csv(data_path)
    print(f"📊 Dataset loaded successfully. Shape: {df.shape[0]} records, {df.shape[1]} features.")
    
    # 2. Define feature sets based on schema analysis
    categorical_cols = [
        'State', 'City', 'Property_Type', 'Furnished_Status', 
        'Public_Transport_Accessibility', 'Parking_Space', 'Security', 
        'Facing', 'Owner_Type', 'Availability_Status'
    ]
    numerical_cols = [
        'BHK', 'Size_in_SqFt', 'Year_Built', 'Floor_No', 'Total_Floors', 
        'Age_of_Property', 'Nearby_Schools', 'Nearby_Hospitals'
    ]
    target_col = 'Price_in_Lakhs'
    
    X = df[categorical_cols + numerical_cols]
    y = df[target_col]
    
    # 3. Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🛠️ Constructing ColumnTransformer Preprocessing Pipeline...")
    # 4. Preprocessing: Numeric scaling + Robust Categorical Ordinal Encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
        ]
    )
    
    # 5. Integrated Production Machine Learning Pipeline
    # HistGradientBoostingRegressor is optimized for large scale data and native category indices
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(
            max_iter=150, 
            learning_rate=0.08, 
            max_depth=8, 
            random_state=42
        ))
    ])
    
    # 6. Fit Pipeline
    print("🏋️ Training Enterprise Ensemble Core (HistGradientBoostingRegressor)...")
    pipeline.fit(X_train, y_train)
    
    # 7. Model Evaluation Metrics
    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print("\n==================================================")
    print("🎯 PRODUCTION MODEL TRAINING EVALUATION METRICS")
    print("==================================================")
    print(f"🔹 Mean Absolute Error (MAE) : ₹ {mae:.2f} Lakhs")
    print(f"🔹 Root Mean Squared Error (RMSE): ₹ {rmse:.2f} Lakhs")
    print(f"🔹 Coefficient of Determination ($R^2$): {r2:.4f}")
    print("==================================================\n")
    
    # 8. Serialize and Save Pipeline Artifact
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    
    elapsed_time = time.time() - start_time
    print(f"💾 Production pipeline saved successfully to: {model_output_path}")
    print(f"⏱️ Total Execution Pipeline Pipeline Time: {elapsed_time:.2f} seconds\n")

if __name__ == '__main__':
    # Configuration paths adjusted relative to project root
    DATA_PATH = os.path.join(os.path.dirname(__file__), '../india_housing_prices.csv')
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'valuation_model.joblib')
    train_production_pipeline(DATA_PATH, MODEL_PATH)
