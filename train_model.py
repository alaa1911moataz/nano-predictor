import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# 1. Load data and drop NaNs from the target variables only
df = pd.read_csv('DrugDelivery_Silver.csv')
df = df.dropna(subset=['Tumor_%ID', 'Selectivity_Index'])

# 2. Define features and targets (keeping both continuous and categorical Zeta columns)
features = ['NP_Class', 'Shape', 'Size (nm)', 'Zeta Potential (mv)', 'Zeta_Category',
            'HAS_PEG', 'Shell Type', 'Administration Dosages (mg/kg)',
            'Time point (h)', 'Tumor Site']
targets = ['Tumor_%ID', 'Selectivity_Index']

X = df[features]
y = df[targets]

# 3. Specify feature types
numeric_features = ['Size (nm)', 'Zeta Potential (mv)', 'Administration Dosages (mg/kg)', 'Time point (h)']
categorical_features = ['NP_Class', 'Shape', 'Zeta_Category', 'HAS_PEG', 'Shell Type', 'Tumor Site']

# 4. Data preprocessing pipeline
# OrdinalEncoder preserves NaNs for categorical variables using handle_unknown='use_encoded_value'
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features), # Numeric features (including NaNs) pass directly to XGBoost
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan), categorical_features) # Text converted to numbers while preserving NaNs
    ])

# 5. Build the pipeline with XGBoost (inherently handles missing values)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(
        XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            missing=np.nan,          # Explicitly instructing XGBoost to handle np.nan values natively
            random_state=42,
            n_jobs=-1
        )
    ))
])

# 6. Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Model training
print("Training XGBoost model (handling missing lab values natively)... ")
model.fit(X_train, y_train)
print("Training completed successfully! Missing value patterns discovered automatically.\n")

# 8. Prediction and evaluation
predictions = model.predict(X_test)

for i, target_name in enumerate(targets):
    y_true = y_test.iloc[:, i]
    y_pred = predictions[:, i]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"Evaluation Metrics for [{target_name}]:")
    print(f"   - R² Score (Coefficient of Determination): {r2:.4f}")
    print(f"   - Mean Absolute Error (MAE): {mae:.4f}")
    print(f"   - Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("-" * 40)

import joblib
joblib.dump(model, 'xgboost_nano_model.pkl')