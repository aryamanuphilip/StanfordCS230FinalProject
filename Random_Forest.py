import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Load dataset
data = pd.read_csv('Sanction_List.csv')

# Check for missing values
data.isnull().sum()

# Drop rows with missing target (or you can impute missing target values)
data.dropna(subset=['is_sanctioned'], inplace=True)

# Handling missing values for features using SimpleImputer (e.g., replacing with the median)
imputer = SimpleImputer(strategy='median')
data[['sanction_entity', 'sanction_type', 'sanction_country', 'address', 'city', 'state', 'province', 
       'vessel_id', 'additional_information', 'sanction_start_date', 'sanction_expiry_date', 'date_of_birth']] = \
    imputer.fit_transform(data[['sanction_entity', 'sanction_type', 'sanction_country', 'address', 'city', 'state', 
                                'province', 'vessel_id', 'additional_information', 'sanction_start_date', 
                                'sanction_expiry_date', 'date_of_birth']])

# Encoding categorical features using LabelEncoder
encoder_entity = LabelEncoder()
encoder_type = LabelEncoder()
encoder_country = LabelEncoder()
encoder_city = LabelEncoder()
encoder_state = LabelEncoder()
encoder_province = LabelEncoder()
encoder_vessel = LabelEncoder()

data['sanction_entity'] = encoder_entity.fit_transform(data['sanction_entity'])
data['sanction_type'] = encoder_type.fit_transform(data['sanction_type'])
data['sanction_country'] = encoder_country.fit_transform(data['sanction_country'])
data['city'] = encoder_city.fit_transform(data['city'])
data['state'] = encoder_state.fit_transform(data['state'])
data['province'] = encoder_province.fit_transform(data['province'])
data['vessel_id'] = encoder_vessel.fit_transform(data['vessel_id'])

# Handle date features (sanction start and expiry dates) by converting to numerical values (e.g., days difference)
data['sanction_start_date'] = pd.to_datetime(data['sanction_start_date'], errors='coerce')
data['sanction_expiry_date'] = pd.to_datetime(data['sanction_expiry_date'], errors='coerce')

data['sanction_duration'] = (data['sanction_expiry_date'] - data['sanction_start_date']).dt.days

# Fill any missing sanction_duration values (e.g., NaT) with a default value (e.g., 0)
data['sanction_duration'].fillna(0, inplace=True)

# Additional feature engineering: Age from date_of_birth
data['date_of_birth'] = pd.to_datetime(data['date_of_birth'], errors='coerce')
data['age'] = (pd.to_datetime('today') - data['date_of_birth']).dt.days / 365

# Drop any rows with missing values after transformation
data.dropna(inplace=True)

# Features and Target
X = data[['sanction_entity', 'sanction_type', 'sanction_country', 'city', 'state', 'province', 'vessel_id', 
          'sanction_duration', 'age']]
y = data['is_sanctioned']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = rf_model.predict(X_test)

# Print classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importances
feature_importances = pd.DataFrame(rf_model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
print("Feature Importances:\n", feature_importances)

# Save the trained model
import joblib
joblib.dump(rf_model, 'sanction_predictor_rf_model.pkl')
