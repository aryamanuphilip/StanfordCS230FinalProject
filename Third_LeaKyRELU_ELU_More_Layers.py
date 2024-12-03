import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from datetime import datetime

# Load dataset
data = pd.read_csv('Sanction_List.csv')

# Check for missing values
print("Missing values before cleaning:")
print(data.isnull().sum())

# Drop rows with missing target values or impute if needed (you can customize this)
data.dropna(subset=['is_sanctioned'], inplace=True)

# Impute missing categorical values with a placeholder
categorical_columns = ['sanction_entity', 'sanction_type', 'sanction_country', 'address', 'city', 'state', 'province', 'vessel_id', 'additional_information']
imputer_cat = SimpleImputer(strategy='most_frequent')
data[categorical_columns] = imputer_cat.fit_transform(data[categorical_columns])

# Impute missing continuous values (e.g., if any numerical features are added in the future)
# continuous_columns = ['some_continuous_column']
# imputer_cont = SimpleImputer(strategy='mean')
# data[continuous_columns] = imputer_cont.fit_transform(data[continuous_columns])

# Encoding categorical features
encoder_entity = LabelEncoder()
encoder_type = LabelEncoder()
encoder_country = LabelEncoder()
encoder_address = LabelEncoder()
encoder_city = LabelEncoder()
encoder_state = LabelEncoder()
encoder_province = LabelEncoder()
encoder_vessel_id = LabelEncoder()
encoder_additional_info = LabelEncoder()

data['sanction_entity'] = encoder_entity.fit_transform(data['sanction_entity'])
data['sanction_type'] = encoder_type.fit_transform(data['sanction_type'])
data['sanction_country'] = encoder_country.fit_transform(data['sanction_country'])
data['address'] = encoder_address.fit_transform(data['address'])
data['city'] = encoder_city.fit_transform(data['city'])
data['state'] = encoder_state.fit_transform(data['state'])
data['province'] = encoder_province.fit_transform(data['province'])
data['vessel_id'] = encoder_vessel_id.fit_transform(data['vessel_id'])
data['additional_information'] = encoder_additional_info.fit_transform(data['additional_information'])

# Handling date_of_birth: Convert to datetime and extract features (age, year, month, day)
data['date_of_birth'] = pd.to_datetime(data['date_of_birth'], errors='coerce')  # Convert to datetime, handle errors
data['age'] = data['date_of_birth'].apply(lambda x: (datetime.now() - x).days // 365 if pd.notnull(x) else 0)

# Handling sanction start and expiry dates
data['sanction_start_date'] = pd.to_datetime(data['sanction_start_date'], errors='coerce')
data['sanction_expiry_date'] = pd.to_datetime(data['sanction_expiry_date'], errors='coerce')

# Calculate the duration of the sanction (in days)
data['sanction_duration'] = (data['sanction_expiry_date'] - data['sanction_start_date']).dt.days

# Extract temporal features from the sanction dates
data['sanction_start_year'] = data['sanction_start_date'].dt.year
data['sanction_start_month'] = data['sanction_start_date'].dt.month
data['sanction_start_day'] = data['sanction_start_date'].dt.day

data['sanction_expiry_year'] = data['sanction_expiry_date'].dt.year
data['sanction_expiry_month'] = data['sanction_expiry_date'].dt.month
data['sanction_expiry_day'] = data['sanction_expiry_date'].dt.day

# Drop the original date_of_birth, sanction_start_date, and sanction_expiry_date columns as we no longer need them
data.drop(columns=['date_of_birth', 'sanction_start_date', 'sanction_expiry_date'], inplace=True)

# Features and Target
X = data[['sanction_entity', 'sanction_type', 'sanction_country', 'address', 'city', 'state', 'province', 'age', 
          'vessel_id', 'additional_information', 'sanction_duration', 'sanction_start_year', 'sanction_start_month', 
          'sanction_start_day', 'sanction_expiry_year', 'sanction_expiry_month', 'sanction_expiry_day']]
y = data['is_sanctioned']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# New DNN Architecture with LeakyReLU and ELU Activation Functions
inputs = layers.Input(shape=(X_train.shape[1],))  # Input layer

# First hidden layer with LeakyReLU activation
x = layers.Dense(256)(inputs)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.BatchNormalization()(x)  # Batch normalization
x = layers.Dropout(0.4)(x)  # Dropout for regularization

# Second hidden layer with ELU activation
x = layers.Dense(128)(x)
x = layers.ELU(alpha=1.0)(x)
x = layers.BatchNormalization()(x)  # Batch normalization
x = layers.Dropout(0.4)(x)  # Dropout for regularization

# Third hidden layer with LeakyReLU activation
x = layers.Dense(64)(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.BatchNormalization()(x)  # Batch normalization
x = layers.Dropout(0.4)(x)  # Dropout for regularization

# Output layer
outputs = layers.Dense(1, activation='sigmoid')(x)

# Build the model
model = models.Model(inputs=inputs, outputs=outputs)

# Summary of the model to check its architecture
model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=30,  # Increased number of epochs
                    batch_size=32, 
                    validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# Predicting on the test set
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Print classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save the trained model
model.save('sanction_predictor_dnn_with_leakyrelu_elu.h5')
