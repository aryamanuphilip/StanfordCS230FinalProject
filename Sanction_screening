import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv('Sanction_List.csv')

# Check for missing values
data.isnull().sum()

# Drop rows with missing values (or you can use imputation strategies if necessary)
data.dropna(inplace=True)

# Encoding categorical features for the new features
encoder_entity = LabelEncoder()
encoder_type = LabelEncoder()
encoder_country = LabelEncoder()
encoder_designation = LabelEncoder()
encoder_vessel_identifier = LabelEncoder()

# Assuming the new columns are available, encode categorical features
data['sanction_entity'] = encoder_entity.fit_transform(data['sanction_entity'])
data['sanction_type'] = encoder_type.fit_transform(data['sanction_type'])
data['sanction_country'] = encoder_country.fit_transform(data['sanction_country'])
data['sanction_individual_designation'] = encoder_designation.fit_transform(data['sanction_individual_designation'])
data['sanction_vessel_identifier'] = encoder_vessel_identifier.fit_transform(data['sanction_vessel_identifier'])

# Convert dates to numeric (e.g., Unix timestamps)
data['sanction_individual_dateOfBirth'] = pd.to_datetime(data['sanction_individual_dateOfBirth'], errors='coerce')
data['sanction_individual_dateOfBirth'] = data['sanction_individual_dateOfBirth'].fillna(pd.to_datetime('1970-01-01')).astype(np.int64) // 10**9

# Optional: Normalize/scale features like dateOfBirth for better model performance
scaler = MinMaxScaler()
data['sanction_individual_dateOfBirth'] = scaler.fit_transform(data[['sanction_individual_dateOfBirth']])

# Handle text fields (e.g., 'sanction_individual_address' and 'sanction_vessel_description')
from sklearn.feature_extraction.text import TfidfVectorizer

# Combine text fields into one and apply TF-IDF vectorizer
data['combined_text'] = data['sanction_individual_address'].fillna('') + " " + data['sanction_vessel_description'].fillna('')
vectorizer = TfidfVectorizer(max_features=100)  # Limit to 100 features for simplicity
text_features = vectorizer.fit_transform(data['combined_text']).toarray()

# Concatenate the text features back into the dataframe
text_df = pd.DataFrame(text_features, columns=[f'text_feature_{i}' for i in range(text_features.shape[1])])
data = pd.concat([data, text_df], axis=1)

# Drop original text columns
data.drop(columns=['sanction_individual_address', 'sanction_vessel_description', 'combined_text'], inplace=True)

# Features and Target
X = data[['sanction_entity', 'sanction_type', 'sanction_country', 'sanction_individual_designation', 
          'sanction_vessel_identifier', 'sanction_individual_dateOfBirth', 'sanction_vessel_id', 
          'sanction_vessel_id2'] + [f'text_feature_{i}' for i in range(text_features.shape[1])]]

y = data['is_sanctioned']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Input layer, shape corresponds to number of features
    Dense(128, activation='relu'),     # Hidden layer with ReLU activation
    Dropout(0.2),                     # Dropout to prevent overfitting
    Dense(64, activation='relu'),      # Another hidden layer
    Dense(32, activation='relu'),      # Another layer to help with feature learning
    Dense(1, activation='sigmoid')     # Output layer, sigmoid for binary classification
])

# Summary of the model to check its architecture
model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=20, 
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
model.save('sanction_predictor_model_extended.h5')
