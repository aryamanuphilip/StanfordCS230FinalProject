import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


# Load dataset
data = pd.read_csv('Sanction_List.csv')

# Check for missing values
data.isnull().sum()

# Drop rows with missing values (or you can use imputation strategies if necessary)
data.dropna(inplace=True)

# Encoding categorical features
encoder_entity = LabelEncoder()
encoder_type = LabelEncoder()
encoder_country = LabelEncoder()

data['sanction_entity'] = encoder_entity.fit_transform(data['sanction_entity'])
data['sanction_type'] = encoder_type.fit_transform(data['sanction_type'])
data['sanction_country'] = encoder_country.fit_transform(data['sanction_country'])

# Features and Target
X = data[['sanction_entity', 'sanction_type', 'sanction_country']]
y = data['is_sanctioned']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Input layer, shape corresponds to number of features
    Dense(128, activation='relu'),     # Hidden layer with ReLU activation
    Dense(64, activation='relu'),      # Another hidden layer
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
model.save('sanction_predictor_model.h5')


