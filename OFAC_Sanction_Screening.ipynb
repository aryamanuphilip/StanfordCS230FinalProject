{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9e41c63f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Embedding, Flatten, Input\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a48164ea",
   "metadata": {},
   "source": [
    "CSV file contains columns: sanction_entity, sanction_type, sanction_country as input feaures and and is_sanctioned is the output or prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "682bfaa8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load dataset\n",
    "data = pd.read_csv('sanctions_data.csv')\n",
    "\n",
    "# Check for missing values\n",
    "data.isnull().sum()\n",
    "\n",
    "# Drop rows with missing values (or you can use imputation strategies if necessary)\n",
    "data.dropna(inplace=True)\n",
    "\n",
    "# Encoding categorical features\n",
    "encoder_entity = LabelEncoder()\n",
    "encoder_type = LabelEncoder()\n",
    "encoder_country = LabelEncoder()\n",
    "\n",
    "data['sanction_entity'] = encoder_entity.fit_transform(data['sanction_entity'])\n",
    "data['sanction_type'] = encoder_type.fit_transform(data['sanction_type'])\n",
    "data['sanction_country'] = encoder_country.fit_transform(data['sanction_country'])\n",
    "\n",
    "# Features and Target\n",
    "X = data[['sanction_entity', 'sanction_type', 'sanction_country']]\n",
    "y = data['is_sanctioned']\n",
    "\n",
    "# Split the dataset into training and test sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "44d4c191",
   "metadata": {},
   "source": [
    "Model Architecture"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bb2a7833",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define the model architecture\n",
    "model = Sequential([\n",
    "    Input(shape=(X_train.shape[1],)),  # Input layer, shape corresponds to number of features\n",
    "    Dense(128, activation='relu'),     # Hidden layer with ReLU activation\n",
    "    Dense(64, activation='relu'),      # Another hidden layer\n",
    "    Dense(1, activation='sigmoid')     # Output layer, sigmoid for binary classification\n",
    "])\n",
    "\n",
    "# Summary of the model to check its architecture\n",
    "model.summary()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "963b5124",
   "metadata": {},
   "source": [
    "compiling the model "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "74b27c2b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compile the model\n",
    "model.compile(optimizer=Adam(learning_rate=0.001),\n",
    "              loss='binary_crossentropy',\n",
    "              metrics=['accuracy'])\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "197cd168",
   "metadata": {},
   "source": [
    "Train the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "63e8c36e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train the model\n",
    "history = model.fit(X_train, y_train, \n",
    "                    epochs=20, \n",
    "                    batch_size=32, \n",
    "                    validation_data=(X_test, y_test))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "72dcce1a",
   "metadata": {},
   "source": [
    "Evaluating the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f6494580",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluate the model on the test set\n",
    "loss, accuracy = model.evaluate(X_test, y_test)\n",
    "print(f'Test Loss: {loss}')\n",
    "print(f'Test Accuracy: {accuracy}')\n",
    "\n",
    "# Predicting on the test set\n",
    "y_pred = (model.predict(X_test) > 0.5).astype(\"int32\")\n",
    "\n",
    "# Print classification report and confusion matrix\n",
    "print(classification_report(y_test, y_pred))\n",
    "print(confusion_matrix(y_test, y_pred))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "68b09d6f",
   "metadata": {},
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
