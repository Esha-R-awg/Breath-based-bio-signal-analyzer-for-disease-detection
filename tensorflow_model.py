import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import joblib

# === Load Dataset ===
df = pd.read_csv("C:/Users/dhars/OneDrive/Breath_based_project/breath_data.csv")
X = df[["MQ3", "MQ135", "Temperature", "Humidity"]]
y = df["Disease"]

# === Label Encoding ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# === Feature Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, stratify=y_categorical, random_state=42
)

# === Model Architecture (with Dropout to reduce overfitting) ===
model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

# === Compile the Model ===
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Early Stopping Callback ===
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# === Train the Model ===
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=150,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)

# === Save Model and Preprocessors ===
model.save("C:/Users/dhars/OneDrive/Breath_based_project/keras_mlp_model.h5")
joblib.dump(scaler, "C:/Users/dhars/OneDrive/Breath_based_project/scaler_final_1.pkl")
joblib.dump(label_encoder, "C:/Users/dhars/OneDrive/Breath_based_project/label_encoder_final_1.pkl")

# === Final Accuracy ===
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

final_train_acc = train_acc[-1]
final_val_acc = val_acc[-1]

print(f"\nFinal Training Accuracy: {final_train_acc * 100:.2f}%")
print(f"Final Validation Accuracy: {final_val_acc * 100:.2f}%")

# === Plot Accuracy and Loss Curves ===
plt.figure(figsize=(14, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
