import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load datasets
real_time_df = pd.read_excel("/content/real_time_data.xlsx", engine='openpyxl')
synthetic_df = pd.read_csv("/content/synthetic_data.csv")

# Combine datasets
combined_df = pd.concat([real_time_df, synthetic_df], ignore_index=True)

# Handle missing values
numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
categorical_cols = combined_df.select_dtypes(include=['object']).columns

# Fill missing values
combined_df[numeric_cols] = combined_df[numeric_cols].fillna(combined_df[numeric_cols].mean())
for col in categorical_cols:
    combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])

# Convert labels and handle invalid entries
combined_df['Condition'] = combined_df['Condition'].map({'Healthy': 0, 'Diabetes': 1})

# Check for remaining NaN and fill with mode
if combined_df['Condition'].isna().any():
    mode_val = combined_df['Condition'].mode()[0]
    combined_df['Condition'] = combined_df['Condition'].fillna(mode_val)

# Final conversion to integer
combined_df['Condition'] = combined_df['Condition'].astype(int)

# Split features and labels
X = combined_df.iloc[:, :-1].values.astype(float)
y = combined_df.iloc[:, -1].values.astype(int)

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Enhanced MLP Class
class MLP:
    def _init_(self, input_size, hidden_size, output_size, lr=0.01, reg_lambda=0.01, dropout_rate=0.3):
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.dropout_rate = dropout_rate
        
        # Initialize weights
        self.w1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.b1 = np.random.uniform(-1, 1, (1, hidden_size))
        self.w2 = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.b2 = np.random.uniform(-1, 1, (1, output_size))
        
        # Training history
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x, training=True):
        self.hidden = self.relu(np.dot(x, self.w1) + self.b1)
        
        # Apply dropout only during training
        if training and self.dropout_rate > 0:
            self.mask = (np.random.rand(*self.hidden.shape) > self.dropout_rate).astype(float)
            self.hidden *= self.mask / (1 - self.dropout_rate)  # Scale for inference
            
        self.output = self.sigmoid(np.dot(self.hidden, self.w2) + self.b2)
        return self.output

    def backward(self, x, y_true):
        # Calculate gradients
        output_error = y_true - self.output
        output_delta = output_error * (self.output * (1 - self.output))
        
        hidden_error = np.dot(output_delta, self.w2.T)
        hidden_delta = hidden_error * (self.hidden > 0).astype(float)
        
        # Apply dropout mask
        if self.dropout_rate > 0:
            hidden_delta *= self.mask / (1 - self.dropout_rate)
        
        # Regularization terms
        d_w2 = np.dot(self.hidden.T, output_delta) - self.reg_lambda * self.w2
        d_b2 = np.sum(output_delta, axis=0, keepdims=True)
        d_w1 = np.dot(x.T, hidden_delta) - self.reg_lambda * self.w1
        d_b1 = np.sum(hidden_delta, axis=0, keepdims=True)
        
        # Update parameters
        self.w2 += self.lr * d_w2
        self.b2 += self.lr * d_b2
        self.w1 += self.lr * d_w1
        self.b1 += self.lr * d_b1

    def train(self, X_train, y_train, X_val, y_val, epochs=100, patience=10, decay_rate=0.95):
        best_val_loss = float('inf')
        wait = 0
        
        for epoch in range(epochs):
            # Learning rate decay
            if epoch % 10 == 0 and epoch != 0:
                self.lr *= decay_rate
            
            # Training loop
            loss = 0
            for x, y in zip(X_train, y_train):
                x = x.reshape(1, -1)
                y = np.array([[y]])
                self.forward(x)
                self.backward(x, y)
                loss += (y - self.output) ** 2
            
            # Calculate metrics
            train_loss = np.mean(loss) + 0.5*self.reg_lambda*(np.sum(self.w1*2) + np.sum(self.w2*2))
            train_acc = self._evaluate(X_train, y_train)
            val_loss = self._calculate_loss(X_val, y_val)
            val_acc = self._evaluate(X_val, y_val)
            
            # Store history
            self.history['loss'].append(train_loss)
            self.history['accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Progress monitoring
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

    def _calculate_loss(self, X, y):
        loss = 0
        for x, y_true in zip(X, y):
            x = x.reshape(1, -1)
            self.forward(x, training=False)  # Disable dropout for evaluation
            loss += (y_true - self.output[0][0]) ** 2
        return np.mean(loss) + 0.5*self.reg_lambda*(np.sum(self.w1*2) + np.sum(self.w2*2))

    def _evaluate(self, X, y):
        return np.mean(self.predict(X) == y)

    def predict(self, X, threshold=0.5):
        return np.array([1 if self.forward(x.reshape(1,-1), training=False)[0][0] >= threshold else 0 for x in X])

# Initialize and train model
mlp = MLP(
    input_size=X_train.shape[1],
    hidden_size=32,
    output_size=1,
    lr=0.01,
    reg_lambda=0.01,  # Stronger regularization
    dropout_rate=0.3   # Increased dropout
)

mlp.train(X_train, y_train, X_test, y_test, epochs=200, patience=15)

# Final evaluation
y_pred = mlp.predict(X_test)
print(f"\nFinal Test Accuracy: {np.mean(y_pred == y_test)*100:.2f}%")
