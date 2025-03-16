import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


real_time_df = pd.read_excel("/content/real_time_data.xlsx", engine='openpyxl')
synthetic_df = pd.read_csv("/content/synthetic_data.csv")

combined_df = pd.concat([real_time_df, synthetic_df], ignore_index=True)

numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
categorical_cols = combined_df.select_dtypes(include=['object']).columns
combined_df[numeric_cols] = combined_df[numeric_cols].fillna(combined_df[numeric_cols].mean())
for col in categorical_cols:
    combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])

combined_df['Condition'] = combined_df['Condition'].map({'Healthy': 0, 'Diabetes': 1})
if combined_df['Condition'].isna().any():
    combined_df['Condition'] = combined_df['Condition'].fillna(combined_df['Condition'].mode()[0])
combined_df['Condition'] = combined_df['Condition'].astype(int)

X = combined_df.iloc[:, :-1].values.astype(float)
y = combined_df.iloc[:, -1].values.astype(int)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def train_and_evaluate(lr, reg_lambda, dropout_rate, hidden_size):
    class MLP:
        def _init_(self, input_size, hidden_size, output_size, lr=0.01, reg_lambda=0.01, dropout_rate=0.3):
            self.lr = lr
            self.reg_lambda = reg_lambda
            self.dropout_rate = dropout_rate
            self.w1 = np.random.uniform(-1, 1, (input_size, hidden_size))
            self.b1 = np.random.uniform(-1, 1, (1, hidden_size))
            self.w2 = np.random.uniform(-1, 1, (hidden_size, output_size))
            self.b2 = np.random.uniform(-1, 1, (1, output_size))

        def relu(self, x):
            return np.maximum(0, x)

        def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))

        def forward(self, x, training=True):
            self.hidden = self.relu(np.dot(x, self.w1) + self.b1)
            if training and self.dropout_rate > 0:
                self.mask = (np.random.rand(*self.hidden.shape) > self.dropout_rate).astype(float)
                self.hidden *= self.mask / (1 - self.dropout_rate)
            self.output = self.sigmoid(np.dot(self.hidden, self.w2) + self.b2)
            return self.output

        def backward(self, x, y_true):
            output_error = y_true - self.output
            output_delta = output_error * (self.output * (1 - self.output))
            hidden_error = np.dot(output_delta, self.w2.T)
            hidden_delta = hidden_error * (self.hidden > 0).astype(float)
            if self.dropout_rate > 0:
                hidden_delta *= self.mask / (1 - self.dropout_rate)

            self.w2 += self.lr * (np.dot(self.hidden.T, output_delta) - self.reg_lambda * self.w2)
            self.b2 += self.lr * np.sum(output_delta, axis=0, keepdims=True)
            self.w1 += self.lr * (np.dot(x.T, hidden_delta) - self.reg_lambda * self.w1)
            self.b1 += self.lr * np.sum(hidden_delta, axis=0, keepdims=True)

        def train(self, X_train, y_train, epochs=100):
            losses = []
            for _ in range(epochs):
                epoch_loss = 0
                for x, y in zip(X_train, y_train):
                    self.forward(x.reshape(1, -1))
                    self.backward(x.reshape(1, -1), np.array([[y]]))
                    epoch_loss += np.mean((self.output - y) ** 2)
                losses.append(epoch_loss / len(X_train))
            return losses

        def predict_proba(self, X):
            return np.array([self.forward(x.reshape(1, -1), training=False)[0][0] for x in X])

        def predict(self, X):
            return (self.predict_proba(X) >= 0.5).astype(int)

    mlp = MLP(X_train.shape[1], hidden_size, 1, lr, reg_lambda, dropout_rate)
    losses = mlp.train(X_train, y_train)
    y_pred = mlp.predict(X_test)
    y_scores = mlp.predict_proba(X_test)

    
    acc = np.mean(y_pred == y_test)
    print(f"Accuracy with lr={lr}, reg_lambda={reg_lambda}, dropout_rate={dropout_rate}, hidden_size={hidden_size}: {acc * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Diabetes'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()


    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

 
    plt.plot(losses)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

params = [
    (0.005, 0.01, 0.3, 32),
    (0.005, 0.01, 0.3, 64),
    (0.001, 0.01, 0.2, 64),
    (0.001, 0.1, 0.3, 128)
]

for lr, reg_lambda, dropout_rate, hidden_size in params:
    train_and_evaluate(lr, reg_lambda, dropout_rate, hidden_size)
