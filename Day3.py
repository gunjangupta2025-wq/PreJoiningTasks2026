import numpy as np

def k_fold_split(X, y, k=5, shuffle=True, random_state=42):
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]
        
    folds_X = np.array_split(X, k)
    folds_y = np.array_split(y, k)
    
    splits = []
    for i in range(k):
        # Use fold i for validation
        X_val, y_val = folds_X[i], folds_y[i]
        
        # Use the remaining folds for training
        X_train = np.vstack([folds_X[j] for j in range(k) if j != i])
        y_train = np.concatenate([folds_y[j] for j in range(k) if j != i])
        
        splits.append((X_train, y_train, X_val, y_val))
    return splits

def plot_learning_curve(model, X_train, y_train, X_val, y_val, metric_fn):
    train_errors, val_errors = [], []
    m_samples = len(X_train)
    
    # Incrementally increase training size
    for i in range(10, m_samples, max(1, m_samples // 20)):
        model.fit(X_train[:i], y_train[:i])
        
        train_preds = model.predict(X_train[:i])
        val_preds = model.predict(X_val)
        
        train_errors.append(metric_fn(y_train[:i], train_preds))
        val_errors.append(metric_fn(y_val, val_preds))
        
    # Plot train_errors vs val_errors over the sample sizes

class LinearRegressionNormal:
    def fit(self, X, y):
        # Add bias column
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Normal Equation
        self.w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.w)

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.w = np.zeros(X_b.shape[1])
        m = len(y)
        
        for _ in range(self.iterations):
            z = np.dot(X_b, self.w)
            predictions = self._sigmoid(z)
            # Gradient of Log Loss
            gradient = np.dot(X_b.T, (predictions - y)) / m
            self.w -= self.lr * gradient
            
    def predict_prob(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return self._sigmoid(np.dot(X_b, self.w))

    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)