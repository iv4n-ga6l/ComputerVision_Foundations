"""
Build a simple linear regression model from scratch using basic calculus.
Then use this model to predict future revenue for a company based on historical data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _initialize_parameters(self, n_features):
        """Initialize weights and bias to zeros"""
        self.weights = np.zeros((n_features, 1))  
        self.bias = 0
    
    def _compute_cost(self, X, y, m):
        """
        Compute Mean Squared Error cost
        
        Parameters:
        X: input features (m x n)
        y: target values (m x 1)
        m: number of training examples
        
        Returns:
        cost: mean squared error
        """
        predictions = np.dot(X, self.weights) + self.bias
        cost = (1/(2*m)) * np.sum((predictions - y) ** 2)
        return cost
    
    def _compute_gradients(self, X, y, m):
        """
        Compute gradients for weights and bias
        
        Parameters:
        X: input features (m x n)
        y: target values (m x 1)
        m: number of training examples
        
        Returns:
        dw: gradient for weights
        db: gradient for bias
        """
        predictions = np.dot(X, self.weights) + self.bias
        
        # Reshape X to match dimensions if needed
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        # Partial derivatives - ensure correct shapes
        dw = (1/m) * np.dot(X.T, (predictions - y))
        db = (1/m) * np.sum(predictions - y)
        
        return dw, db
    
    def fit(self, X, y):
        """
        Train the linear regression model using gradient descent
        
        Parameters:
        X: training features (m x n)
        y: target values (m x 1)
        """
        # Ensure X and y are numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X if needed
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        # Reshape y if needed
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Number of training examples and features
        m, n = X.shape
        
        # Initialize parameters
        self._initialize_parameters(n)
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Compute cost and gradients
            cost = self._compute_cost(X, y, m)
            dw, db = self._compute_gradients(X, y, m)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Store cost history
            self.cost_history.append(cost)
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Parameters:
        X: input features
        
        Returns:
        predictions: predicted values
        """
        # Ensure X is a numpy array and reshape if needed
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        return np.dot(X, self.weights) + self.bias


# Create sample company data
def create_company_data():
    # Create monthly revenue data for the past 5 years
    years = 5
    months = years * 12
    
    # Starting date
    start_date = datetime(2019, 1, 1)
    dates = [start_date + timedelta(days=30*i) for i in range(months)]
    
    # Create revenue data with trend, seasonality, and some randomness
    base_revenue = 1000000  # Starting revenue
    trend = np.linspace(0, 500000, months)  # Upward trend
    seasonality = 100000 * np.sin(np.linspace(0, 2*np.pi*years, months))  # Seasonal pattern
    noise = np.random.normal(0, 50000, months)  # Random variations
    
    revenue = base_revenue + trend + seasonality + noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'Date': dates,
        'Revenue': revenue
    })
    
    return data


if __name__ == "__main__":
    # Get company data
    company_data = create_company_data()
    
    # Prepare features (use month number as feature)
    X = np.arange(len(company_data))
    y = company_data['Revenue'].values
    
    # Create and train model
    model = LinearRegression(learning_rate=0.0001, n_iterations=2000)
    model.fit(X, y)
    
    # Make predictions for next 12 months
    future_months = np.arange(len(X), len(X) + 12)
    future_predictions = model.predict(future_months)
    
    # Print results
    print("\nRevenue Predictions for Next 12 Months:")
    print("-" * 50)
    for i, pred in enumerate(future_predictions):
        future_date = company_data['Date'].iloc[-1] + timedelta(days=30*(i+1))
        print(f"{future_date.strftime('%B %Y')}: ${pred[0]:,.2f}")
    
    print("\nModel Performance Metrics:")
    print("-" * 50)
    predictions = model.predict(X)
    mse = np.mean((predictions - y.reshape(-1, 1)) ** 2)
    rmse = np.sqrt(mse)
    print(f"Root Mean Square Error: ${rmse:,.2f}")
    
    # Plotting
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 6))
        
        # Plot historical data
        plt.plot(company_data['Date'], y, color='blue', label='Historical Revenue', marker='o')
        
        # Plot predictions for existing data
        plt.plot(company_data['Date'], predictions, color='green', label='Model Fit', linestyle='--')
        
        # Plot future predictions
        future_dates = [company_data['Date'].iloc[-1] + timedelta(days=30*i) for i in range(1, 13)]
        plt.plot(future_dates, future_predictions, color='red', label='Future Predictions', linestyle='--')
        
        plt.title('Company Revenue Trend and Predictions')
        plt.xlabel('Date')
        plt.ylabel('Revenue ($)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Plot cost history
        plt.figure(figsize=(10, 4))
        plt.plot(model.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Training Cost History')
        plt.grid(True)
        plt.tight_layout()
        
        plt.show()
    except ImportError:
        print("\nMatplotlib not installed. Skipping plots.")