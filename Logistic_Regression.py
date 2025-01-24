import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

df = pd.read_csv("D:/Data Science/real_life_logistic_regression.csv")

X = df[['Age']].values
Y = df['Insurance_Bought'].values  

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=5)

lr = LogisticRegression(random_state=42).fit(x_train, y_train)

y_hat = lr.predict(x_test)
y_prob = lr.predict_proba(X)
decision_boundary = -lr.intercept_ / lr.coef_
mse = mean_squared_error(y_test, y_hat)
print("Mean squared error : ",mse)

sorted_indices = np.argsort(X.flatten())
X_sorted = X[sorted_indices]
y_prob_sorted = y_prob[sorted_indices]

plt.scatter(X, Y, color="blue", alpha=0.5, label="Data points")
plt.plot(X_sorted, y_prob_sorted[:, 1], color="red", label="Insurance bought probability")
plt.plot(X_sorted, y_prob_sorted[:, 0], color="black", label="Insurance not bought probability")
plt.axvline(decision_boundary, color="green", linestyle="--", label="Decision Boundary")
plt.xlabel("Age")
plt.ylabel("Insurance Bought Probability")
plt.grid(True)
plt.legend()
plt.show()


