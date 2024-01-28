import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from numpy import where
import matplotlib.pyplot as plt

# Load the dataset (replace this with the actual path to your CSV file)
df = pd.read_csv('path_to_your_gaze_data.csv', names=["X", "Y"])

# Define thresholds for normal gaze area (modify these values based on your requirement)
x_min, x_max, y_min, y_max = -6, 1505, 27, 839

# Process the gaze data
status = -1
elapsed_time = []
c = 0

for index, row in df.iterrows():
    if row['X'] > x_max or row['X'] < x_min or row['Y'] > y_max or row['Y'] < y_min:
        if status == 0:
            elapsed_time.append(c)
            c = 0
        status = 1
        c += 1
    else:
        if status == 1:
            elapsed_time.append(c)
            c = 0
        status = 0
        c += 1

# Prepare data for OneClassSVM
df_new = pd.DataFrame({'status': [status] * len(elapsed_time), 'elapsed_time': elapsed_time})
df_new = df_new.astype({"status": int, "elapsed_time": int})

# OneClassSVM for anomaly detection
svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.1)
svm.fit(df_new)
pred = svm.predict(df_new)

# Identifying anomalies (potential cheating behavior)
anom_index = where(pred == -1)
values = df_new.loc[anom_index]
df_new['svm_pred'] = pred

# Result interpretation
result = 'non-cheat'
for index, row in df_new.iterrows():
    if row['svm_pred'] == -1 and row['status'] == 0:
        result = 'cheat'
        break

# Output result
print("Analysis result:", result)

# Optional: Plotting for visualization
plt.scatter(df['X'], df['Y'], c=df_new['svm_pred'])
plt.title("Gaze Data with SVM Prediction")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
