import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Path to the directory containing JSON files
directory_path = '/data/{user_name}/NeuSE/dataset/combine_record/'

# List to store all records
data_records = []

# Load all JSON files from the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".json"):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r') as file:
            data = json.load(file)
            data_records.append(data)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data_records)

# Define features and target
X = df.drop('decision', axis=1)  # Features
y = df['decision']               # Target

# Get unique filenames
unique_filenames = X['IR_filename'].unique()

# Split the filenames into train and test
train_files, test_files = train_test_split(unique_filenames, test_size=0.2, random_state=42)

# Filter both DataFrames based on the split
X_train = X[X['IR_filename'].isin(train_files)]
X_test = X[X['IR_filename'].isin(test_files)]
y_train = y.loc[X_train.index]
y_test = y.loc[X_test.index]

X_train = X_train.drop(['IR_filename', 'Merge_point_addr'], axis=1)
X_test = X_test.drop(['IR_filename', 'Merge_point_addr'], axis=1)

# Step 1: Identify the number of instances in the minority class
class_counts = y_test.value_counts()
minority_class = class_counts.idxmin()
minority_count = class_counts.min()

# Step 2: Downsample the majority classes
sampled_dfs = []
for cls in y_test.unique():
    cls_subset = y_test[y_test == cls]
    # Downsample each class to the size of the minority class
    sampled_dfs.append(cls_subset.sample(n=minority_count, random_state=42))

# Concatenate the downsampled DataFrames
downsampled_y_test = pd.concat(sampled_dfs)

# Step 3: Apply the same indices to X_test
downsampled_X_test = X_test.loc[downsampled_y_test.index]

# Initialize and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(downsampled_X_test)

# Save the model to a file
model_filename = '/data/{user_name}/NeuSE/test_model/combine_random_forest_model_0424.joblib'
dump(clf, model_filename)

# Evaluate the model
accuracy = accuracy_score(downsampled_y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(downsampled_y_test, y_pred))

# Predict probabilities for the test set
y_scores = clf.predict_proba(downsampled_X_test)[:, 1]  # Get the probability for the positive class

# Compute ROC AUC values
fpr, tpr, thresholds = roc_curve(downsampled_y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Compute Precision-Recall values
precision, recall, _ = precision_recall_curve(downsampled_y_test, y_scores)
pr_auc = average_precision_score(downsampled_y_test, y_scores)

# Plotting ROC Curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Plotting Precision-Recall Curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.show()
plt.savefig("/data/{user_name}/NeuSE/test_model/combine_random_forest_model_0424_eval_curve.png")

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

forest_importances = pd.Series(importances, index=X_train.columns)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()
plt.savefig("/data/{user_name}/NeuSE/test_model/combine_random_forest_model_0424_fea_imp.png")