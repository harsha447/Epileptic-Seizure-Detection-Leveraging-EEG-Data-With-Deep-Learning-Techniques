import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# Load the dataset
file_path = 'Epileptic Seizure Recognition.csv'
df = pd.read_csv(file_path)

# Drop the identifier column
df = df.drop(columns=['Unnamed'])

# Split data into features and labels
X = df.drop(columns=['y'])
y = df['y']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
clf = SVC()
clf.fit(X_train, y_train)

# Save the trained model to a file
model_filename = 'svm_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(clf, file)