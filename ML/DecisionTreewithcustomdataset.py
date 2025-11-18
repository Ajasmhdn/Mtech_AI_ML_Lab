# Step 1: Import Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 2: Define the Play Tennis Dataset and save to CSV
data = pd.DataFrame([
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
], columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])

# Save dataset as CSV
data.to_csv("/content/drive/MyDrive/Mtech/AILab/play_tennis.csv", index=False)
print("✅ Dataset saved successfully at: /content/drive/MyDrive/Mtech/AILab/play_tennis.csv")

# Step 3: Load Dataset
df = pd.read_csv("/content/drive/MyDrive/Mtech/AILab/play_tennis.csv")
print("\nDataset Preview:\n", df.head())

# Step 4: Encode Categorical Columns
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

print("\nEncoded Dataset:\n", df)


# Step 5: Split Data into Features and Target
X = df.drop(columns=['PlayTennis'])
y = df['PlayTennis']

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Step 6: Train Decision Tree (ID3 uses 'entropy')
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test)

print("\n✅ Predicted Values:", y_pred)
print("🎯 Actual Values:", list(y_test.values))

# Step 8: Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Visualize the Decision Tree
plt.figure(figsize=(10,6))
plot_tree(model,
          feature_names=X.columns,
          class_names=['No', 'Yes'],
          filled=True,
          rounded=True)
plt.title("Decision Tree using ID3 Algorithm (Entropy)")
plt.show()
