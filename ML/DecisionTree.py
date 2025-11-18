from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load iris dataset using df.
df = load_iris()
X = df.data
y = df.target

# # Check data (optional)
# print("Features:\n", X[:5])
# print("Target:\n", y[:5])
# print("Target Names:", df.target_names)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Create Decision Tree model (ID3 uses entropy)
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

# 4. Predict & evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=df.target_names))

# 5. Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=df.target_names)
disp.plot(cmap='Blues')
plt.title("Decision Tree - Confusion Matrix")
plt.show()

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plot_tree(model, feature_names=df.feature_names, class_names=list(df.target_names), filled=True)
plt.show()


# 7. Single sample prediction
print("Predicted:", df.target_names[model.predict([X_test[0]])[0]])
