# students-lifestyle-project
using decision tree and naive bayes model 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('student_lifestyle_dataset.csv')
data.head()

data.isnull().sum()

data.shape

data.duplicated().sum()

label_encoder = LabelEncoder()
data['Stress_Level'] = label_encoder.fit_transform(data['Stress_Level'])
data.head()

correlation=data.corr()
correlation

sns.heatmap(correlation,annot=True,cmap='pink')

X = data.drop(columns=['Student_ID', 'Stress_Level'])
y = data['Stress_Level']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=label_encoder.classes_, filled=True, fontsize=10)
plt.title("Decision Tree Visualization")
plt.show()

from sklearn.naive_bayes import GaussianNB


gnb = GaussianNB()
gnb.fit(X_train, y_train)


y_pred_gnb = gnb.predict(X_test)


print("Gaussian Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_gnb))
print(f"Accuracy: {accuracy_score(y_test, y_pred_gnb)}")
