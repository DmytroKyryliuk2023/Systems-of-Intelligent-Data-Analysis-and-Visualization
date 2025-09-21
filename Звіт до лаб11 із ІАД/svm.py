import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# === 1. Завантаження даних ===
file_path = './iris.csv'  # заміни на свій файл
df = pd.read_csv(file_path)

# === 2. Розділення ознак і цільової змінної ===
X = df.iloc[:, :-1]                # усі стовпці, крім останнього — ознаки
y_raw = df.iloc[:, -1]             # останній стовпець — текстові класи

# === 3. Кодування класів у числа ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# === 4. Розбиття на train/test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === 5. Класифікація SVM з poly kernel ===
clf = SVC(kernel='poly', degree=3, C=1.0)  # можна змінити degree або C для експериментів
clf.fit(X_train, y_train)

# === 6. Прогнозування і оцінка ===
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
