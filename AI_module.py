import os
import csv
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier

# Path to save CSV and models
output_csv = r"C:\Users\tareq\OneDrive\Desktop\AIProject2\labels.csv"
model_dir = r"C:\Users\tareq\OneDrive\Desktop\AIProject2\models"
os.makedirs(model_dir, exist_ok=True)

folders = [
    r"C:\Users\tareq\OneDrive\Desktop\AIProject2\animal_data\animal_data\Bird",
    r"C:\Users\tareq\OneDrive\Desktop\AIProject2\animal_data\animal_data\Cat",
    r"C:\Users\tareq\OneDrive\Desktop\AIProject2\animal_data\animal_data\Cow",
    r"C:\Users\tareq\OneDrive\Desktop\AIProject2\animal_data\animal_data\Dog",
    r"C:\Users\tareq\OneDrive\Desktop\AIProject2\animal_data\animal_data\Lion",
    r"C:\Users\tareq\OneDrive\Desktop\AIProject2\animal_data\animal_data\Panda"
]

def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def rename_images(folder_path):
    label = os.path.basename(folder_path)
    files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    for i, filename in enumerate(files, start=1):
        ext = os.path.splitext(filename)[1]
        new_name = f"{label}{i}{ext}"
        dst = os.path.join(folder_path, new_name)
        src = os.path.join(folder_path, filename)
        if os.path.abspath(dst) != os.path.abspath(src):
            if os.path.exists(dst):
                print(f"Skipping: {new_name} already exists.")
                continue
            os.rename(src, dst)
            print(f"Renamed: {filename} → {new_name}")

def run_renaming_all_folders():
    for folder in folders:
        print(f"\nRenaming files in: {folder}")
        rename_images(folder)

def read_file():
    image_labels = []
    for folder in folders:
        label = os.path.basename(folder)
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            if os.path.isfile(file_path):
                image_labels.append([file_path, label])
    with open(output_csv, mode="w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'label'])
        writer.writerows(image_labels)
    return output_csv

def train_module(csv_path):
    x, y = [], []
    bins = (8, 8, 8)
    with open(csv_path, mode="r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            image_path = row[0]
            label = row[1]
            if not os.path.exists(image_path):
                continue
            image = cv2.imread(image_path)
            if image is None:
                continue
            image = cv2.resize(image, (64, 64))
            features = extract_color_histogram(image, bins)
            x.append(features)
            y.append(label)
    x = np.array(x)
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, test_size=0.2, random_state=42)

    nb_model = GaussianNB()
    nb_model.fit(x_train, y_train)
    joblib.dump(nb_model, os.path.join(model_dir, "naive_bayes_model.pkl"))

    dt_model = DecisionTreeClassifier(random_state=15, criterion="entropy")
    dt_model.fit(x_train, y_train)
    joblib.dump(dt_model, os.path.join(model_dir, "decision_tree_model.pkl"))

    mlp_model = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, alpha=0.001, random_state=47, activation="relu")
    mlp_model.fit(x_train, y_train)
    joblib.dump(mlp_model, os.path.join(model_dir, "mlp_model.pkl"))

    models = {
        "Naive Bayes": nb_model,
        "Decision Tree": dt_model,
        "MLP Classifier": mlp_model
    }

    return models, x_test, y_test, dt_model, y_train

def evaluate_accuracy_and_metrics(models, x_test, y_test, class_names):
    for name, model in models.items():
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=class_names)
        print(f"\n{name} Accuracy: {acc:.4f}")
        print(f"{name} Confusion Matrix:\n{cm}")
        print(f"{name} Precision/Recall/F1-score:\n{classification_report(y_test, y_pred, labels=class_names, digits=3)}")

def visualize_decision_tree(dt_model, class_names):
    plt.figure(figsize=(18, 8))
    plot_tree(
        dt_model,
        filled=True,
        feature_names=[f"bin{i}" for i in range(dt_model.n_features_in_)],
        class_names=class_names,
        rounded=True,
        fontsize=7,
        max_depth=3 # This is going to plot 3 levels of the tree.
    )
    plt.title("Decision Tree Visualization (First 3 Levels)")
    plt.tight_layout()
    plt.show()

def main():
    run_renaming_all_folders()
    csv_path = read_file()
    models, x_test, y_test, dt_model, y_train = train_module(csv_path)
    class_names = sorted(list(set(y_train) | set(y_test)))
    evaluate_accuracy_and_metrics(models, x_test, y_test, class_names)
    visualize_decision_tree(dt_model, class_names)

main()
