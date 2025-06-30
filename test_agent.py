import joblib
import cv2
import numpy as np

model = joblib.load(r"C:\Users\tareq\OneDrive\Desktop\AIProject2\models\mlp_model.pkl")

def preprocess_image(image_path, size=(64, 64), bins=(8, 8, 8)):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    image = cv2.resize(image, size)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    features = hist.flatten()
    return features.reshape(1, -1)

def test_agent(image_path):
    try:
        features = preprocess_image(image_path)
    except Exception as e:
        print(f"{e}")
        return
    prediction = model.predict(features)[0]
    print("Prediction:", prediction)

test_image = r"C:\Users\tareq\OneDrive\Desktop\Grosser_Panda.jpg"
test_agent(test_image)
