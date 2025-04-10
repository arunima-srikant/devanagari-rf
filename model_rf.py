import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
import time

# Load preprocessed data
with open('processed_data.pkl', 'rb') as f:
    x_train, y_train, x_val, y_val = pickle.load(f)

# Split training data for cross-validation
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
start_time = time.time()
# Initialize Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=200,         # Fewer trees
    max_depth=20,            # Limit tree depth
    random_state=42
)

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, Y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}%")

# Train the model
model.fit(X_train, Y_train)
print('model trained.')
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print(start_time)
"""
#to test a single image:
test_img = "ML/project/edited_im/37-gray1.png"

# Load the image
img = Image.open(test_img)
            
# Convert the image to a numpy array
imgnump = asarray(img)
imgnump = imgnump.reshape(-1)  # Flatten the image


# Evaluate on test data
Y_pred = model.predict(x_val)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")

# Evaluate on validation data
val_accuracy = accuracy_score(y_val, model.predict(x_val))
print(f"Validation Set Accuracy: {val_accuracy * 100:.2f}%")

# Plot confusion matrix
cm = confusion_matrix(Y_test, Y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='viridis', xticks_rotation='vertical')
plt.show()
"""