import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
import time
import csv

# Load preprocessed data
with open('processed_data.pkl', 'rb') as f:
    x_train, y_train, x_val, y_val = pickle.load(f)
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

pred = model.predict(x_val)
accuracy = accuracy_score(y_val, pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
counter = 0
for i in range(0, len(y_val)):
    if pred[i]!= y_val[i]:
        counter +=1
print(f'incorrectly classified: {counter}')

"""
img = Image.fromarray(x_val[124].reshape(64,64,4))
img = Image.fromarray(x_val[160].reshape(64,64,4))
img.show()

out_file = open('predictions.txt', 'w', encoding='utf-8')
writer=csv.writer(out_file,lineterminator='\n')
for p in pred:
    writer.writerow(p)
out_file.close()

# Plot confusion matrix
cm = confusion_matrix(y_val, pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='viridis', xticks_rotation='vertical')
plt.show()
"""