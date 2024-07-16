import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import os
import cv2
from tqdm import tqdm  # Import tqdm for progress bar

# Paths and classes definitions
data_dir = 'C:\\class work\\python\\datasets\\Cats_and_dogs_data\\training_set'
classes = {'cats': 0, 'dogs': 1}

# Load images
X = []
Y = []

for cls in classes:
    pth = os.path.join(data_dir, cls)
    for j in os.listdir(pth):
        img = cv2.imread(os.path.join(pth, j))
        img = cv2.resize(img, (200, 200))
        img = img / 255.0
        X.append(img.flatten())
        Y.append(classes[cls])

X = np.array(X)
Y = np.array(Y)

# Display the first image
X_reshaped = X[0].reshape((200, 200, 3))
plt.imshow(X_reshaped)
plt.show()

# Reshape X
X_updated = X.reshape(len(X), -1)
print(X_updated.shape)

# Splitting data
xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10, test_size=0.05)

print("Train split: ", xtrain.shape)
print("Test split: ", xtest.shape)
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())

# Model training with progress bar
sv = SVC()

# Fit the model with progress bar
num_epochs = 10  # Number of iterations or epochs
batch_size = 100  # Adjust batch size based on your system's memory and performance
num_batches = num_epochs * (len(xtrain) // batch_size)

with tqdm(total=num_batches) as pbar:
    for epoch in range(num_epochs):
        for batch_start in range(0, len(xtrain), batch_size):
            batch_end = min(batch_start + batch_size, len(xtrain))
            sv.fit(xtrain[batch_start:batch_end], ytrain[batch_start:batch_end])
            pbar.update(1)

print("Training Score:", sv.score(xtrain, ytrain))
print("Testing Score:", sv.score(xtest, ytest))
pred = sv.predict(xtest)
misclassified = np.where(ytest != pred)
print("Misclassified indices:", misclassified)
print("Total Misclassified Samples: ", len(misclassified[0]))

# Plot 20 misclassified images
num_misclassified_to_plot = 20
plt.figure(figsize=(20, 10))

for i, index in enumerate(misclassified[0][:num_misclassified_to_plot]):
    misclassified_image = xtest[index].reshape((200, 200, 3))
    plt.subplot(4, 5, i + 1)
    plt.imshow(misclassified_image)
    plt.title(f'Pred: {pred[index]}, Actual: {ytest[index]}')
    plt.axis('off')

plt.tight_layout()
plt.show()
