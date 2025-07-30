import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf

# Function to load image paths and their labels
def load_path(path):
    """
    Load X-ray dataset from the given folder structure.
    """
    dataset = []
    for folder in os.listdir(path):  # E.g., Elbow, Hand, Shoulder
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            for patient in os.listdir(folder_path):  # E.g., patient11186
                patient_path = os.path.join(folder_path, patient)
                for study in os.listdir(patient_path):  # E.g., study1_positive
                    if study.endswith('positive'):
                        label = 'fractured'
                    elif study.endswith('negative'):
                        label = 'normal'
                    else:
                        continue
                    study_path = os.path.join(patient_path, study)
                    for img in os.listdir(study_path):  # E.g., image1.png
                        img_path = os.path.join(study_path, img)
                        if os.path.isfile(img_path):
                            dataset.append(
                                {
                                    'label': folder,  # Elbow, Hand, Shoulder
                                    'image_path': img_path
                                }
                            )
    return dataset


# Load data from the specified directory
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(THIS_FOLDER, 'Dataset/test')  # Use test folder
data = load_path(image_dir)

# Extract filepaths and labels
labels = []
filepaths = []

Labels = ["Elbow", "Hand", "Shoulder"]  # Map folder names as class labels
for row in data:
    labels.append(row['label'])
    filepaths.append(row['image_path'])

filepaths = pd.Series(filepaths, name='Filepath')
labels = pd.Series(labels, name='Label')

images = pd.concat([filepaths, labels], axis=1)

# Test Image Data Generator
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

test_images = test_generator.flow_from_dataframe(
    dataframe=images,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# Load the pre-trained model
model_path = os.path.join(THIS_FOLDER, "weights/ResNet50_BodyPartsCopy.h5")
model = tf.keras.models.load_model(model_path)

# Evaluate the model on the test dataset
# Load the pre-trained model
model_path = os.path.join(THIS_FOLDER, "weights/ResNet50_BodyPartsCopy.h5")
model = tf.keras.models.load_model(model_path)

# Ensure the model is compiled correctly
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Evaluate the model on the test dataset
results = model.evaluate(test_images, verbose=1)
print(f"Test Accuracy: {np.round(results[1] * 100, 2)}%")

#
# # Generate predictions (optional, for analysis)
# predictions = model.predict(test_images)
# predicted_classes = np.argmax(predictions, axis=1)
# true_classes = test_images.classes
# class_labels = list(test_images.class_indices.keys())
#
# # Print classification report (optional, for detailed results)
# from sklearn.metrics import classification_report
# report = classification_report(true_classes, predicted_classes, target_names=class_labels)
# print(report)
