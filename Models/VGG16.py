import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from PIL import Image

# Define race mappings based on UTKFace
race_mapping = {
    0: 'White',
    1: 'Black',
    2: 'Asian',
    3: 'Indian',
    4: 'Others'
}

# Set the path to the UTKFace dataset folder
dataset_path = r"C:\Users\Gabriel\Documents\Inge-2I\UTK_Dataset"  # Update with your dataset path

# Create a DataFrame for dataset information
data = []
for file in os.listdir(dataset_path):
    if file.endswith(".jpg"):
        parts = file.split("_")
        if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
            age = int(parts[0])
            gender = int(parts[1])  # 0 for male, 1 for female
            race = int(parts[2])
            image_path = os.path.join(dataset_path, file)
            data.append((image_path, age, gender, race))
        else:
            print(f"Skipping file with unexpected format: {file}")

# Create a DataFrame from the valid data
df = pd.DataFrame(data, columns=["image_path", "age", "gender", "race"])

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Data generator class
class UTKFaceGenerator(Sequence):
    def __init__(self, dataframe, batch_size=32, target_size=(224, 224), shuffle=True):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.on_epoch_end()  # Shuffle at the start

    def __len__(self):
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        batch_df = self.dataframe.iloc[index * self.batch_size:(index + 1) * self.batch_size]
        images = []
        genders = []
        ages = []
        races = []

        for _, row in batch_df.iterrows():
            img = Image.open(row["image_path"])
            img = img.resize(self.target_size)
            img_array = np.array(img) / 255.0  # Normalize
            if img_array.shape == (224, 224, 3):
                images.append(img_array)
                genders.append(row["gender"])
                ages.append(row["age"])
                races.append(row["race"])
            else:
                print(f"Skipping image with unexpected shape: {row['image_path']}")

        images = np.array(images)
        genders = np.array(genders)
        ages = np.array(ages)
        races = np.array(races)

        return images, [genders, ages, races]

    def on_epoch_end(self):
        if self.shuffle:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

# Create data generators for training and testing sets
batch_size = 32
train_generator = UTKFaceGenerator(train_df, batch_size=batch_size)
test_generator = UTKFaceGenerator(test_df, batch_size=batch_size)

# Create the model as before
base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = Flatten()(base_model.output)

gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)
age_output = Dense(1, activation='linear', name='age_output')(x)
race_output = Dense(5, activation='softmax', name='race_output')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=[gender_output, age_output, race_output])

# Compile the model with appropriate loss functions and metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss={
        'gender_output': 'binary_crossentropy',
        'age_output': 'mse',
        'race_output': 'categorical_crossentropy'
    },
    metrics={
        'gender_output': ['accuracy', tf.keras.metrics.Precision()],
        'age_output': ['mae'],
        'race_output': ['accuracy']
    }
)

# Train the model using the data generators
model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,
    verbose=1
)

# Evaluate the model using the test generator
results = model.evaluate(test_generator)

print(f"Test Gender Accuracy: {results[1] * 100:.2f}%")
print(f"Test Age MAE: {results[2]:.2f}")
print(f"Test Race Accuracy: {results[3] * 100:.2f}%")
