# 📦 Imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os

# 📁 Dataset Path Check
train_dir = r'C:\Programming\dataset\train'
test_dir = r'C:\Programming\dataset\test'

if not os.path.exists(train_dir):
    raise FileNotFoundError("Training folder not found. Please check the path.")
if not os.path.exists(test_dir):
    raise FileNotFoundError("Test folder not found. Please check the path.")

# 🔄 Data Generators
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    test_dir,
    target_size=(100, 100),
    class_mode='categorical'
)

# Model Definition
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: apple, banana, orange
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 📊 Train Model
history = model.fit(train_data, epochs=10, validation_data=val_data)

# 📈 Plot and Save Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("accuracy_plot.png")
plt.show()

# 🔍 Predict Single Image
# Update this path to any image you want to test
img_path = r"C:\Programming\dataset\test\Apple Golden 1\73_100.jpg"                 # Apple

if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")

img = load_img(img_path, target_size=(100, 100))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
class_names = ['apple', 'banana', 'orange']
predicted_class = class_names[np.argmax(prediction)]

print("Predicted class:", predicted_class)
