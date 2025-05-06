import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# Audio processing
import librosa
import librosa.display

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle, class_weight

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.optimizers import Adam
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load and label dataset
dataset = []
for folder in ["./set_a/**/*.wav", "./set_b/**/*.wav"]:
    for filename in glob.iglob(folder, recursive=True):
        if os.path.isfile(filename):
            # get the parent folder as label source
            label_folder = os.path.basename(os.path.dirname(filename)).lower()

            # skip unwanted labels
            if any(x in label_folder for x in ["unlabelled", "artifact"]):
                continue

            # classify label
            if "normal" in label_folder:
                label = "normal"
            else:
                label = "abnormal"

            try:
                duration = librosa.get_duration(filename=filename)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            if duration >= 3:
                slice_size = 3
                iterations = int((duration - slice_size) / (slice_size - 1)) + 1
                initial_offset = (duration - ((iterations * (slice_size - 1)) + 1)) / 2
                for i in range(iterations):
                    offset = initial_offset + i * (slice_size - 1)
                    dataset.append(
                        {
                            "filename": filename,
                            "label": label,
                            "offset": offset,
                        }
                    )

# convert to DataFrame
dataset = pd.DataFrame(dataset)
dataset = shuffle(dataset, random_state=42)
dataset.info()

# Plot dataset distribution
plt.figure(figsize=(4, 6))
dataset.label.value_counts().plot(kind="bar", title="Dataset distribution")
plt.show()

# Split into train and test (80-20)
train, test = train_test_split(dataset, test_size=0.2, random_state=42)
print(f"Train: {len(train)}")
print(f"Test: {len(test)}")

# Visualize waveplot, spectrogram and MFCC
plt.figure(figsize=(20, 10))
idx = 0
for label in dataset.label.unique():
    y, sr = librosa.load(dataset[dataset.label == label].filename.iloc[33], duration=3)
    idx += 1
    plt.subplot(2, 3, idx)
    plt.title(f"{label} waveplot")
    librosa.display.waveshow(y=y, sr=sr)

    idx += 1
    plt.subplot(2, 3, idx)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
    )
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, hop_length=512, x_axis="time", y_axis="mel")
    plt.title(f"{label} mel spectrogram")

    idx += 1
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
    plt.subplot(2, 3, idx)
    librosa.display.specshow(mfccs, x_axis="time")
    plt.title(f"{label} MFCC (Mel Spectrogram)")
plt.show()


# Extract features
def extract_features(audio_path, offset):
    y, sr = librosa.load(audio_path, offset=offset, duration=3)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
    )
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
    return mfccs


x_train = [
    extract_features(train.filename.iloc[i], train.offset.iloc[i])
    for i in tqdm(range(len(train)))
]
x_test = [
    extract_features(test.filename.iloc[i], test.offset.iloc[i])
    for i in tqdm(range(len(test)))
]

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

# Encode labels
encoder = LabelEncoder()
encoder.fit(train.label)
y_train = encoder.transform(train.label)
y_test = encoder.transform(test.label)
class_weights = class_weight.compute_class_weight(
    "balanced", np.unique(y_train), y_train
)

# Reshape and one-hot encode labels
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build CNN model
model = Sequential()
model.add(
    Conv2D(
        16,
        kernel_size=2,
        activation="relu",
        input_shape=(x_train.shape[1], x_train.shape[2], 1),
    )
)
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(32, kernel_size=2, activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=2, activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=2, activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))

model.add(GlobalAveragePooling2D())
model.add(Dense(len(encoder.classes_), activation="softmax"))
model.summary()

# Compile model
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

# Train model
history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=300,
    validation_data=(x_test, y_test),
    class_weight=class_weights,
    shuffle=True,
)

# Plot loss and accuracy curves
plt.figure(figsize=[14, 6])
plt.subplot(211)
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()

plt.subplot(212)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Curves")
plt.legend()
plt.tight_layout()
plt.show()

# Evaluate model
scores = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])

# Predict and show classification report
predictions = model.predict(x_test, verbose=1)
y_true = [encoder.classes_[np.argmax(y)] for y in y_test]
y_pred = [encoder.classes_[np.argmax(pred)] for pred in predictions]
print(classification_report(y_true, y_pred))

# Save model
model.save("heartbeat_classifier.keras")

from keras.models import load_model
import numpy as np

# Load model
model = load_model("heartbeat_classifier.keras")

# File to be classified
classify_file = "my_heartbeat.wav"

# Extract features and reshape
x_test = [extract_features(classify_file, 0.5)]
x_test = np.asarray(x_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# Predict
pred = model.predict(x_test, verbose=1)

# Get predicted class
pred_class = np.argmax(pred, axis=1)

# Output
if pred_class[0] == 1:
    print("Normal heartbeat")
    print("Confidence:", pred[0][1])
else:
    print("Abnormal heartbeat")
    print("Confidence:", pred[0][0])
