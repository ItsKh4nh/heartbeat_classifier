
# Heartbeat Sound Classification

## Project Overview
This project implements a deep learning system for classifying heartbeat sounds into different categories: Normal, Murmur, Extrahls (extra heart sounds), Artifact, and Extrasystole. The classifier uses audio processing techniques to extract features from heartbeat recordings and employs a Convolutional Neural Network (CNN) to categorize them accurately.

The model can help identify abnormal heart sounds that might indicate various heart conditions, providing a potential tool for preliminary cardiac assessment.

## Project Structure
```
heartbeat_classifier/
├── data/               # Dataset directory
│   ├── set_a/          
│   ├── set_b/          
├── models/             # Saved model files
├── main.py             # Main script for training and inference
└── requirements.txt    # Project dependencies
```

## Technologies Used
- **Python 3.x**
- **Audio Processing**: librosa
- **Data Manipulation**: NumPy, Pandas
- **Machine Learning**: TensorFlow/Keras, Scikit-learn
- **Visualization**: Matplotlib
- **Utilities**: tqdm

## Dataset
The model is trained on the [Heartbeat Sounds Dataset](https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds) from Kaggle, which contains recordings of various heart sounds including:
- Normal heart sounds
- Murmurs
- Extra heart sounds (extrahls)
- Artifacts
- Extrasystoles

## Results
The model achieves excellent performance (up to 90% Accuracy) across different heartbeat sound categories:

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| artifact    | 0.98      | 1.00   | 0.99     | 40      |
| extrahls    | 0.65      | 1.00   | 0.79     | 13      |
| extrasystole| 0.70      | 0.79   | 0.75     | 24      |
| murmur      | 0.88      | 0.92   | 0.90     | 99      |
| normal      | 0.95      | 0.88   | 0.91     | 217     |
|             |           |        |          |         |
| accuracy    |           |        | 0.90     | 393     |
| macro avg   | 0.83      | 0.92   | 0.87     | 393     |
| weighted avg| 0.91      | 0.90   | 0.90     | 393     |

## Installation

```bash
# Clone the repository
git clone https://github.com/ItsKh4nh/heartbeat_classifier
cd heartbeat_classifier

# Create a virtual environment (Optional but recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model
To train the model from scratch:

```bash
python main.py
```
or
```bash
jupyter notebook main.ipynb
```

This will:
1. Load and preprocess audio data from the dataset
2. Extract MFCC features from the audio files
3. Train a CNN model on the extracted features
4. Save the trained model to the `models/` directory
5. Display performance metrics and visualizations

### Classifying New Heartbeat Recordings
To classify a new heartbeat recording, modify the `file_to_classify` variable:

```python
# File to classify
file_to_classify = "your_heartbeat_recording.wav"
```