

import os          # traversing directories
import json        # saving data in json format
import math        # allows to perform ceiling
import librosa     # extracting audio features
import numpy as np # shaping multidimensional data
from tqdm.notebook import tqdm # progress bars


class AudioFeatureExtractor():


    def __init__(self, duration, sample_rate=22050, num_segments=5, samples_per_label=-1):
        """Initializes the Audio Feature Extractor.

        Note:
            Keeping `samples_per_label` initialized to `-1` will result in processing every
            available track in the dataset path.
        """
        # Initialize the passed parameters
        self.duration = duration
        self.sample_rate = sample_rate
        self.num_segments = num_segments
        self.samples_per_label = samples_per_label

        # Calculate dependent attributes
        self.samples_per_track = duration * sample_rate
        self.samples_per_segment = self.samples_per_track // num_segments

        # Define default constants
        self._DEFAULT_CONSTANTS = {
            "n_mfcc": 13,
            "n_fft": 2048,
            "hop_length": 512,
        }


    def _validate_requests(self, requests):

        mapping = {
            "mfcc": self._extract_mfcc,
        }

        # Define requests
        valid_requests = {}
        requests = [(req, {}) if isinstance(req, str) else req for req in requests]

        if len(requests) == 0:
            # If no request is specified, raise an error
            raise AttributeError("At least 1 feature extraction request must be specified")

        # Loop through every request
        for request_name, params in requests:
            if request_name not in mapping.keys():
                # If request name is not valid, raise an error
                raise AttributeError(f"'{request_name}' is not a valid feature extraction request")

            # Define items for `valid_requests`
            func = mapping[request_name]
            args = dict(self._DEFAULT_CONSTANTS)
            args.update(params)

            # Add valid request to dictionary
            valid_requests[func] = args

        return valid_requests



    def _extract_mfcc(self, signal, params):
  
        mfcc = librosa.feature.mfcc(y = signal,
                                    sr=self.sample_rate,
                                    n_mfcc=params["n_mfcc"],
                                    n_fft=params["n_fft"],
                                    hop_length=params["hop_length"])

        # The expected number of generated mfcc vectors (required to ensure consistency)
        expected_n_vectors = math.ceil(self.samples_per_segment / params["hop_length"])

        if mfcc.shape[1] != expected_n_vectors:
            # Repeat the last col if it doesn't exist (may occur occasionally)
            mfcc = np.resize(mfcc, (params["n_mfcc"], expected_n_vectors))

        return mfcc.T.tolist() # more comforatable representation and parsable by JSON


    def _segmented_extraction(self, filepath, extractions):
        """Performs feature extraction for every track segment.

        Args:
            filepath (str): The path to audio file
            extractions (dict): The dictionary which maps extraction function to its parameters

        Returns:
            feature_segments (list): The list of preprocessed track segments with features
        """
        # Load the sound file and initialize segment list
        signal, sr = librosa.load(filepath, sr=self.sample_rate)
        feature_segments = []

        # Loop through the specified number of segments
        for segment in range(self.num_segments):
            # Take a segment from signal to preprocess it
            start_sample = self.samples_per_segment * segment
            end_sample = start_sample + self.samples_per_segment
            signal_segment = signal[start_sample:end_sample]
            features_group = []

            # Loop through every extraction request
            for func, params in extractions.items():
                # Generate features for the given segment and append to the group
                features_group.append(func(signal_segment, params))

            # Append the group of features to the segment
            feature_segments.append([features_group])

        return feature_segments


    def _walk_extract(self, dataset_path, extractions):

        data = {"semantic_labels": [], "targets": [], "features": []}

        # Loop through every labeled directory
        for i, f in enumerate(os.scandir(dataset_path)):
            print(f"Processing target {i} ({f.name})")

            # Save semantic and target labels
            data["semantic_labels"].append(f.name)
            data["targets"].append(i)
            features = []

            # Sort the file paths in case the songs are sequence-dependent data per label
            samples = [sample for sample in os.scandir(f.path)]
            samples.sort(key=lambda x: x.name)

            # Loop through the generated paths (up till `samples_per_label`)
            for sample in tqdm(samples[:self.samples_per_label]):
                if sample.is_file():
                    # If it is a file, generate segmented features for a single stem
                    feature_segments = self._segmented_extraction(sample.path, extractions)

                if sample.is_dir():
                    # If it is a folder with stems, generate segmented features for each stem
                    feature_segments = [self._segmented_extraction(os.path.join(sample.path, stem),
                                        extractions) for stem in sorted(os.listdir(sample.path))]
                    feature_segments = np.concatenate(feature_segments, axis=1).tolist()

                # Append segment-based features
                features.append(feature_segments)

            # Append song-based features
            data["features"].append(features)

        return data


    def extract_features(self, dataset_path, *args):

        extractions = self._validate_requests(args)
        return self._walk_extract(dataset_path, extractions)


    def restructure(self, inputs, targets, meanings, label_level=2, feature_level=None, squeeze=True, tolist=False):
        if feature_level is None:
            # Default feature level
            feature_level = label_level

        # Feature level cannot be lower than label level
        assert feature_level >= label_level

        # Apply labeling transformation
        targets = np.resize(targets, inputs.shape[label_level::-1]).T.reshape(-1)
        meanings = np.resize(meanings, inputs.shape[label_level::-1]).T.reshape(-1)
        inputs = inputs.reshape(-1, *inputs.shape[label_level+1:])

        # Apply "refeaturing" transformation
        inputs = np.moveaxis(inputs, 4 - label_level, feature_level - label_level)

        if inputs.shape[feature_level - label_level] == 1 and squeeze:
            # If the feature dimension is unnecessary, remove it
            inputs = inputs.squeeze(axis=feature_level-label_level)

        if tolist:
            # Convert to list object
            inputs = inputs.tolist()
            targets = targets.tolist()
            meanings = meanings.tolist()

        return inputs, targets, meanings


    def save_data(self, data, json_path):
        """Saves the data as a JSON file.

        Note:
            The `data` must be interpretable by JSON parser, i.e., a python list object. It should
            contain 3 entries as described in ``sef.walk_extract``, of any shape.

        Args:
            data (dict): The dictionary containg labels and inputs
            json_path (str): The path to the JSON file. Must end with a file name, e.g., "data.json"
        """
        # Create output dir if it doesn't exist
        if os.path.exists(os.path.dirname(json_path)):
            os.makedirs(os.path.dirname(json_path))

        # Save the data to a json file
        with open(json_path, 'w') as fp:
            json.dump(data, fp, indent=4)

        print("Data successfully saved as a JSON file.")


    def load_data(self, json_path):
        """Loads the data from a JSON file.

        Note:
            The JSON file must contain 3 entries described in ``sef.walk_extract``, of the exact shape.

        Returns:
            tuple: A tuple of 3 entries as described in ``self.walk_extract``:
                * `inputs`: The extracted features for each track
                * `targets`: Numeric representation of semantic labels
                * `meanings`: The name for each label
        """
        # Load the file into memory
        with open(json_path, 'r') as fp:
            data = json.load(fp)

        # Each entry should be a numpy array
        inputs = np.array(data["features"])
        targets = np.array(data["targets"])
        meanings = np.array(data["semantic_labels"])

        return inputs, targets, meanings


    def extract_save(self, dataset_path, json_path, *args):
        """Generic method to extract and save the features.

        Args:
            dataset_path (str): The path to the root directory of the labeled dataset files
            json_path (str): The path to the JSON file. Must end with a file name, e.g., "data.json"
            *args: The requested feature extractions.
                Each argument is a tuple taking 2 values - the name of the feature extraction and the
                parameters to use for that extraction (e.g., "hop_length", "n_fft") or 1 value - just
                the name. Extraction parameters that are not provided are set to default values.
        """
        extracted_features = self.extract_features(dataset_path, *args)
        self.save_data(extracted_features, json_path)


    def load_restructure(self, json_path, label_level=2, feature_level=None, squeeze=True, tolist=False):
        """Generic method to load and restructure the features.

        Args:
            json_path (str): The path to the JSON file. Must end with a file name, e.g., "data.json"
            label_level (int): The level at which each input dimension should have a label (`2` by default)
            feature_level (int): The level at which an input dimension is split to features (`None` by default)
            squeeze (bool): Whether to squeeze the feature dimension if it is of length one (`True` by default)
            tolist (bool): Whether to return the outputs as python list objects (`False` by default)

        Returns:
            tuple: A tuple of 3 entries as described in ``self.load_data``:
        """
        extracted_features = self.load_data(json_path)
        return self.restructure(*extracted_features, label_level, feature_level, squeeze, tolist)

# File path constants
SONGS_DATA_PATH = "/content/drive/MyDrive/Data/genres_original"
STEMS_DATA_PATH = "/content/drive/MyDrive/Data/genres_stems"
SONGS_JSON_PATH = "data_songs.json"
STEMS_JSON_PATH = "data_stems.json"

# Our extractor object
extractor = AudioFeatureExtractor(30, num_segments=10, samples_per_label=20)

# Feature extraction for songs ~40 sec
print("Processing original songs\n")
extractor.extract_save(SONGS_DATA_PATH, SONGS_JSON_PATH, "mfcc")

# Feature extraction for stems ~20 min
print("\nProcessing song stems\n")
extractor.extract_save(STEMS_DATA_PATH, STEMS_JSON_PATH, "mfcc")

# Targets and labels will be the same for each segment
inputs_songs, targets, meanings = extractor.load_restructure(SONGS_JSON_PATH)
inputs_stems, _, _ = extractor.load_restructure(STEMS_JSON_PATH)

# Confirm the shapes are what we expect them to be
print("\nSong features shape:", inputs_songs.shape)
print("Stem features shape:", inputs_stems.shape)

from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf
import random

SEED = 54321 # gobal seed

def reset_random_seeds():
    """Restes the random seeds for reproducable results."""
    os.environ['PYTHONHASHSEED'] = str(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def split_data(X, y, test_size=.2, validation_size=.2):
    """Splits the data into train, validation and test sets.

    Args:
        X (ndarray): The input data
        y (ndarray): The target labels
        test_size (float): The proportion of the test set
        validation_size (float): The proportion of the validation set

    Returns:
        tuple: A tuple of train, validation and test inputs and labels
    """
    # Channel should be moved to the last dimension
    X = np.moveaxis(X, 1, 3)

    # Perform the splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=SEED)

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_shape):
    """Creates a CNN model.

    Args:
        input_shape (tuple): The shape of the input dimensions of teh form (H, W, C)

    Returns:
        tuple: the created model and an optimizer (Adam) for the model
    """
    # Create a model
    model = keras.Sequential()

    # Add first conv layer
    model.add(keras.layers.Conv2D(32, 3, activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D(3, 2, "same"))
    model.add(keras.layers.BatchNormalization())

    # Add second conv layer
    model.add(keras.layers.Conv2D(32, 3, activation="relu"))
    model.add(keras.layers.MaxPool2D(3, 2, "same"))
    model.add(keras.layers.BatchNormalization())

    # Add third conv layer
    model.add(keras.layers.Conv2D(32, 2, activation="relu"))
    model.add(keras.layers.MaxPool2D(2, 2, "same"))
    model.add(keras.layers.BatchNormalization())

    # Flatten the output and add a dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(.3))

    # Output layer
    model.add(keras.layers.Dense(10, activation="softmax"))

    # Create an optimizer for the model
    optimizer = keras.optimizers.Adam(learning_rate=2e-3)

    return model, optimizer

# Assure deterministic results
reset_random_seeds()

# Create a merged representation of inputs
inputs_merge = np.concatenate((inputs_songs, inputs_stems), axis=1)

# Generate train, validation and test data
X_train_songs, X_val_songs, X_test_songs, y_train_songs, y_val_songs, y_test_songs = split_data(inputs_songs, targets)
X_train_stems, X_val_stems, X_test_stems, y_train_stems, y_val_stems, y_test_stems = split_data(inputs_stems, targets)
X_train_merge, X_val_merge, X_test_merge, y_train_merge, y_val_merge, y_test_merge = split_data(inputs_merge, targets)

# Build the models
model_songs, opt_songs = build_model(X_train_songs.shape[1:])
model_stems, opt_stems = build_model(X_train_stems.shape[1:])
model_merge, opt_merge = build_model(X_train_merge.shape[1:])

# Compile the models
model_songs.compile(optimizer=opt_songs, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model_stems.compile(optimizer=opt_stems, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model_merge.compile(optimizer=opt_merge, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the models
print("Training the model using original tracks as inputs...")
model_songs.fit(X_train_songs, y_train_songs, validation_data=(X_val_songs, y_val_songs), batch_size=32, epochs=20)
print("\nTraining the model using stems as inputs...")
model_stems.fit(X_train_stems, y_train_stems, validation_data=(X_val_stems, y_val_stems), batch_size=32, epochs=20)
print("\nTraining the model using merged inputs...")
model_merge.fit(X_train_merge, y_train_merge, validation_data=(X_val_merge, y_val_merge), batch_size=32, epochs=20)

# Get the accuracies
print("\nThe final accuracies for original, stems, merged respectively:")
test_error_songs, test_accuracy_songs = model_songs.evaluate(X_test_songs, y_test_songs)
test_error_stems, test_accuracy_stems = model_stems.evaluate(X_test_stems, y_test_stems)
test_error_merge, test_accuracy_merge = model_merge.evaluate(X_test_merge, y_test_merge)