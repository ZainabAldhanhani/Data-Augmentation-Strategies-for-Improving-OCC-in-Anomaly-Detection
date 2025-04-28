import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score
import matplotlib.pyplot as plt

def main(dataset_path):
    # ------------------------------------------------------------------------------
    # Dataset Loading
    # ------------------------------------------------------------------------------
    DATASET_PATH = dataset_path
    input_shape = (200, 200, 1)

    def load_train_dataset(directory):
        X = []
        for folder in os.listdir(directory):
            folder_path = os.path.join(directory, folder)
            if os.path.isdir(folder_path) and folder.startswith("Train"):
                for filename in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, filename)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            img = cv2.resize(img, (200, 200))
                            img = img.astype('float32') / 255.0
                            X.append(img)
                    except Exception as e:
                        print(f"Error processing image {img_path}: {e}")
        X = np.array(X)
        return X.reshape(-1, 200, 200, 1)

    def load_test_dataset(directory):
        X = []
        for folder in os.listdir(directory):
            folder_path = os.path.join(directory, folder)
            if os.path.isdir(folder_path) and not folder.endswith("_gt") and folder.startswith("Test"):
                for filename in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, filename)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            img = cv2.resize(img, (200, 200))
                            img = img.astype('float32') / 255.0
                            X.append(img)
                    except Exception as e:
                        print(f"Error processing image {img_path}: {e}")
        X = np.array(X)
        return X.reshape(-1, 200, 200, 1)

    # Abnormal frame ranges for each test video (as provided)
    abnormal_frame_ranges = [
        range(61, 181),
        range(95, 181),
        range(1, 147),
        range(31, 181),
        range(1, 130),
        range(1, 160),
        range(46, 181),
        range(1, 181),
        range(1, 121),
        range(1, 151),
        range(1, 181),
        range(88, 181)
    ]

    def load_and_divide_test_dataset(directory, abnormal_frame_ranges):
        normal_frames = []
        abnormal_frames = []
        normal_labels = []
        abnormal_labels = []
        folder_index = 0

        for folder in sorted(os.listdir(directory)):
            folder_path = os.path.join(directory, folder)
            if os.path.isdir(folder_path) and not folder.endswith("_gt") and folder.startswith("Test"):
                frame_range = abnormal_frame_ranges[folder_index]
                file_index = 1

                print(f"Processing folder: {folder}")
                print(f"Abnormal frame range: {frame_range}")

                for filename in sorted(os.listdir(folder_path)):
                    img_path = os.path.join(folder_path, filename)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            img = cv2.resize(img, (200, 200))
                            img = img.astype('float32') / 255.0

                            if file_index in frame_range:
                                abnormal_frames.append(img)
                                abnormal_labels.append(1)
                            else:
                                normal_frames.append(img)
                                normal_labels.append(0)
                        file_index += 1
                    except Exception as e:
                        print(f"Error processing image {img_path}: {e}")
                folder_index += 1

        normal_frames = np.array(normal_frames).reshape(-1, 200, 200, 1)
        abnormal_frames = np.array(abnormal_frames).reshape(-1, 200, 200, 1)
        normal_labels = np.zeros((len(normal_labels), 1))
        abnormal_labels = np.ones((len(abnormal_frames), 1))
        return normal_frames, abnormal_frames, normal_labels, abnormal_labels

    # Load datasets
    try:
        UCSD_xTrain = load_train_dataset(os.path.join(DATASET_PATH, "Train"))
        print("Shape of X_train:", UCSD_xTrain.shape)
    except Exception as e:
        print(f"Error loading training dataset: {e}")
        raise

    try:
        UCSD_xTest = load_test_dataset(os.path.join(DATASET_PATH, "Test"))
        print("Shape of X_test:", UCSD_xTest.shape)
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        raise

    try:
        UCSD_xTest_normal, UCSD_xTest_abnormal, UCSD_yTest_normal, UCSD_yTest_abnormal = load_and_divide_test_dataset(
            os.path.join(DATASET_PATH, "Test"),
            abnormal_frame_ranges
        )
        print("Shape of UCSD_xTest_normal:", UCSD_xTest_normal.shape)
        print("Shape of UCSD_xTest_abnormal:", UCSD_xTest_abnormal.shape)
        print("Shape of UCSD_yTest_normal:", UCSD_yTest_normal.shape)
        print("Shape of UCSD_yTest_abnormal:", UCSD_yTest_abnormal.shape)
    except Exception as e:
        print(f"Error loading and dividing test dataset: {e}")
        raise

    # ------------------------------------------------------------------------------
    # Visualization: Plot Sample Frames
    # ------------------------------------------------------------------------------
    def plot_images(images, title, n=5):
        plt.figure(figsize=(15, 3))
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.imshow(images[i].reshape(input_shape[:2]), cmap='gray')
            plt.axis('off')
        plt.suptitle(title)
        plt.show()

    print("Displaying some training frames:")
    plot_images(UCSD_xTrain, "Training Frames", n=5)
    print("Displaying some testing frames:")
    plot_images(UCSD_xTest, "Testing Frames", n=5)

    # ------------------------------------------------------------------------------
    # Simulated Environmental Effects (Rain, Fog, Occlusion)
    # ------------------------------------------------------------------------------
    def add_rain(image, rain_intensity=0.5, drop_length=10):
        """
        Simulates rain by adding white streaks to the image.
        """
        rain_layer = np.zeros_like(image)
        num_drops = int(rain_intensity * image.shape[0] * image.shape[1] / 500)
        for _ in range(num_drops):
            x = np.random.randint(0, image.shape[0])
            y = np.random.randint(0, image.shape[1])
            for l in range(drop_length):
                if x + l < image.shape[0]:
                    rain_layer[x + l, y] = 1.0
        rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
        return np.clip(image + rain_layer * 0.5, 0, 1)

    def add_fog(image, fog_intensity=0.5):
        """
        Simulates fog by blending the image with a soft white mask.
        """
        h, w = image.shape
        fog = np.random.rand(h, w)
        fog = cv2.GaussianBlur(fog, (31, 31), 0)
        fog = (fog - fog.min()) / (fog.max() - fog.min())
        return np.clip(image * (1 - fog_intensity) + fog_intensity * fog, 0, 1)

    def add_occlusion(image, occlusion_ratio=0.2):
        """
        Simulates occlusion by overlaying a black rectangle.
        """
        h, w = image.shape
        occ_h = int(h * occlusion_ratio)
        occ_w = int(w * occlusion_ratio)
        x = np.random.randint(0, h - occ_h)
        y = np.random.randint(0, w - occ_w)
        occluded = image.copy()
        occluded[x:x+occ_h, y:y+occ_w] = 0.0
        return occluded

    def simulate_environmental_effects(image, effect_type="rain"):
        """
        Applies a simulated environmental effect to the image.
        effect_type can be "rain", "fog", or "occlusion".
        """
        if effect_type == "rain":
            return add_rain(image)
        elif effect_type == "fog":
            return add_fog(image)
        elif effect_type == "occlusion":
            return add_occlusion(image)
        else:
            return image

    # Generate and display simulated environmental effects on training frames
    num_samples = 5
    env_effects = ["rain", "fog", "occlusion"]
    for effect in env_effects:
        augmented_env = []
        for i in range(num_samples):
            img = UCSD_xTrain[i].squeeze()
            aug_img = simulate_environmental_effects(img, effect_type=effect)
            augmented_env.append(aug_img.reshape(input_shape[:2]))
        print(f"Displaying some training frames with simulated {effect} effect:")
        plot_images(np.array(augmented_env).reshape(-1, 200, 200, 1), f"{effect.capitalize()} Augmented Frames", n=5)

    # ------------------------------------------------------------------------------
    # Deep SVDD Implementation
    # ------------------------------------------------------------------------------
    def create_model(input_shape):
        """
        Defines a convolutional neural network to map images into a latent space.
        """
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(32, activation=None)(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    # Create the model
    model = create_model(input_shape)
    model.summary()

    # ------------------------------------------------------------------------------
    # Step 1: Compute the Center c in the Latent Space Using Training Data
    # ------------------------------------------------------------------------------
    train_features = model.predict(UCSD_xTrain, batch_size=128)
    c = np.mean(train_features, axis=0)
    c[np.abs(c) < 1e-6] = 1e-6  # Avoid near-zero values
    c_tf = tf.constant(c, dtype=tf.float32)

    # ------------------------------------------------------------------------------
    # Step 2: Define the Deep SVDD Loss Function
    # ------------------------------------------------------------------------------
    def deep_svdd_loss(y_true, y_pred):
        """
        Computes the loss as the squared Euclidean distance between the latent representation and center c.
        """
        return tf.reduce_mean(tf.reduce_sum(tf.square(y_pred - c_tf), axis=1))

    # ------------------------------------------------------------------------------
    # Step 3: Compile and Train the Model
    # ------------------------------------------------------------------------------
    dummy_labels = np.zeros((UCSD_xTrain.shape[0], 1))  # Dummy labels (unsupervised training)

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                loss=deep_svdd_loss)

    model.fit(UCSD_xTrain, dummy_labels, epochs=50, batch_size=128, validation_split=0.1)

    # ------------------------------------------------------------------------------
    # Step 4: Compute Anomaly Scores on Test Data
    # ------------------------------------------------------------------------------
    def compute_anomaly_scores(model, data, center):
        """
        Computes anomaly scores as the squared Euclidean distance from the latent representation to the center.
        """
        features = model.predict(data, batch_size=128)
        scores = np.sum((features - center)**2, axis=1)
        return scores

    test_scores = compute_anomaly_scores(model, UCSD_xTest, c)
    print("Mean anomaly score for overall test set:", np.mean(test_scores))

    normal_scores = compute_anomaly_scores(model, UCSD_xTest_normal, c)
    abnormal_scores = compute_anomaly_scores(model, UCSD_xTest_abnormal, c)
    print("Mean anomaly score for normal test frames:", np.mean(normal_scores))
    print("Mean anomaly score for abnormal test frames:", np.mean(abnormal_scores))

    # ------------------------------------------------------------------------------
    # Evaluation: F1 Score and Precision-Recall AUC
    # ------------------------------------------------------------------------------
    y_true = np.concatenate([UCSD_yTest_normal, UCSD_yTest_abnormal]).ravel()
    y_scores = np.concatenate([normal_scores, abnormal_scores])

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    max_f1 = np.max(f1_scores)
    optimal_threshold = thresholds[np.argmax(f1_scores)] if thresholds.size > 0 else None

    print("Maximum F1 Score: {:.4f}".format(max_f1))
    print("Optimal threshold for classification: {:.4f}".format(optimal_threshold))

    y_pred = (y_scores >= optimal_threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    print("F1 Score at optimal threshold: {:.4f}".format(f1))

    pr_auc = average_precision_score(y_true, y_scores)
    print("Precision-Recall AUC (Average Precision): {:.4f}".format(pr_auc))

    
if __name__ == '__main__':
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
