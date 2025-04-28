import argparse
import os
import cv2  # OpenCV for image operations (https://opencv.org/) cite:oaicite:1
import numpy as np
import tensorflow as tf  # TensorFlow for deep learning (https://www.tensorflow.org/) cite:oaicite:2
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score  # scikit-learn (https://scikit-learn.org/stable/) cite:oaicite:3
import matplotlib.pyplot as plt  # matplotlib for plotting (https://matplotlib.org/) cite:oaicite:4

def main(dataset_path):
    
    # ------------------------------------------------------------------------------
    # Dataset Setup
    # ------------------------------------------------------------------------------
    DATASET_PATH = dataset_path
    input_shape = (200, 200, 1)

    # ------------------------------------------------------------------------------
    # 1. Data Loading Functions (unchanged)
    # ------------------------------------------------------------------------------
    def load_train_dataset(directory):
        """
        Loads images from the given directory (which should contain subfolders for each video).
        Each subfolder (e.g. '01', '02', ...) is assumed to hold frames for one video.
        """
        X = []
        for folder in sorted(os.listdir(directory)):
            folder_path = os.path.join(directory, folder)
            if os.path.isdir(folder_path):
                for filename in sorted(os.listdir(folder_path)):
                    img_path = os.path.join(folder_path, filename)
                    if os.path.isfile(img_path):
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
        """
        Loads images from the given directory (which should contain subfolders for each video).
        """
        X = []
        for folder in sorted(os.listdir(directory)):
            folder_path = os.path.join(directory, folder)
            if os.path.isdir(folder_path):
                for filename in sorted(os.listdir(folder_path)):
                    img_path = os.path.join(folder_path, filename)
                    if os.path.isfile(img_path):
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

    # ------------------------------------------------------------------------------
    # 2. Abnormal Frame Ranges (Manually Provided, unchanged)
    # ------------------------------------------------------------------------------
    abnormal_frame_ranges = [
        range(78, 393),
        range(273, 725),
        range(295, 583),
        range(380, 650),
        range(469, 787),
        range(345, 857),
        range(423, 564),
        range(21, 31),
        range(136, 497),
        range(571, 638),
        range(21, 309),
        range(539, 646),
        range(259, 459),
        range(399, 486),
        range(498, 588),
        range(632, 731),
        range(21, 100),
        range(21, 286),
        range(109, 241),
        range(65, 169),
        range(14, 67),
    ]

    def load_and_divide_test_dataset(directory, abnormal_frame_ranges):
        """
        Loads test images from subfolders and splits them into normal (label=0)
        and abnormal (label=1) frames based on pre-defined abnormal_frame_ranges.
        """
        normal_frames = []
        abnormal_frames = []
        normal_labels = []
        abnormal_labels = []
        folder_index = 0

        for folder in sorted(os.listdir(directory)):
            folder_path = os.path.join(directory, folder)
            if not os.path.isdir(folder_path):
                continue
            if folder_index >= len(abnormal_frame_ranges):
                print(f"Warning: More subfolders than abnormal_frame_ranges entries. Skipping '{folder}'")
                continue

            frame_range = abnormal_frame_ranges[folder_index]
            file_index = 1

            print(f"Processing folder: {folder}")
            print(f"Abnormal frame range: {frame_range}")

            for filename in sorted(os.listdir(folder_path)):
                img_path = os.path.join(folder_path, filename)
                if not os.path.isfile(img_path):
                    continue
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
        normal_labels = np.zeros((len(normal_frames), 1))
        abnormal_labels = np.ones((len(abnormal_frames), 1))
        return normal_frames, abnormal_frames, normal_labels, abnormal_labels

    # ------------------------------------------------------------------------------
    # 3. Load the Training & Testing Data (unchanged)
    # ------------------------------------------------------------------------------
    train_path = os.path.join(DATASET_PATH, "training", "frames")
    test_path  = os.path.join(DATASET_PATH, "testing", "frames")

    try:
        avenue_xTrain = load_train_dataset(train_path)
        print("Shape of avenue_xTrain:", avenue_xTrain.shape)
        if avenue_xTrain.shape[0] == 0:
            raise ValueError("Training dataset is empty. Check directory: " + train_path)
    except Exception as e:
        print(f"Error loading training dataset: {e}")
        raise

    try:
        avenue_xTest = load_test_dataset(test_path)
        print("Shape of avenue_xTest:", avenue_xTest.shape)
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        raise

    try:
        (avenue_xTest_normal,
        avenue_xTest_abnormal,
        avenue_yTest_normal,
        avenue_yTest_abnormal) = load_and_divide_test_dataset(test_path, abnormal_frame_ranges)
        print("Shape of avenue_xTest_normal:", avenue_xTest_normal.shape)
        print("Shape of avenue_xTest_abnormal:", avenue_xTest_abnormal.shape)
        print("Shape of avenue_yTest_normal:", avenue_yTest_normal.shape)
        print("Shape of avenue_yTest_abnormal:", avenue_yTest_abnormal.shape)
    except Exception as e:
        print(f"Error loading/dividing test dataset: {e}")
        raise

    # ------------------------------------------------------------------------------
    # 4. Visualization Function (unchanged)
    # ------------------------------------------------------------------------------
    def plot_images(images, title, n=5):
        plt.figure(figsize=(15, 3))
        for i in range(min(n, len(images))):
            plt.subplot(1, n, i+1)
            plt.imshow(images[i].reshape(input_shape[:2]), cmap='gray')
            plt.axis('off')
        plt.suptitle(title)
        plt.show()

    print("Displaying some training frames:")
    plot_images(avenue_xTrain, "Training Frames", n=5)
    print("Displaying some testing frames:")
    plot_images(avenue_xTest, "Testing Frames", n=5)

    # ------------------------------------------------------------------------------
    # 5. CUTPASTE AUGMENTATION FUNCTIONS
    # ------------------------------------------------------------------------------
    def cutpaste_augmentation(image, area_ratio=(0.02, 0.15), aspect_ratio_range=(0.3, 1/0.3)):
        """
        Performs CutPaste augmentation on a single image.
        A random patch is cut from the image and pasted into a different location.
        
        Adapted from:
        "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization"  
        (Li et al., 2021, https://arxiv.org/abs/2106.08992) cite:oaicite:5
        """
        H, W, C = image.shape
        image_area = H * W
        patch_area = np.random.uniform(area_ratio[0], area_ratio[1]) * image_area
        aspect_ratio = np.random.uniform(aspect_ratio_range[0], aspect_ratio_range[1])
        patch_h = int(round(np.sqrt(patch_area * aspect_ratio)))
        patch_w = int(round(np.sqrt(patch_area / aspect_ratio)))
        patch_h = min(patch_h, H)
        patch_w = min(patch_w, W)
        # Randomly select patch location to cut
        x1 = np.random.randint(0, H - patch_h + 1)
        y1 = np.random.randint(0, W - patch_w + 1)
        patch = image[x1:x1+patch_h, y1:y1+patch_w, :].copy()
        # Randomly select location to paste the patch
        x2 = np.random.randint(0, H - patch_h + 1)
        y2 = np.random.randint(0, W - patch_w + 1)
        augmented = image.copy()
        augmented[x2:x2+patch_h, y2:y2+patch_w, :] = patch
        return augmented

    def create_cutpaste_dataset(X):
        """
        For each image in the training dataset, generate a CutPaste-augmented version.
        """
        augmented_images = np.array([cutpaste_augmentation(img) for img in X])
        return augmented_images

    # Generate CutPaste augmented training data
    augmented_xTrain = create_cutpaste_dataset(avenue_xTrain)

    # Create labels: original images are normal (0), and augmented images are synthetic anomalies (1)
    labels_normal = np.zeros((avenue_xTrain.shape[0], 1))
    labels_anomaly = np.ones((augmented_xTrain.shape[0], 1))

    # Combine the normal and augmented images
    X_train_combined = np.concatenate([avenue_xTrain, augmented_xTrain], axis=0)
    y_train_combined = np.concatenate([labels_normal, labels_anomaly], axis=0)

    # Shuffle the combined training dataset
    perm = np.random.permutation(X_train_combined.shape[0])
    X_train_combined = X_train_combined[perm]
    y_train_combined = y_train_combined[perm]

    # ------------------------------------------------------------------------------
    # 5a. Visualization: Display Augmented Training Frames
    # ------------------------------------------------------------------------------
    # Here we display a few examples of the CutPaste augmented images.
    print("Displaying some CutPaste augmented training frames:")
    # Slicing the first 5 augmented images for display
    plot_images(augmented_xTrain[:5], "CutPaste Augmented Frames", n=5)

    # ------------------------------------------------------------------------------
    # 6. Build a ResNet152-based Model for CutPaste-based Anomaly Detection
    # ------------------------------------------------------------------------------
    def create_resnet_model_cutpaste(input_shape):
        """
        Creates a model that uses ResNet152 as a feature extractor (replicating the grayscale
        channel to 3 channels) and adds a binary classification head.
        
        The classifier is trained to distinguish between original (normal) images and 
        CutPaste-augmented (anomalous) images.
        """
        base_input = layers.Input(shape=input_shape)  # (200,200,1)
        # Replicate the single channel to 3 channels
        x = layers.Concatenate()([base_input, base_input, base_input])  # now (200,200,3)
        base_model = tf.keras.applications.ResNet152(
            include_top=False,
            weights='imagenet',
            input_tensor=x
        )
        # Freeze ResNet152 layers (optional)
        for layer in base_model.layers:
            layer.trainable = False
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        # Binary classification head: probability of being a CutPaste-augmented anomaly
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs=base_input, outputs=outputs)
        return model

    model_cutpaste = create_resnet_model_cutpaste(input_shape)
    model_cutpaste.summary()

    # ------------------------------------------------------------------------------
    # 7. Train the Model using Binary Crossentropy Loss
    # ------------------------------------------------------------------------------
    model_cutpaste.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

    model_cutpaste.fit(X_train_combined, y_train_combined, epochs=50, batch_size=128, validation_split=0.1)

    # ------------------------------------------------------------------------------
    # 8. Evaluate on Test Data using the Trained Classifier
    # ------------------------------------------------------------------------------
    # Here, we use the same division of test data (normal vs. abnormal) as before.
    # We compute the anomaly score as the classifier’s predicted probability.

    normal_probs = model_cutpaste.predict(avenue_xTest_normal, batch_size=128)
    abnormal_probs = model_cutpaste.predict(avenue_xTest_abnormal, batch_size=128)

    y_true_test = np.concatenate([avenue_yTest_normal, avenue_yTest_abnormal]).ravel()
    y_scores_test = np.concatenate([normal_probs, abnormal_probs]).ravel()

    precisions, recalls, thresholds = precision_recall_curve(y_true_test, y_scores_test)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    max_f1 = np.max(f1_scores)
    optimal_idx = np.argmax(f1_scores) if thresholds.size > 0 else None
    optimal_threshold = thresholds[optimal_idx] if optimal_idx is not None else None

    print("Maximum F1 Score: {:.4f}".format(max_f1))
    print("Optimal threshold for classification: {:.4f}".format(optimal_threshold))

    y_pred_test = (y_scores_test >= optimal_threshold).astype(int) if optimal_threshold is not None else np.zeros_like(y_true_test)
    f1_val_test = f1_score(y_true_test, y_pred_test)
    print("F1 Score at optimal threshold: {:.4f}".format(f1_val_test))

    pr_auc_test = average_precision_score(y_true_test, y_scores_test)
    print("Precision-Recall AUC (Average Precision): {:.4f}".format(pr_auc_test))

    
if __name__ == '__main__':
    

    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
