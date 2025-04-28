import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

def main(dataset_path):
    # Set image size and input shape (using 256x256 grayscale)
    input_shape = (256, 256, 1)

    # Define the path to the bottle folder of MVTec AD
    bottle_path = dataset_path

    # ------------------------------------------------------------------------------
    # CutPaste (Patch-Based Augmentation)
    # ------------------------------------------------------------------------------
    # This function implements a CutPaste augmentation technique by randomly extracting a
    # patch from an image and pasting it to another location.
    # It is inspired by the ideas discussed in the paper "CutPaste: Self-Supervised Learning 
    # for Anomaly Detection and Localization" (https://arxiv.org/abs/2203.06516) cite

    def cutpaste_augmentation(image, scale_range=(0.02, 0.15)):
        """
        Applies CutPaste augmentation by randomly extracting a patch from the image
        and pasting it at a random location.
        
        Parameters:
        image: 2D numpy array (grayscale image)
        scale_range: tuple defining the range of the patch area relative to the image area.
        
        Returns:
        Augmented image with a pasted patch.
        """
        h, w = image.shape
        area = h * w
        
        # Determine patch area based on a random scale factor within the specified range
        scale = np.random.uniform(scale_range[0], scale_range[1])
        patch_area = scale * area
        
        # Randomly select an aspect ratio for the patch
        aspect_ratio = np.random.uniform(0.3, 1/0.3)
        patch_h = int(np.sqrt(patch_area / aspect_ratio))
        patch_w = int(np.sqrt(patch_area * aspect_ratio))
        
        patch_h = np.clip(patch_h, 1, h - 1)
        patch_w = np.clip(patch_w, 1, w - 1)
        
        # Randomly select a location to extract the patch
        x1 = np.random.randint(0, h - patch_h)
        y1 = np.random.randint(0, w - patch_w)
        patch = image[x1:x1+patch_h, y1:y1+patch_w].copy()
        
        # Randomly select a location to paste the patch
        x2 = np.random.randint(0, h - patch_h)
        y2 = np.random.randint(0, w - patch_w)
        augmented = image.copy()
        augmented[x2:x2+patch_h, y2:y2+patch_w] = patch
        
        return augmented

    # ------------------------------------------------------------------------------
    # Data Loading and Preprocessing
    # ------------------------------------------------------------------------------
    def load_train_images(path, target_size=input_shape[:2]):
        """
        Loads training images from the 'train/good' subfolder.
        Normal images are expected to be in the "good" folder.
        """
        train_dir = os.path.join(path, 'train')
        good_folder = os.path.join(train_dir, 'good')
        image_files = [os.path.join(good_folder, f) for f in os.listdir(good_folder) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        images = []
        for file in image_files:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, target_size)
                img = img.astype('float32') / 255.0
                images.append(img)
        images = np.array(images)
        return images.reshape(-1, target_size[0], target_size[1], 1)

    def load_test_images_and_labels(path, target_size=input_shape[:2]):
        """
        Loads test images from all subfolders in 'test'. 
        Images in the 'good' folder are labeled as 0 (normal), while
        images in all other subfolders are labeled as 1 (anomalous).
        """
        test_dir = os.path.join(path, 'test')
        images = []
        labels = []  # 0 for normal, 1 for anomaly
        subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        for sub in subdirs:
            sub_path = os.path.join(test_dir, sub)
            image_files = [os.path.join(sub_path, f) for f in os.listdir(sub_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for file in image_files:
                img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    img = img.astype('float32') / 255.0
                    images.append(img)
                    # Label normal images (from 'good') as 0, otherwise label as 1.
                    label = 0 if sub.lower() == 'good' else 1
                    labels.append(label)
        images = np.array(images)
        labels = np.array(labels)
        return images.reshape(-1, target_size[0], target_size[1], 1), labels

    # Load the datasets
    X_train = load_train_images(bottle_path)
    X_test, y_test = load_test_images_and_labels(bottle_path)

    print("Train images shape:", X_train.shape)
    print("Test images shape:", X_test.shape)
    print("Test labels shape:", y_test.shape)

    # ------------------------------------------------------------------------------
    # Visualization (Optional)
    # ------------------------------------------------------------------------------
    def plot_images(images, title, n=5):
        plt.figure(figsize=(15, 3))
        for i in range(n):
            plt.subplot(1, n, i+1)
            plt.imshow(images[i].reshape(input_shape[:2]), cmap='gray')
            plt.axis('off')
        plt.suptitle(title)
        plt.show()

    # Visualize the raw training and test images
    plot_images(X_train, "Training Images")
    plot_images(X_test, "Test Images")

    # ------------------------------------------------------------------------------
    # Generate and Display CutPaste Augmented Training Frames
    # ------------------------------------------------------------------------------
    # For demonstration, some training images are augmented using the CutPaste technique.
    augmented_images = []
    num_samples = 5
    for i in range(num_samples):
        # Squeeze to remove the channel dimension for augmentation
        img = X_train[i].squeeze()
        aug_img = cutpaste_augmentation(img)
        augmented_images.append(aug_img.reshape(input_shape[:2]))

    print("Displaying some CutPaste augmented training frames:")
    # Updated reshaping to use the original image dimensions (256x256)
    plot_images(np.array(augmented_images).reshape(-1, input_shape[0], input_shape[1], 1),
                "CutPaste Augmented Frames", n=5)

    # ------------------------------------------------------------------------------
    # Deep SVDD Model Implementation
    # ------------------------------------------------------------------------------
    def create_model(input_shape):
        """
        Defines a simple CNN that maps images into a latent space.
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

    model = create_model(input_shape)
    model.summary()

    # ------------------------------------------------------------------------------
    # Step 1: Compute the Center "c" in the Latent Space Using the Training Data
    # ------------------------------------------------------------------------------
    train_features = model.predict(X_train, batch_size=32)
    c = np.mean(train_features, axis=0)
    c[np.abs(c) < 1e-6] = 1e-6  # Adjust near-zero values to avoid numerical issues
    c_tf = tf.constant(c, dtype=tf.float32)

    # ------------------------------------------------------------------------------
    # Step 2: Define the Deep SVDD Loss Function
    # ------------------------------------------------------------------------------
    def deep_svdd_loss(y_true, y_pred):
        """
        Computes the loss as the squared Euclidean distance from the latent representation to center c.
        """
        return tf.reduce_mean(tf.reduce_sum(tf.square(y_pred - c_tf), axis=1))

    # Since Deep SVDD is unsupervised, create dummy labels for training.
    dummy_labels = np.zeros((X_train.shape[0], 1))

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                loss=deep_svdd_loss)

    # ------------------------------------------------------------------------------
    # Step 3: Train the Deep SVDD Model
    # ------------------------------------------------------------------------------
    model.fit(X_train, dummy_labels, epochs=50, batch_size=32, validation_split=0.1)

    # ------------------------------------------------------------------------------
    # Step 4: Compute Anomaly Scores for the Test Data
    # ------------------------------------------------------------------------------
    def compute_anomaly_scores(model, data, center):
        features = model.predict(data, batch_size=32)
        scores = np.sum((features - center)**2, axis=1)
        return scores

    test_scores = compute_anomaly_scores(model, X_test, c)
    print("Mean anomaly score for test set:", np.mean(test_scores))

    # ------------------------------------------------------------------------------
    # Evaluation: F1 Score and Precision-Recall AUC (Average Precision)
    # ------------------------------------------------------------------------------
    precisions, recalls, thresholds = precision_recall_curve(y_test, test_scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    max_f1 = np.max(f1_scores)
    optimal_threshold = thresholds[np.argmax(f1_scores)] if thresholds.size > 0 else 0.0

    y_pred = (test_scores >= optimal_threshold).astype(int)
    final_f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, test_scores)

    print("Maximum F1 Score (from PR curve): {:.4f}".format(max_f1))
    print("Optimal threshold for classification: {:.4f}".format(optimal_threshold))
    print("F1 Score at optimal threshold: {:.4f}".format(final_f1))
    print("Precision-Recall AUC (Average Precision): {:.4f}".format(pr_auc))

    
if __name__ == '__main__':
    

    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
