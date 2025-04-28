import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve

def main(dataset_path):
  
    # Set image size and input shape (using 256x256 grayscale)
    input_shape = (256, 256, 1)

    # Define the path to the bottle folder of MVTec AD
    bottle_path = dataset_path

    # -------------------------------
    # Data Loading and Preprocessing
    # -------------------------------
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
        labels = []
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
                    # Label normal images (from 'good') as 0, others as 1.
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

    # -------------------------------
    # Visualization (Optional)
    # -------------------------------
    def plot_images(images, title, n=5):
        plt.figure(figsize=(15, 3))
        for i in range(n):
            plt.subplot(1, n, i+1)
            plt.imshow(images[i].reshape(input_shape[:2]), cmap='gray')
            plt.axis('off')
        plt.suptitle(title)
        plt.show()

    # Visualize some raw training images
    plot_images(X_train, "Training Images", n=5)

    # ------------------------------------------------------------------------------
    # Fourier Domain Adaptation (Frequency-Based Augmentation)
    # ------------------------------------------------------------------------------
    def fourier_domain_augmentation(image, beta=0.1):
        """
        Applies frequency-based augmentation using Fourier Domain Adaptation (FDA).
        This function perturbs the low-frequency amplitude components of the image.
        
        Parameters:
        image: 2D numpy array (grayscale image)
        beta: controls the size of the low-frequency band (relative to image dimensions)
        
        Returns:
        Augmented image with modified low-frequency components.
        """
        h, w = image.shape
        # Compute the 2D Fourier Transform and shift the zero frequency component to the center
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        
        amplitude = np.abs(fshift)
        phase = np.angle(fshift)
        
        # Determine the size of the low-frequency region to perturb
        b = int(beta * min(h, w))
        c_h, c_w = h // 2, w // 2
        
        # Perturb the low-frequency amplitude using a random scaling factor
        amp_aug = amplitude.copy()
        random_factor = np.random.uniform(0.5, 1.5, size=(2 * b, 2 * b))
        amp_aug[c_h-b:c_h+b, c_w-b:c_w+b] *= random_factor
        
        # Reconstruct the modified Fourier transform and inverse-transform to obtain the augmented image
        fshift_aug = amp_aug * np.exp(1j * phase)
        f_ishift = np.fft.ifftshift(fshift_aug)
        img_back = np.fft.ifft2(f_ishift)
        img_aug = np.abs(img_back)
        
        # Normalize the augmented image to [0, 1]
        img_aug = (img_aug - img_aug.min()) / (img_aug.max() - img_aug.min() + 1e-8)
        return img_aug

    # Generate and display Fourier Domain augmented training frames
    fourier_images = []
    num_samples = 5
    for i in range(num_samples):
        # Remove the channel dimension for augmentation processing
        img = X_train[i].squeeze()
        fourier_img = fourier_domain_augmentation(img, beta=0.1)
        fourier_images.append(fourier_img.reshape(input_shape[:2]))

    print("Displaying some Fourier Domain Augmented training frames:")
    plot_images(np.array(fourier_images).reshape(-1, input_shape[0], input_shape[1], 1),
                "Fourier Domain Augmented Frames", n=5)

    # -------------------------------
    # Deep SVDD Model Implementation
    # -------------------------------
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

    # Since Deep SVDD is unsupervised, we create dummy labels
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

    # -------------------------------
    # Evaluation: F1 Score and Precision-Recall AUC (Average Precision)
    # -------------------------------
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
