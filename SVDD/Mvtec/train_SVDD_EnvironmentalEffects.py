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
    # Simulated Environmental Effects (Rain, Fog, Occlusion)
    # ------------------------------------------------------------------------------
    def add_rain(image, rain_intensity=0.5, drop_length=10):
        """
        Simulates rain by adding white streaks to the image.
        
        Parameters:
        image: 2D numpy array (grayscale image)
        rain_intensity: controls the number of rain drops.
        drop_length: length of each rain drop.
        
        Returns:
        Image with simulated rain effect.
        """
        rain_layer = np.zeros_like(image)
        num_drops = int(rain_intensity * image.shape[0] * image.shape[1] / 500)
        for _ in range(num_drops):
            x = np.random.randint(0, image.shape[0])
            y = np.random.randint(0, image.shape[1])
            for l in range(drop_length):
                if x + l < image.shape[0]:
                    rain_layer[x + l, y] = 1.0
        # Soften the rain effect by blurring
        rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
        return np.clip(image + rain_layer * 0.5, 0, 1)

    def add_fog(image, fog_intensity=0.5):
        """
        Simulates fog by blending the image with a soft white mask.
        
        Parameters:
        image: 2D numpy array (grayscale image)
        fog_intensity: controls the strength of the fog effect.
        
        Returns:
        Image with simulated fog effect.
        """
        h, w = image.shape
        fog = np.random.rand(h, w)
        fog = cv2.GaussianBlur(fog, (31, 31), 0)
        fog = (fog - fog.min()) / (fog.max() - fog.min())
        return np.clip(image * (1 - fog_intensity) + fog_intensity * fog, 0, 1)

    def add_occlusion(image, occlusion_ratio=0.2):
        """
        Simulates occlusion by overlaying a black rectangle on the image.
        
        Parameters:
        image: 2D numpy array (grayscale image)
        occlusion_ratio: relative size of the occluded area.
        
        Returns:
        Image with simulated occlusion effect.
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
        
        Parameters:
        image: 2D numpy array (grayscale image)
        effect_type: one of "rain", "fog", or "occlusion".
        
        Returns:
        Augmented image with the specified environmental effect.
        """
        if effect_type == "rain":
            return add_rain(image)
        elif effect_type == "fog":
            return add_fog(image)
        elif effect_type == "occlusion":
            return add_occlusion(image)
        else:
            return image

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
        Images in the 'good' folder are labeled as 0 (normal), while images
        in all other subfolders are labeled as 1 (anomalous).
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
                    # Label images from 'good' as 0, others as 1 (anomalous)
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
    # Generate and Display Simulated Environmental Effects on Training Frames
    # ------------------------------------------------------------------------------
    num_samples = 5
    env_effects = ["rain", "fog", "occlusion"]

    for effect in env_effects:
        augmented_env = []
        # Process the first num_samples images from X_train
        for i in range(num_samples):
            # Remove the channel dimension for augmentation processing
            img = X_train[i].squeeze()
            aug_img = simulate_environmental_effects(img, effect_type=effect)
            augmented_env.append(aug_img.reshape(input_shape[:2]))
        print(f"Displaying some training frames with simulated {effect} effect:")
        # Use input_shape dimensions (256x256) for reshaping before plotting
        plot_images(np.array(augmented_env).reshape(-1, input_shape[0], input_shape[1], 1),
                    f"{effect.capitalize()} Augmented Frames", n=5)

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

    # Create dummy labels since Deep SVDD is unsupervised.
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
