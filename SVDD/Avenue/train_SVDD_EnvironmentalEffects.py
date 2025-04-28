import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates


def main(dataset_path):

    # ------------------------------------------------------------------------------
    # Elastic Deformations (Deformable Augmentation)
    # ------------------------------------------------------------------------------
    def elastic_transform(image, alpha, sigma, random_state=None):
        """
        Applies elastic deformation to an image.
        
        Parameters:
        image: 2D numpy array (grayscale image)
        alpha: Scaling factor that controls the intensity of the deformation.
        sigma: Standard deviation for the Gaussian filter (controls smoothness).
        random_state: (optional) NumPy RandomState for reproducibility.
        
        Returns:
        Elastically deformed image.
        
        Based on the method described by Simard et al. (2003).
        """
        if random_state is None:
            random_state = np.random.RandomState(None)
        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        distorted_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
        return distorted_image

    # ------------------------------------------------------------------------------
    # Fourier Domain Adaptation (Frequency-Based Augmentation)
    # ------------------------------------------------------------------------------
    def fourier_domain_augmentation(image, beta=0.1):
        """
        Applies frequency-based augmentation using Fourier Domain Adaptation (FDA).
        This function perturbs the low-frequency amplitude components of the image.
        
        Parameters:
        image: 2D numpy array (grayscale image)
        beta: Controls the size of the low-frequency band (relative to image dimensions).
        
        Returns:
        Augmented image with modified low-frequency components.
        
        This technique leverages the Fourier transform ([NumPy FFT](https://numpy.org/doc/stable/reference/routines.fft.html))
        and is inspired by methods in domain adaptation (see [Fourier Domain Adaptation for Semantic Segmentation](https://openaccess.thecvf.com/content_CVPR_2021/html/Liu_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2021_paper.html)).
        """
        h, w = image.shape
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        
        amplitude = np.abs(fshift)
        phase = np.angle(fshift)
        
        # Determine the size of the low-frequency region
        b = int(beta * min(h, w))
        c_h, c_w = h // 2, w // 2
        
        # Augment the low-frequency amplitude by applying a random scaling factor
        amp_aug = amplitude.copy()
        random_factor = np.random.uniform(0.5, 1.5, size=(2*b, 2*b))
        amp_aug[c_h-b:c_h+b, c_w-b:c_w+b] *= random_factor
        
        # Reconstruct the FFT with augmented amplitude
        fshift_aug = amp_aug * np.exp(1j * phase)
        f_ishift = np.fft.ifftshift(fshift_aug)
        img_back = np.fft.ifft2(f_ishift)
        img_aug = np.abs(img_back)
        # Normalize the image to [0, 1]
        img_aug = (img_aug - img_aug.min()) / (img_aug.max() - img_aug.min() + 1e-8)
        return img_aug

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

    # ------------------------------------------------------------------------------
    # Dataset Setup
    # ------------------------------------------------------------------------------
    DATASET_PATH = dataset_path
    input_shape = (200, 200, 1)

    # ------------------------------------------------------------------------------
    # 1. Data Loading Functions
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
    # 2. Abnormal Frame Ranges (Manually Provided)
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
    # 3. Load the Training & Testing Data
    # ------------------------------------------------------------------------------
    # Update paths to point to the 'frames' subdirectory
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
    # 4. Visualization Function
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
    # Augmentation: Elastic Deformation on Training Data
    # ------------------------------------------------------------------------------
    elastic_images = []
    num_samples = 5
    for i in range(num_samples):
        img = avenue_xTrain[i].squeeze()  # Use the loaded training image
        deformed_img = elastic_transform(img, alpha=34, sigma=4)
        elastic_images.append(deformed_img.reshape(input_shape[:2]))

    print("Displaying some Elastically Deformed training frames:")
    plot_images(np.array(elastic_images).reshape(-1, 200, 200, 1), "Elastic Deformation Augmented Frames", n=5)

    # ------------------------------------------------------------------------------
    # Augmentation: Fourier Domain Augmentation on Training Data
    # ------------------------------------------------------------------------------
    fourier_images = []
    for i in range(num_samples):
        img = avenue_xTrain[i].squeeze()  # Use the loaded training image
        fourier_img = fourier_domain_augmentation(img, beta=0.1)
        fourier_images.append(fourier_img.reshape(input_shape[:2]))

    print("Displaying some Fourier Domain Augmented training frames:")
    plot_images(np.array(fourier_images).reshape(-1, 200, 200, 1), "Fourier Domain Augmented Frames", n=5)

    # ------------------------------------------------------------------------------
    # Augmentation: Simulated Environmental Effects on Training Data
    # ------------------------------------------------------------------------------
    env_effects = ["rain", "fog", "occlusion"]
    for effect in env_effects:
        augmented_env = []
        for i in range(num_samples):
            img = avenue_xTrain[i].squeeze()  # Use the loaded training image
            aug_img = simulate_environmental_effects(img, effect_type=effect)
            augmented_env.append(aug_img.reshape(input_shape[:2]))
        print(f"Displaying some training frames with simulated {effect} effect:")
        plot_images(np.array(augmented_env).reshape(-1, 200, 200, 1), f"{effect.capitalize()} Augmented Frames", n=5)

    # ------------------------------------------------------------------------------
    # 5. ResNet152-based Model for Deep SVDD
    # ------------------------------------------------------------------------------
    def create_resnet_model(input_shape):
        """
        Creates a ResNet152-based model for Deep SVDD.
        The model expects grayscale images (200,200,1), replicates the single channel to 3 channels,
        passes them through ResNet152 (with pretrained ImageNet weights), and then adds dense layers to yield
        a 32-dimensional latent representation.
        """
        base_input = layers.Input(shape=input_shape)  # (200,200,1)
        
        # Replicate the grayscale channel to form a 3-channel image.
        x = layers.Concatenate()([base_input, base_input, base_input])  # now (200,200,3)
        
        # Load ResNet152 without the top layers, using pretrained weights.
        base_model = tf.keras.applications.ResNet152(
            include_top=False,
            weights='imagenet',
            input_tensor=x
        )
        
        # Freeze ResNet152 layers (optional)
        for layer in base_model.layers:
            layer.trainable = False
        
        # Global Average Pooling to reduce spatial dimensions.
        x = layers.GlobalAveragePooling2D()(base_model.output)
        
        # A dense layer for additional abstraction.
        x = layers.Dense(128, activation='relu')(x)
        
        # Final output layer for the latent representation (32 dimensions).
        outputs = layers.Dense(32, activation=None)(x)
        
        model = models.Model(inputs=base_input, outputs=outputs)
        return model

    model = create_resnet_model(input_shape)
    model.summary()

    # ------------------------------------------------------------------------------
    # 6. Compute Center c in the Latent Space (Deep SVDD)
    # ------------------------------------------------------------------------------
    train_features = model.predict(avenue_xTrain, batch_size=128)
    c = np.mean(train_features, axis=0)
    c[np.abs(c) < 1e-6] = 1e-6  # Avoid near-zero values
    c_tf = tf.constant(c, dtype=tf.float32)

    # ------------------------------------------------------------------------------
    # 7. Deep SVDD Loss Function
    # ------------------------------------------------------------------------------
    def deep_svdd_loss(y_true, y_pred):
        return tf.reduce_mean(tf.reduce_sum(tf.square(y_pred - c_tf), axis=1))

    dummy_labels = np.zeros((avenue_xTrain.shape[0], 1))
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss=deep_svdd_loss)

    if avenue_xTrain.shape[0] < 128:
        print("Warning: Training dataset size is smaller than the batch size.")

    # Train for 50 epochs (or adjust as needed)
    model.fit(avenue_xTrain, dummy_labels, epochs=50, batch_size=128, validation_split=0.1)

    # ------------------------------------------------------------------------------
    # 8. Compute Anomaly Scores on Test Data
    # ------------------------------------------------------------------------------
    def compute_anomaly_scores(model, data, center):
        features = model.predict(data, batch_size=128)
        scores = np.sum((features - center)**2, axis=1)
        return scores

    test_scores = compute_anomaly_scores(model, avenue_xTest, c)
    print("Mean anomaly score (entire test set):", np.mean(test_scores))

    normal_scores = compute_anomaly_scores(model, avenue_xTest_normal, c)
    abnormal_scores = compute_anomaly_scores(model, avenue_xTest_abnormal, c)
    print("Mean anomaly score (normal frames):", np.mean(normal_scores))
    print("Mean anomaly score (abnormal frames):", np.mean(abnormal_scores))

    # ------------------------------------------------------------------------------
    # 9. Evaluation: F1 Score and Precision-Recall AUC
    # ------------------------------------------------------------------------------
    y_true = np.concatenate([avenue_yTest_normal, avenue_yTest_abnormal]).ravel()
    y_scores = np.concatenate([normal_scores, abnormal_scores])

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    max_f1 = np.max(f1_scores)
    optimal_idx = np.argmax(f1_scores) if thresholds.size > 0 else None
    optimal_threshold = thresholds[optimal_idx] if optimal_idx is not None else None

    print("Maximum F1 Score: {:.4f}".format(max_f1))
    print("Optimal threshold for classification: {:.4f}".format(optimal_threshold))

    y_pred = (y_scores >= optimal_threshold).astype(int) if optimal_threshold is not None else np.zeros_like(y_true)
    f1_val = f1_score(y_true, y_pred)
    print("F1 Score at optimal threshold: {:.4f}".format(f1_val))

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
