import os
import numpy as np
import cv2
import albumentations as A
import random
from datetime import datetime
from sklearn.metrics import classification_report
import tensorflow as tf
from keras_unet_collection import models, losses
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.optimizers import Adam
import pickle
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt



class DataPaths:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.e_ophtha = {
            'train': {'EX': {'images': [], 'masks': []}, 'MA': {'images': [], 'masks': []}},
            'val': {'EX': {'images': [], 'masks': []}, 'MA': {'images': [], 'masks': []}},
            'test': {'EX': {'images': [], 'masks': []}, 'MA': {'images': [], 'masks': []}}
        }
        self.idrid = {
            'train': {'EX': {'images': [], 'masks': []}, 'MA': {'images': [], 'masks': []}},
            'val': {'EX': {'images': [], 'masks': []}, 'MA': {'images': [], 'masks': []}},
            'test': {'EX': {'images': [], 'masks': []}, 'MA': {'images': [], 'masks': []}}
        }

    def _get_files(self, directory):
        """Get all files in the directory, recursively."""
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_paths.append(os.path.join(root, file))
        return file_paths

    def run(self):
        # Load data for e_ophtha EX and MA datasets
        ex_base = os.path.join(self.base_dir, "e_ophtha_EX")
        ma_base = os.path.join(self.base_dir, "e_ophtha_MA")

        # Exudates
        ex_images = self._get_files(os.path.join(ex_base, "EX"))
        ex_masks = self._get_files(os.path.join(ex_base, "Annotation_EX"))
        self.e_ophtha['train']['EX'] = {'images': ex_images[:30], 'masks': ex_masks[:30]}
        self.e_ophtha['val']['EX'] = {'images': ex_images[30:37], 'masks': ex_masks[30:37]}
        self.e_ophtha['test']['EX'] = {'images': ex_images[37:], 'masks': ex_masks[37:]}

        # Microaneurysms
        ma_images = self._get_files(os.path.join(ma_base, "MA"))
        ma_masks = self._get_files(os.path.join(ma_base, "Annotation_MA"))
        self.e_ophtha['train']['MA'] = {'images': ma_images[:104], 'masks': ma_masks[:104]}
        self.e_ophtha['val']['MA'] = {'images': ma_images[104:118], 'masks': ma_masks[104:118]}
        self.e_ophtha['test']['MA'] = {'images': ma_images[118:], 'masks': ma_masks[118:]}

        # Idrid data
        train_imgs = self._get_files(os.path.join(self.base_dir, "idrid", "train_images"))
        train_ex_masks = self._get_files(os.path.join(self.base_dir, "idrid", "train_ex_masks"))
        train_ma_masks = self._get_files(os.path.join(self.base_dir, "idrid", "train_ma_masks"))
        test_imgs = self._get_files(os.path.join(self.base_dir, "idrid", "test_images"))
        test_ex_masks = self._get_files(os.path.join(self.base_dir, "idrid", "test_ex_masks"))
        test_ma_masks = self._get_files(os.path.join(self.base_dir, "idrid", "test_ma_masks"))

        self.idrid['train']['EX'] = {'images': train_imgs[:44], 'masks': train_ex_masks[:44]}
        self.idrid['val']['EX'] = {'images': train_imgs[44:], 'masks': train_ex_masks[44:]}
        self.idrid['test']['EX'] = {'images': test_imgs, 'masks': test_ex_masks}

        self.idrid['train']['MA'] = {'images': train_imgs[:44], 'masks': train_ma_masks[:44]}
        self.idrid['val']['MA'] = {'images': train_imgs[44:], 'masks': train_ma_masks[44:]}
        self.idrid['test']['MA'] = {'images': test_imgs, 'masks': test_ma_masks}


class DataPreprocessor:
    @staticmethod
    def create_inputs(path_images, path_masks):
        X = np.zeros((len(path_images), 256, 256, 3), dtype=np.uint8)
        Y = np.zeros((len(path_masks), 256, 256), dtype=bool)

        for i, (image_path, mask_path) in enumerate(zip(path_images, path_masks)):
            img = cv2.imread(image_path, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            X[i] = img

            mask = cv2.imread(mask_path, 0)
            mask = cv2.resize(mask, (256, 256))
            Y[i] = mask

        return X, Y


class DataAugmentation:
    def __init__(self, X_train, Y_train, num_aug):
        self.X_train = X_train
        self.Y_train = Y_train
        self.num_aug = num_aug - len(X_train)

    def augment_images(self):
        x_array = np.zeros((self.num_aug, 256, 256, 3), dtype=np.uint8)
        y_array = np.zeros((self.num_aug, 256, 256), dtype=np.uint8)

        transform = A.Compose([
            A.HorizontalFlip(p=0.6),
            A.RandomBrightnessContrast(p=0.3),
            A.VerticalFlip(p=0.6),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.2),
            A.Rotate(p=0.5, limit=(-45, 45)),
            A.MultiplicativeNoise(p=1.0),
        ])

        for i in range(self.num_aug):
            idx = random.randint(0, len(self.X_train) - 1)
            img = self.X_train[idx]
            mask = self.Y_train[idx].astype(np.uint8)

            transformed = transform(image=img, mask=mask)
            x_array[i] = transformed['image']
            y_array[i] = transformed['mask']

        return x_array, y_array

class ExperimentConfig:
    def __init__(self, config, base_dir='/content/drive/MyDrive/DRV2'):
        """
        Initialize experiment configuration.

        Parameters:
            config (dict): Experiment configuration dictionary.
            base_dir (str): Base directory for saving experiment results.
        """
        self.data = config.get('data', 'Original')
        self.batch_size = config.get('batch_size', 16)
        self.backbone = config.get('backbone', 'ResNet50')
        self.batch_norm = config.get('batch_norm', True)
        self.loss = config.get('loss', losses.unified_focal_loss)
        self.lr = config.get('lr', 1e-4)
        self.epochs = config.get('epochs', 100)
        self.callbacks_config = config.get('callbacks', [])
        self.base_dir = base_dir

        # Create experiment directory dynamically
        self.dir_ = os.path.join(
            base_dir,
            self.data,
            'experiments',
            f'exp_batch_size_{self.batch_size}_{self.backbone}_{self.loss.__name__}_lr_{self.lr}_epochs_{self.epochs}'
        )
        self._create_directories()

        # Configure callbacks
        self.callbacks = self._configure_callbacks()

        # Initialize attributes for model and training history
        self.model = None
        self.history = None

    def _create_directories(self):
        """
        Create necessary directories for the experiment.
        """
        os.makedirs(self.dir_, exist_ok=True)
        os.makedirs(os.path.join(self.dir_, 'idrid'), exist_ok=True)
        os.makedirs(os.path.join(self.dir_, 'e_ophtha'), exist_ok=True)

    def _configure_callbacks(self):
        """
        Configure callbacks for training.

        Returns:
            list: List of TensorFlow callbacks.
        """
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.dir_, 'model.keras'),
                monitor='val_dice_coef',
                verbose=1,
                save_best_only=True,
                mode='max'
            )
        ]

        if 'early_stopping' in self.callbacks_config:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    mode='auto',
                    verbose=1,
                    patience=5
                )
            )

        if 'reduce_lr_on_plateau' in self.callbacks_config:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    patience=3,
                    verbose=1,
                    factor=0.5,
                    min_lr=1e-7
                )
            )

        return callbacks

    def run(self, X_train, Y_train, val_dataset):
        """
        Run the experiment: build, train, and save the model.

        Parameters:
            X_train (np.ndarray): Training data.
            Y_train (np.ndarray): Training labels.
            val_dataset (tuple): Validation dataset (X_val, Y_val).
        """
        self.model = models.att_unet_2d(
            (256, 256, 3), filter_num=[64, 128, 256, 512, 1024],
            n_labels=1, stack_num_down=2, stack_num_up=2,
            activation='ReLU', atten_activation='ReLU', attention='add',
            output_activation='Sigmoid', batch_norm=self.batch_norm,
            pool=False, unpool=False, backbone=self.backbone,
            weights='imagenet', freeze_backbone=True, freeze_batch_norm=True,
            name='attunet'
        )

        self.model.compile(
            loss=self.loss,
            optimizer=Adam(learning_rate=self.lr),
            metrics=['accuracy', losses.dice_coef]
        )

        start_time = datetime.now()

        self.history = self.model.fit(
            X_train, Y_train,
            verbose=1,
            batch_size=self.batch_size,
            validation_data=val_dataset,
            shuffle=False,
            epochs=self.epochs,
            callbacks=self.callbacks
        )

        execution_time = datetime.now() - start_time
        print(f"Attention UNet execution time: {execution_time}")

        # Save training history
        history_path = os.path.join(self.dir_, 'history.pkl')
        with open(history_path, 'wb') as history_file:
            pickle.dump(self.history.history, history_file)

    def plot_loss_acc(self):
        """
        Plot training and validation loss and accuracy.
        """
        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['loss'], 'y', label='Training Loss')
        plt.plot(epochs, history['val_loss'], 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['dice_coef'], 'y', label='Training Dice Coefficient')
        plt.plot(epochs, history['val_dice_coef'], 'r', label='Validation Dice Coefficient')
        plt.title('Training and Validation Dice Coefficient')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Coefficient')
        plt.legend()

        plt.show()

    def plot_predictions(self, X_test, Y_test, data_name):
        """
        Plot predictions on the test set.

        Parameters:
            X_test (np.ndarray): Test images.
            Y_test (np.ndarray): Test labels.
            data_name (str): Dataset name ('idrid' or 'e_ophtha').
        """
        print(f'Printing prediction masks for {data_name} data...')
        for idx in range(X_test.shape[0]):
            test_img = X_test[idx]
            ground_truth = Y_test[idx]
            test_img_input = np.expand_dims(test_img, axis=0)
            prediction = (self.model.predict(test_img_input)[0, :, :, 0] > 0.5).astype(np.uint8)

            plt.figure(figsize=(16, 8))
            plt.subplot(1, 3, 1)
            plt.title('Test Image')
            plt.imshow(test_img, cmap='gray')

            plt.subplot(1, 3, 2)
            plt.title('Ground Truth')
            plt.imshow(ground_truth, cmap='gray')

            plt.subplot(1, 3, 3)
            plt.title('Prediction')
            plt.imshow(prediction, cmap='gray')

            plt.show()

    def calculate_metrics(self, X_test, Y_test, data_name):
        """
        Calculate IoU and classification report for the test set.

        Parameters:
            X_test (np.ndarray): Test images.
            Y_test (np.ndarray): Test labels.
            data_name (str): Dataset name ('idrid' or 'e_ophtha').
        """
        iou_values = []
        flat_ground_truth = Y_test.ravel()
        flat_predictions = []

        for idx in range(X_test.shape[0]):
            test_img_input = np.expand_dims(X_test[idx], axis=0)
            prediction = (self.model.predict(test_img_input)[0, :, :, 0] > 0.5).astype(np.uint8)
            flat_predictions.extend(prediction.ravel())

            iou = MeanIoU(num_classes=2)
            iou.update_state(Y_test[idx], prediction)
            iou_values.append(iou.result().numpy())

        # Save IoU results
        iou_df = pd.DataFrame(iou_values, columns=['IoU'])
        iou_path = os.path.join(self.dir_, data_name, 'iou.csv')
        iou_df.to_csv(iou_path, index=False)

        # Save classification report
        report = classification_report(flat_ground_truth, flat_predictions)
        report_path = os.path.join(self.dir_, data_name, 'classification_report.txt')
        with open(report_path, 'w') as file:
            file.write(report)

        print(report)
