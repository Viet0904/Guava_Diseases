import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LambdaCallback
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

# Tạo tên tệp log
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = "CNN_IMG224_BATCH32_EPOCHS10_Adam_{}.log".format(current_time)
logging.basicConfig(filename=log_filename, level=logging.INFO, filemode="a")

# Định nghĩa các thông số cần thiết
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 8
EPOCHS = 10

# Tạo các đường dẫn đến các thư mục chứa dữ liệu
train_dir = "./input/train"
test_dir = "./input/test"
valid_dir = "./input/val"

# Tạo data generators cho tập huấn luyện
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

# Tạo data generators cho tập kiểm thử và validation
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_data = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

valid_data = valid_datagen.flow_from_directory(
    valid_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

# Create a CNN model
model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

# Compile mô hình với các metric mới
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
)


# Tạo một callback để log thông tin sau mỗi epoch
def epoch_end_log(epoch, logs):
    logging.info(f"Epoch {epoch+1}/{EPOCHS}")
    logging.info(
        f"loss: {logs['loss']:.5f} - accuracy: {logs['accuracy']:.5f} - val_loss: {logs['val_loss']:.5f} - val_accuracy: {logs['val_accuracy']:.5f}"
    )


epoch_end_callback = LambdaCallback(on_epoch_end=epoch_end_log)

# Train the model with the new callback
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data,
    verbose=1,
    callbacks=[epoch_end_callback],
)

# Save the model in the native Keras format
model.save("./model.keras")
logging.info("Mô hình đã được lưu sau khi đào tạo")

# Load the model from the correct path
loaded_model = models.load_model("./model.keras")
logging.info("Mô hình đã được tải lại")

# Predictions
predictions = loaded_model.predict(test_data)

# Evaluate the model on test data
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_data)
# Định dạng các giá trị với 5 số thập phân
test_loss_formatted = float("{:.5f}".format(test_loss))
test_accuracy_formatted = float("{:.5f}".format(test_accuracy))
test_precision_formatted = float("{:.5f}".format(test_precision))
test_recall_formatted = float("{:.5f}".format(test_recall))
valid_loss, valid_accuracy, valid_precision, valid_recall = model.evaluate(valid_data)
valid_loss_formatted = float("{:.5f}".format(valid_loss))
valid_accuracy_formatted = float("{:.5f}".format(valid_accuracy))
valid_precision_formatted = float("{:.5f}".format(valid_precision))
valid_recall_formatted = float("{:.5f}".format(valid_recall))


logging.info(f"Test Loss: {test_loss_formatted:.5f}")
logging.info(f"Test Accuracy: {test_accuracy_formatted:.5f}")
logging.info(f"Validation Precision: {test_precision_formatted:.5f}")
logging.info(f"Validation Recall: {test_recall_formatted:.5f}")
logging.info(f"Validation Loss: {valid_loss_formatted:.5f}")
logging.info(f"Validation Accuracy: {valid_accuracy_formatted:.5f}")
logging.info(f"Validation Loss: {valid_precision_formatted:.5f}")
logging.info(f"Validation Accuracy: {valid_recall_formatted:.5f}")

# Đóng tệp log khi hoàn thành
logging.shutdown()

# Sủ dụng CNN - Deep learning
