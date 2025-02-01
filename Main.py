import os
import sys

from keras.src.utils import set_random_seed

from config.PropertiesResolver import PropertiesResolver
from data_processing.ColorSpaceConverter import create_dataset_tf
from models import ModelDefinition
from utils.GPU_Helper import check_gpu_health, set_gpu_device

# ------------------- Env setup -------------------
set_random_seed(42)

if len(sys.argv) != 2:
    print("Default run, taking 1 GPU")
    gpu_num = 1
else:
    gpu_num = sys.argv[1]

gpu_index = int()
properties = PropertiesResolver("app.properties")

check_gpu_health()
set_gpu_device(gpu_index)


# ------------------- Parameters ------------------- (probably move to properties file if possible)
color_space = "YIQ"
img_size = (128, 128)
batch_size = 32
input_shape = img_size + (4,) if color_space == "CMYK" else img_size + (3,)
num_classes = len(os.listdir("datasets/Lymphoma/train"))


# ------------------- Dataset loading -------------------
train_dataset = create_dataset_tf("datasets/Lymphoma/train", img_size, batch_size, color_space=color_space)
val_dataset = create_dataset_tf("datasets/Lymphoma/val", img_size, batch_size, color_space=color_space)
test_dataset = create_dataset_tf("datasets/Lymphoma/test", img_size, batch_size, color_space=color_space)


# ------------------- Basic CNN Model -------------------
model = ModelDefinition.get_basic_cnn_model(input_shape=input_shape, num_classes=num_classes)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=5)

# plt.plot(history.history['accuracy'])
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.show()

test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.4f}")
