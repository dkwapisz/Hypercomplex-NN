import os
import sys

from HypercomplexKeras import Algebra
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
dataset_path = "datasets/Lymphoma"
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "train")

color_space = "CMYK"
img_size = (128, 128)
batch_size = 64
hypercomplex = True
input_shape = img_size + (4,) if (color_space == "CMYK" or hypercomplex) else img_size + (3,)
num_classes = len(os.listdir(train_path))
algebras = [Algebra.Quaternions, Algebra.Klein4, Algebra.Cl20, Algebra.Coquaternions, Algebra.Cl11, Algebra.Bicomplex,
            Algebra.Tessarines]


# ------------------- Dataset loading -------------------
train_dataset = create_dataset_tf(train_path, img_size, batch_size, color_space, hypercomplex)
val_dataset = create_dataset_tf(val_path, img_size, batch_size, color_space, hypercomplex)
test_dataset = create_dataset_tf(test_path, img_size, batch_size, color_space, hypercomplex)

# ------------------- Basic CNN Model -------------------
# model = ModelDefinition.get_basic_cnn_model(input_shape, num_classes)
model = ModelDefinition.get_hypercomplex_cnn_model(input_shape, num_classes, Algebra.Quaternions)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=100)

# plt.plot(history.history['accuracy'])
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.show()

test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.4f}")
