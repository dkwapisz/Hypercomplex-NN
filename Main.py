import sys

from keras.src.utils import set_random_seed
from matplotlib import pyplot as plt

from config.PropertiesResolver import PropertiesResolver
from data_processing.ColorSpaceConverter import create_dataset
from models import ModelDefinition
from utils.GPU_Helper import check_gpu_health, set_gpu_device

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


train_data = create_dataset("datasets/Lymphoma/train", (64, 64),
                            color_space="RGB")

val_data = create_dataset("datasets/Lymphoma/val", (64, 64),
                            color_space="RGB")

test_data = create_dataset("datasets/Lymphoma/test", (64, 64),
                            color_space="RGB")

model = ModelDefinition.get_basic_cnn_model()

history = model.fit(train_data, validation_data=val_data, epochs=5, batch_size=128)

plt.plot(history.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()


