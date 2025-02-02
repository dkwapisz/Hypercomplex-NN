import os

from HypercomplexKeras import Algebra

from data_processing.ColorSpaceConverter import create_dataset_tf
from models import ModelDefinition
from utils.GPU_Helper import set_gpu_device

set_gpu_device(0)

color_spaces = ["RGB", "HSV", "YUV", "YIQ", "CMYK"]

img_size = (128, 128)
batch_size = 64
num_classes = len(os.listdir("datasets/Lymphoma/train"))
train_path = "datasets/Lymphoma/train"
val_path = "datasets/Lymphoma/val"
test_path = "datasets/Lymphoma/test"

algebras = {
    "Quaternions": Algebra.Quaternions,
    "Klein4": Algebra.Klein4,
    "Cl20": Algebra.Cl20,
    "Coquaternions": Algebra.Coquaternions,
    "Cl11": Algebra.Cl11,
    "Bicomplex": Algebra.Bicomplex,
    "Tessarines": Algebra.Tessarines
}


hypercomplex = True
for color_space in color_spaces:
    for algebra_name, algebra in algebras.items():
        input_shape = img_size + (4,) if (color_space == "CMYK" or hypercomplex) else img_size + (3,)

        train_dataset = create_dataset_tf(train_path, img_size, batch_size, color_space, hypercomplex)
        val_dataset = create_dataset_tf(val_path, img_size, batch_size, color_space, hypercomplex)
        test_dataset = create_dataset_tf(test_path, img_size, batch_size, color_space, hypercomplex)

        assert train_dataset is not None, f"Train dataset creation failed for color space {color_space} and algebra {algebra_name}"
        assert val_dataset is not None, f"Validation dataset creation failed for color space {color_space} and algebra {algebra_name}"
        assert test_dataset is not None, f"Test dataset creation failed for color space {color_space} and algebra {algebra_name}"

        for image, label in train_dataset.take(1):
            assert image.shape[1:] == input_shape, f"Train dataset image shape mismatch for color space {color_space} and algebra {algebra_name}"
        for image, label in val_dataset.take(1):
            assert image.shape[1:] == input_shape, f"Validation dataset image shape mismatch for color space {color_space} and algebra {algebra_name}"
        for image, label in test_dataset.take(1):
            assert image.shape[1:] == input_shape, f"Test dataset image shape mismatch for color space {color_space} and algebra {algebra_name}"

        # Expect no errors
        model = ModelDefinition.get_hypercomplex_cnn_model(input_shape, num_classes, algebra)

        history = model.fit(train_dataset, validation_data=val_dataset, epochs=3, verbose=0)
        assert history.history['accuracy'][-1] > 0, f"Training failed for color space {color_space}"

        eval_res = model.evaluate(test_dataset, batch_size=10)
        assert eval_res[1] > 0, f"Accuracy evaluation failed for color space {color_space} and algebra {algebra_name}"

        print(f"--------- | Case HyperComplex: {hypercomplex} | {color_space} | {str(algebra_name)} PASSED | ---------")


hypercomplex = False
for color_space in color_spaces:
    input_shape = img_size + (4,) if (color_space == "CMYK" or hypercomplex) else img_size + (3,)

    train_dataset = create_dataset_tf(train_path, img_size, batch_size, color_space, hypercomplex)
    val_dataset = create_dataset_tf(val_path, img_size, batch_size, color_space, hypercomplex)
    test_dataset = create_dataset_tf(test_path, img_size, batch_size, color_space, hypercomplex)

    assert train_dataset is not None, f"Train dataset creation failed for color space {color_space}"
    assert val_dataset is not None, f"Validation dataset creation failed for color space {color_space}"
    assert test_dataset is not None, f"Test dataset creation failed for color space {color_space}"

    for image, label in train_dataset.take(1):
        assert image.shape[1:] == input_shape, f"Train dataset image shape mismatch for color space {color_space}"
    for image, label in val_dataset.take(1):
        assert image.shape[
               1:] == input_shape, f"Validation dataset image shape mismatch for color space {color_space}"
    for image, label in test_dataset.take(1):
        assert image.shape[1:] == input_shape, f"Test dataset image shape mismatch for color space {color_space}"

    # Expect no errors
    model = ModelDefinition.get_basic_cnn_model(input_shape, num_classes)

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=3, verbose=0)
    assert history.history['accuracy'][-1] > 0, f"Training failed for color space {color_space}"

    eval_res = model.evaluate(test_dataset, batch_size=10)
    assert eval_res[1] > 0, f"Accuracy evaluation failed for color space {color_space}"

    print(f"--------- | Case HyperComplex: {hypercomplex} | {color_space} PASSED | ---------")

