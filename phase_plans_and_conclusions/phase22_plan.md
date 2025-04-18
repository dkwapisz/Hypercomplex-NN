## Phase 22 - Test permutation of color channels in HyperComplex models

In this phase, the models from the previous phase will be tested with the same architecture, but this time, the order of
the color channels in HyperComplex models will be permuted. The goal is to verify which permutation of color channels
are the best. **Base for permutation is provided in Color transformations section below.**

### Phase parameters

- Image size: (100, 100)
- Dataset: 8 classes, 1200 images per class
- Dataset URL - [Blood Cells](https://www.kaggle.com/datasets/bzhbzh35/peripheral-blood-cell)
- Types: HCNN
- Algebras: Quaternions, Bicomplex
- Color spaces: YUV
- Models used: Quaternions-YUV, Bicomplex-YUV

| Run number | Proportion (%) | Training set (per class) | Validation set (per class) | Test set (per class) | Training set (all) | Validation set (all) | Test set (all) |
|------------|----------------|--------------------------|----------------------------|----------------------|--------------------|----------------------|----------------|
| Run1       | **70/20/10**   | 840                      | 240                        | 120                  | 6720               | 1920                 | 960            |

### Color transformations

#### YUV

All 24 permutations of the color channels will be tested. Base for permutation is [Y, U, V, fourth_channel], where:
- **Y -> Y (Luminance)**
- **U -> Magnitude * cos(2θ)**, where Magnitude = sqrt(U² + V²) and θ = atan2(V, U)
- **V -> Magnitude * sin(2θ)**
- **fourth_channel -> exp(-Magnitude)**

### Models

#### Convolutional Neural Network

## HyperComplex Convolutional Neural Network

```python
def build_model(input_shape, num_classes, metrics, algebra):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(HyperConv2D(8, (3, 3), activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(HyperConv2D(16, (3, 3), activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(HyperConv2D(32, (3, 3), activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(HyperConv2D(64, (5, 5), activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)

    return model
```