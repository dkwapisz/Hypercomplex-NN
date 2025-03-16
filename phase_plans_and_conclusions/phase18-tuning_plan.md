## Phase 15 - Tuning models

Tuning the models to improve their performance.

### Phase parameters

- Image size: (100, 100)
- Dataset: 8 classes, 1200 images per class
- Dataset URL - [Blood Cells](https://www.kaggle.com/datasets/bzhbzh35/peripheral-blood-cell)
- Types: CNN, HCNN
- Algebras: Quaternions, Bicomplex
- Color spaces: RGB, YUV
- Models used: CNN-RGB, CNN-YUV, Quaternions-YUV, Bicomplex-YUV

| Run number | Proportion (%) | Training set (per class) | Validation set (per class) | Test set (per class) | Training set (all) | Validation set (all) | Test set (all) |
|------------|----------------|--------------------------|----------------------------|----------------------|--------------------|----------------------|----------------|
| Run1       | **30/20/50**   | 360                      | 240                        | 600                  | 2880               | 1920                 | 4800           |

### Color transformations

#### RGB

- **CNN**: The image remains in its original RGB format.
- **HCNN**: The RGB channels are transformed into 4 dimensions:
    - **log(1 + Magnitude)**, where Magnitude = sqrt(R² + G² + B²)
    - **Phase RG** = atan2(G, R)
    - **Phase RB** = atan2(B, R)
    - **Magnitude * cos(Phase RG + Phase RB)**

#### YUV

- **CNN/HCNN**: The U and V channels are transformed into polar coordinates:
    - **Y (Luminance)**
    - **Magnitude * cos(2θ)**, where Magnitude = sqrt(U² + V²) and θ = atan2(V, U)
    - **Magnitude * sin(2θ)**
    - **exp(-Magnitude)**

### Models

#### Convolutional Neural Network

```python
def objective_cnn(trial, train_dataset, val_dataset, input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))

    num_layers = trial.suggest_int("num_layers", 2, 6)
    for i in range(num_layers):
        filters = trial.suggest_categorical(f"filters_{i}", [8, 16, 32, 64, 128, 256])
        kernel_size_num = trial.suggest_categorical(f"kernels_size_{i}", [3, 5, 7])
        model.add(Conv2D(filters, (kernel_size_num, kernel_size_num), padding='same', activation='relu'))
        if trial.suggest_categorical(f"pool_{i}", [True, False]):
            strides = trial.suggest_categorical(f"strides_{i}", [None, 1, 2])
            model.add(MaxPooling2D(strides=strides))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=["accuracy"])

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=200, verbose=0,
                        callbacks=[early_stopping_tuning])

    return history.history['val_accuracy'][-1]
```

#### HyperComplex Convolutional Neural Network

```python
def objective_hcnn(trial, train_dataset, val_dataset, input_shape, num_classes, algebra):
    model = Sequential()
    model.add(Input(shape=input_shape))

    num_layers = trial.suggest_int("num_layers", 2, 6)
    for i in range(num_layers):
        filters = trial.suggest_categorical(f"filters_{i}", [2, 4, 8, 16, 32, 64])
        kernel_size_num = trial.suggest_categorical(f"kernels_size_{i}", [3, 5, 7])
        model.add(HyperConv2D(filters, (kernel_size_num, kernel_size_num), padding='SAME', activation='relu',
                              algebra=algebra))
        if trial.suggest_categorical(f"pool_{i}", [True, False]):
            strides = trial.suggest_categorical(f"strides_{i}", [None, 1, 2])
            model.add(MaxPooling2D(strides=strides))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=["accuracy"])

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=200, verbose=0,
                        callbacks=[early_stopping_tuning])

    return history.history['val_accuracy'][-1]
```