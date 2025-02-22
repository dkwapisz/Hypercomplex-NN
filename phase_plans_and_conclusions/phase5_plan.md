## Phase 5 - Final comparison of color transformations between CNN and HCNN

The aim of this phase is to make a final comparison between CNNs and HCNNs with the number of layers and filters that
were used in the previous phases. The colour space transformations that were the best in the previous 4 phases were
used. The split proportions of the dataset have been slightly altered in order to obtain more diverse results.

### Phase parameters

- Image size: (100, 100)
- Dataset: 8 classes, 1200 images per class
- Dataset URL - [Blood Cells](https://www.kaggle.com/datasets/bzhbzh35/peripheral-blood-cell)
- Types: CNN, HCNN
- Algebras: Quaternions, Cl20, Coquaternions, Cl11, Bicomplex, Tessarines
- Color spaces: RGB, HSV, YUV, YIQ

| Run number | Proportion (%) | Training set (per class) | Validation set (per class) | Test set (per class) | Training set (all) | Validation set (all) | Test set (all) |
|------------|----------------|--------------------------|----------------------------|----------------------|--------------------|----------------------|----------------|
| Run1       | **1/20/79**    | 12                       | 240                        | 948                  | 96                 | 1920                 | 7584           |
| Run2       | **3/20/77**    | 36                       | 240                        | 924                  | 288                | 1920                 | 7392           |
| Run3       | **5/20/75**    | 60                       | 240                        | 900                  | 480                | 1920                 | 7200           |
| Run4       | **10/20/70**   | 120                      | 240                        | 840                  | 960                | 1920                 | 6720           |
| Run5       | **20/20/60**   | 240                      | 240                        | 720                  | 1920               | 1920                 | 5760           |
| Run6       | **30/20/50**   | 360                      | 240                        | 600                  | 2880               | 1920                 | 4800           |
| Run7       | **40/20/40**   | 480                      | 240                        | 480                  | 3840               | 1920                 | 3840           |
| Run8       | **50/20/30**   | 600                      | 240                        | 360                  | 4800               | 1920                 | 2880           |
| Run9       | **60/20/20**   | 720                      | 240                        | 240                  | 5760               | 1920                 | 1920           |
| Run10      | **70/20/10**   | 840                      | 240                        | 120                  | 6720               | 1920                 | 960            |

### Color transformations

#### RGB

- CNN: The image remains in its original RGB format.
- **HCNN**: The RGB channels are transformed into 4 dimensions:
    - **log(1 + Magnitude)**, where Magnitude = sqrt(R² + G² + B²)
    - **Phase RG** = atan2(G, R)
    - **Phase RB** = atan2(B, R)
    - **Magnitude * cos(Phase RG + Phase RB)**

#### HSV

- CNN: The image remains in its original HSV format.
- **HCNN**: The HSV channels are transformed into 4 dimensions:
    - **V**
    - **exp(S) * cos(θ)**, where θ = H * 2π
    - **exp(S) * sin(θ)**
    - **sin(V * π)**

#### YUV

- CNN: The image remains in its original YUV format.
- **HCNN**: The U and V channels are transformed into polar coordinates:
    - **Y (Luminance)**
    - **Magnitude * cos(2θ)**, where Magnitude = sqrt(U² + V²) and θ = atan2(V, U)
    - **Magnitude * sin(2θ)**
    - **exp(-Magnitude)**

#### YIQ

- CNN: The image remains in its original YIQ format.
- **HCNN**: The I and Q channels are transformed into polar coordinates:
    - **Y (Luminance)**
    - **Magnitude * cos(2θ)**, where Magnitude = sqrt(I² + Q²) and θ = atan2(Q, I)
    - **Magnitude * sin(2θ)**
    - **exp(-Magnitude)**

### Models

#### Convolutional Neural Network

```python
def build_model(input_shape, num_classes, metrics):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)

    return model
```

#### HyperComplex Convolutional Neural Network

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

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)

    return model
```