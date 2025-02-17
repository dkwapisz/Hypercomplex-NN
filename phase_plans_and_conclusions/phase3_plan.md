## Phase 3 - Further color space transformation analysis

The aim of this phase is to further test the transformation of the colour space relative to no transformation, where the
4th channel is the zero channel. This phase is a re-test of the models with the Klein4 algebra and CMYK colour space
discarded, but with the zero channels retained as the 4th channel of the image (except for HSV, where transformation was better than zero channel). 
Only HyperComplex models were tested in this phase. This phase is closely linked to phase 4, where transformations have already been tested.

### Phase parameters

- Image size: (100, 100)
- Dataset: 8 classes, 1200 images per class
- Dataset URL - [Blood Cells](https://www.kaggle.com/datasets/bzhbzh35/peripheral-blood-cell)
- Algebras: Quaternions, Cl20, Coquaternions, Cl11, Bicomplex, Tessarines
- Color spaces: RGB, HSV, YUV, YIQ

| Run number | Proportion (%) | Training set (per class) | Validation set (per class) | Test set (per class) | Training set (all) | Validation set (all) | Test set (all) |
|------------|----------------|--------------------------|----------------------------|----------------------|--------------------|----------------------|----------------|
| Run1       | **1/1/98**     | 12                       | 12                         | 1176                 | 96                 | 96                   | 9408           |
| Run2       | **3/3/94**     | 36                       | 36                         | 1128                 | 288                | 288                  | 9024           |
| Run3       | **5/5/90**     | 60                       | 60                         | 1080                 | 480                | 480                  | 8640           |
| Run4       | **10/10/80**   | 120                      | 120                        | 960                  | 960                | 960                  | 7680           |
| Run5       | **15/15/70**   | 180                      | 180                        | 840                  | 1440               | 1440                 | 6720           |
| Run6       | **20/20/60**   | 240                      | 240                        | 720                  | 1920               | 1920                 | 5760           |
| Run7       | **40/20/40**   | 480                      | 240                        | 480                  | 3840               | 1920                 | 3840           |
| Run8       | **50/20/30**   | 600                      | 240                        | 360                  | 4800               | 1920                 | 2880           |
| Run9       | **60/20/20**   | 720                      | 240                        | 240                  | 5760               | 1920                 | 1920           |
| Run10      | **80/10/10**   | 960                      | 120                        | 120                  | 7680               | 960                  | 960            |

### Color transformations

#### RGB

- CNN: The image is kept in its original RGB format.
- HCNN: A fourth channel filled with zeros is added to match hypercomplex representation.

#### HSV

- CNN: The RGB image is converted to HSV.
- HCNN: The channels are split into:
    - **V** (Value)
    - **S * cos(θ)** (where θ = H * 2π)
    - **S * sin(θ)**
    - **V * cos(θ)**

#### YUV

- CNN: The RGB image is converted to YUV.
- HCNN: A fourth channel filled with zeros is added to match hypercomplex representation.

#### YIQ

- CNN: The RGB image is converted to YIQ.
- HCNN: A fourth channel filled with zeros is added to match hypercomplex representation.

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