# Run 1

## CNN-RGB

```python
def build_model(input_shape, num_classes, metrics):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(strides=2))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(strides=2))

    model.add(Conv2D(128, (7, 7), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)

    return model
```

## CNN-YUV

```python
def build_model(input_shape, num_classes, metrics):
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(strides=2))
    
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(256, (7, 7), padding='same', activation='relu'))
    model.add(MaxPooling2D(strides=2))
    
    model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)
    
    return model
```

## HyperComplex-YUV-Bicomplex

```python
def build_model(input_shape, num_classes, metrics, algebra):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(HyperConv2D(4, (7, 7), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D(strides=2))

    model.add(HyperConv2D(64, (5, 5), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(HyperConv2D(32, (7, 7), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(HyperConv2D(4, (7, 7), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(HyperConv2D(8, (7, 7), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(HyperConv2D(32, (7, 7), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)

    return model
```

## HyperComplex-YUV-Quaternions

```python
def build_model(input_shape, num_classes, metrics, algebra):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(HyperConv2D(32, (3, 3), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(HyperConv2D(32, (7, 7), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D(strides=2))

    model.add(HyperConv2D(64, (3, 3), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)

    return model
```

# Run 2

## CNN-RGB

```python
def build_model(input_shape, num_classes, metrics):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))

    model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)

    return model
```

## CNN-YUV

```python
def build_model(input_shape, num_classes, metrics):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(strides=2))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(strides=2))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(strides=2))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)

    return model
```

## HyperComplex-YUV-Bicomplex

```python
def build_model(input_shape, num_classes, metrics, algebra):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(HyperConv2D(64, (3, 3), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(HyperConv2D(64, (5, 5), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(HyperConv2D(64, (5, 5), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(HyperConv2D(64, (7, 7), padding='SAME', activation='relu', algebra=algebra))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)

    return model
```

## HyperComplex-YUV-Quaternions

```python
def build_model(input_shape, num_classes, metrics, algebra):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(HyperConv2D(32, (5, 5), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D(strides=2))

    model.add(HyperConv2D(8, (3, 3), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(HyperConv2D(4, (3, 3), padding='SAME', activation='relu', algebra=algebra))

    model.add(HyperConv2D(64, (3, 3), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(HyperConv2D(16, (3, 3), padding='SAME', activation='relu', algebra=algebra))

    model.add(HyperConv2D(64, (5, 5), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D(strides=2))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)

    return model
```

# Run 3

## CNN-RGB

```python
def build_model(input_shape, num_classes, metrics):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (7, 7), padding='same', activation='relu'))
    model.add(MaxPooling2D(strides=2))

    model.add(Conv2D(8, (5, 5), padding='same', activation='relu'))

    model.add(Conv2D(16, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

    model.add(Conv2D(16, (7, 7), padding='same', activation='relu'))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)

    return model
```

## CNN-YUV

```python
def build_model(input_shape, num_classes, metrics):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)

    return model
```

## HyperComplex-YUV-Bicomplex

```python
def build_model(input_shape, num_classes, metrics, algebra):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(HyperConv2D(4, (7, 7), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(HyperConv2D(32, (7, 7), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D(strides=2))

    model.add(HyperConv2D(16, (5, 5), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(HyperConv2D(64, (7, 7), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D(strides=2))

    model.add(HyperConv2D(64, (3, 3), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(HyperConv2D(16, (5, 5), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)

    return model
```

## HyperComplex-YUV-Quaternions

```python
def build_model(input_shape, num_classes, metrics, algebra):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(HyperConv2D(16, (7, 7), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D(strides=2))

    model.add(HyperConv2D(32, (5, 5), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D(strides=2))

    model.add(HyperConv2D(32, (3, 3), padding='SAME', activation='relu', algebra=algebra))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=metrics)

    return model
```
