## Phase 1 - Data Distribution

Image size: (100, 100)
Dataset: 8 classes, 1200 images per class

| Run number | Proportion (%) | Training set (per class) | Validation set (per class) | Test set (per class) | Training set (all) | Validation set (all) | Test set (all) |
|------------|----------------|--------------------------|----------------------------|----------------------|--------------------|----------------------|----------------|
| Run1       | **1/1/98**     | 12                       | 12                         | 1176                 | 96                 | 96                   | 9408           |
| Run2       | **3/3/94**     | 36                       | 36                         | 1128                 | 288                | 288                  | 9024           |
| Run3       | **5/5/90**     | 60                       | 60                         | 1080                 | 400                | 400                  | 8800           |
| Run4       | **10/10/80**   | 120                      | 120                        | 960                  | 800                | 800                  | 8000           |
| Run5       | **15/15/70**   | 180                      | 180                        | 840                  | 1200               | 1200                 | 6720           |
| Run6       | **20/20/60**   | 240                      | 240                        | 720                  | 1600               | 1600                 | 6400           |
| Run7       | **40/20/40**   | 480                      | 240                        | 480                  | 3200               | 1600                 | 4800           |
| Run8       | **50/20/30**   | 600                      | 240                        | 360                  | 4000               | 1600                 | 3200           |
| Run9       | **60/20/20**   | 720                      | 240                        | 240                  | 4800               | 1600                 | 3200           |
| Run10      | **80/10/10**   | 960                      | 120                        | 120                  | 6400               | 800                  | 800            |

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
- HCNN: The U and V channels are transformed into polar coordinates:
    - **Y (Luminance)**
    - **Magnitude * cos(θ)**
    - **Magnitude * sin(θ)**
    - **Y * cos(θ)**

#### YIQ

- CNN: The RGB image is converted to YIQ.
- HCNN: The I and Q channels are transformed into polar coordinates:
    - **Y (Luminance)**
    - **Magnitude * cos(θ)**
    - **Magnitude * sin(θ)**
    - **Y * cos(θ)**

#### CMYK

- CNN/HCNN: The image is converted to CMYK using:
    - **C = (R - K) / (1 - K)**
    - **M = (G - K) / (1 - K)**
    - **Y = (B - K) / (1 - K)**
    - **K = min(1 - R, 1 - G, 1 - B)**