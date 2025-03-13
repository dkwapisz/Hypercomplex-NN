| model                          | mean_average_accuracy |
|:-------------------------------|----------------------:|
| HyperComplex-YUV-Bicomplex     |              0.849959 |
| HyperComplex-YUV-Quaternions   |              0.846124 |
| CNN-YUV                        |              0.836162 |
| HyperComplex-RGB-Cl20          |              0.832738 |
| HyperComplex-RGB-Cl11          |              0.831507 |
| HyperComplex-YUV-Coquaternions |              0.825826 |
| HyperComplex-YUV-Tessarines    |              0.814843 |
| CNN-RGB                        |              0.793581 |

For further analysis and hyperparameter optimization, the two best hypercomplex neural networks and the CNN-RGB neural
network will be selected as the most classical approach to image classification. These are the
HyperComplex-YUV-Bicomplex and HyperComplex-YUV-Quaternions models, which have the highest mean average accuracy and
mean average F1-Score. The CNN-RGB model was the worst performing model in both metrics, but it will be included in the
analysis to compare the performance of the hypercomplex models with the most classical approach.