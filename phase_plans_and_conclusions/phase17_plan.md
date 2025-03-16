The plan is to use following models from phase15_models.md:

1. Run1-3:
    - CNN-RGB -> CNN-RGB
    - CNN-YUV -> CNN-YUV
    - HyperComplex-YUV-Quaternions -> CNN-YUV architecture with HyperConv2D instead of Conv2D and filters = filters/4
    - HyperComplex-YUV-Bicomplex -> CNN-YUV architecture with HyperConv2D instead of Conv2D and filters = filters/4

All other phase parameters are the same as in phase15_plan.md.