def calculateParamsHyperConv(filters, kernel_size, input_size, layer_before_flatten_shape, num_classes):
    color_channel_shape = input_size[2]
    flatten_shape = layer_before_flatten_shape[0] * layer_before_flatten_shape[1] * layer_before_flatten_shape[2]
    params_sum = 0

    for i in range(len(filters)):
        last_filters_value = 1 if i == 0 else filters[i-1]
        params = (kernel_size[i][0] * kernel_size[i][1] * last_filters_value + 1) * filters[i] * color_channel_shape
        params_sum += params
        print(f"HyperConv2D_{i+1}: {params}")

    dense_params = num_classes * (flatten_shape + 1)
    params_sum += dense_params

    print(f"Dense: {dense_params}")
    print(f"Total params: {params_sum}")

# CNN - 931432

#filters = (8, 16, 32, 64) # - params/4 -> 228 608
#filters = (11, 22, 45, 90) # - params/2 -> 450 420
#filters = (16, 32, 64, 128) # - params -> 912 896
filters = (4, 8, 16)

layer_before_flatten_shape = (10, 10, 64)
kernel_size = [(3, 3), (3, 3), (3, 3)]
input_size = (100, 100, 4)
num_classes = 8

calculateParamsHyperConv(filters, kernel_size, input_size, layer_before_flatten_shape, num_classes)