import sys

from config.PropertiesResolver import PropertiesResolver
from utils.GPU_Helper import check_gpu_health, set_gpu_device

if len(sys.argv) != 2:
    print("Usage: python3 Main.py <gpu_index>")
    sys.exit(1)



gpu_index = int(sys.argv[1])
print(gpu_index)
properties = PropertiesResolver("app.properties")

check_gpu_health()
set_gpu_device(gpu_index)

