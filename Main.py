from config.PropertiesResolver import PropertiesResolver
from utils.HealthcheckGPU import check_gpu_health

check_gpu_health()

properties = PropertiesResolver("app.properties")