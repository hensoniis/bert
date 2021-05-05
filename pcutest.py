import torch
#torch.cuda.current_device()
#torch.cuda._initialized = True
#torch.cuda._lazy_init() 
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())
print(torch.cuda.current_device())

