import torch
class DeviceAwareDataLoader:
    def __init__(self, data_dict, device='cpu'):
        self.data = data_dict
        self.device = torch.device(device)
        
    def __getitem__(self, key):
        val = self.data[key]
        return val.to(self.device) if isinstance(val, torch.Tensor) else val