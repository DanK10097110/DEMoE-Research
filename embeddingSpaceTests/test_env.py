import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torchvision
import torchaudio

print(f"--- 2026 AI Environment Check ---")
print(f"PyTorch Version: {torch.__version__}") # Should be 2.10.0+
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU in use:      {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version:    {torch.version.cuda}")