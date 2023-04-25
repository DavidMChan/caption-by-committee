from typing import Optional

import torch


def select_device(device: Optional[str] = None) -> Optional[str]:

    if device is None:
        # Check if CUDA is available
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    if isinstance(device, str):
        if device == "cuda":
            # If the user specifies only CUDA, use the GPU with the most free memory
            free_memory = [
                torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                for i in range(torch.cuda.device_count())
            ]
            device = "cuda:" + str(torch.argmax(torch.tensor(free_memory)).item())
        elif device.startswith("cuda:"):
            # If the user specifies a specific GPU, make sure it exists
            if int(device[5:]) >= torch.cuda.device_count():
                raise ValueError(f"Invalid device: {device}")

        return device
