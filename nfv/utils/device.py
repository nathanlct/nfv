import subprocess

import torch


def get_mps_info():
    # Run system_profiler command to get display information
    gpu_info = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True)

    # Extract lines that mention the GPU name and core count
    lines = gpu_info.stdout.splitlines()
    gpu_name = ""
    core_count = ""

    for line in lines:
        if "Chipset Model" in line:
            gpu_name = line.split(": ")[1]  # Extract the GPU name
        elif "Total Number of Cores" in line:
            core_count = line.split(": ")[1]  # Extract the number of cores

    # Format  the result
    return f"{gpu_name} {core_count} cores"


def get_gpu_memory():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], stdout=subprocess.PIPE, text=True, check=True
        )
        memory_usage = [int(x) for x in result.stdout.strip().split("\n")]
        return memory_usage
    except FileNotFoundError:
        print("nvidia-smi not found. Please ensure it's installed and available in your PATH.")
        return None


def get_device(verbose=True):
    if torch.cuda.is_available():
        memory_usage = get_gpu_memory()
        if memory_usage is not None:
            # Filter out Nvidia DGX GPUs
            excluded_gpus = []
            for i in range(torch.cuda.device_count()):
                if "DGX" in torch.cuda.get_device_name(i):
                    excluded_gpus.append(i)
            # Filter memory usage for valid GPUs
            valid_gpus = [i for i in range(torch.cuda.device_count()) if i not in excluded_gpus]
            if valid_gpus:
                selected_gpu = min(valid_gpus, key=lambda i: memory_usage[i])  # Select GPU with least memory used
                device = torch.device(f"cuda:{selected_gpu}")
                if verbose:
                    print(
                        f"Using GPU : {torch.cuda.get_device_name(selected_gpu)} (GPU {selected_gpu}) "
                        f"with {memory_usage[selected_gpu]:.2f} MiB used."
                    )
            else:
                print("No suitable GPU found. Defaulting to CPU.")
                device = torch.device("cpu")
            if verbose:
                print(f"Using GPU : {torch.cuda.get_device_name(selected_gpu)} (GPU {selected_gpu}) with {memory_usage[selected_gpu]} MiB used.")
        else:
            print("Could not determine GPU memory usage. Defaulting to CUDA device 0.")
            device = torch.device("cuda:0")

        # device = torch.device('cuda')
        # if verbose:
        #     print("Using GPU :", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print(f"Using GPU : {get_mps_info()}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using cpu device")

    return device
