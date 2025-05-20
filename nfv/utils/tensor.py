import numpy as np
import torch


def get_dtype(x):
    if isinstance(x, torch.Tensor):
        return x.dtype
    elif isinstance(x, np.ndarray):
        if x.dtype == np.float64:
            return torch.float64
        elif x.dtype == np.float32:
            return torch.float32
        else:
            raise ValueError(f"Unsupported type {type(x)}")
    elif isinstance(x, float):
        return torch.float64
    else:
        raise ValueError(f"Unsupported type {type(x)}")


def ensure_tensor(func):
    def wrapper(self, x):
        x = torch.as_tensor(x, dtype=get_dtype(x))
        return func(self, x)

    return wrapper


def batch_iterator(*data, batch_size):
    """Yields batches of size `batch_size` from `data`, or the full data if `batch_size` is None."""
    if batch_size is None:
        yield data
    else:
        for i in range(0, len(data[0]), batch_size):
            yield (x[i : i + batch_size] for x in data)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
