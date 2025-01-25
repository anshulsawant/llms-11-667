import torch
import io


def determine_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def estimate_model_disk_size(model: torch.nn.Module) -> int:
    with io.BytesIO() as byte_stream:
        torch.save(model.state_dict(), byte_stream)
        return byte_stream.tell()


# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def enable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
