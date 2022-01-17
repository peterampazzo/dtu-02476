import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    tensorboard_trace_handler)

from src.models.conv_nn import ConvNet


def inference():

    model = ConvNet(1024, 512)
    inputs = torch.randn(5, 3, 224, 224)

    # To see the results use tensorboard --logdir src/profiling
    # or open tensorboard directly in vscode

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=tensorboard_trace_handler("src/profiling"),
    ):
        with record_function("model_forward_pass"):
            model(inputs)


if __name__ == "__main__":
    inference()
