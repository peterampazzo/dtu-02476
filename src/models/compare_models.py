from datetime import datetime
from typing import List

import torch
from torch import nn, optim

from src.models.conv_nn import ConvNet
from src.models.conv_nn_quantized import ConvNet_quantized
from src.models.kornia_trans import transform

# create a model instance
model = ConvNet(1024, 512)
model_fp32 = ConvNet_quantized(1024, 512)
scripted_model = torch.jit.script(model)

# attach a global qconfig, which contains information about what kind
# of observers to attach. Use 'fbgemm' for server inference and
# 'qnnpack' for mobile inference. Other quantization configurations such
# as selecting symmetric or assymetric quantization and MinMax or L2Norm
# calibration techniques can be specified here.
model_fp32.qconfig = torch.quantization.get_default_qconfig("fbgemm")

# Fuse the activations to preceding layers, where applicable.
# This needs to be done manually depending on the model architecture.
# Common fusions include `conv + relu` and `conv + batchnorm + relu`
qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [["conv1", "relu"]])

# Prepare the model for static quantization. This inserts observers in
# the model that will observe activation tensors during calibration.
model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)


train_data = torch.load("data/processed/train.pt")
train_set = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

train_loss: List[float] = []


def run_model():
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    with open("reports/quantization/training_not_quantized.txt", "w") as f:
        start_time = datetime.now()
        for e in range(6):
            running_loss = 0
            print("We are starting the", e, "epochs")
            for i, (images, labels) in enumerate(train_set):
                optimizer.zero_grad()

                images = transform(images)  # kornia transformations

                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i > 5:
                    break

            result = f"Finished epoch: {e+1} - Training loss: {running_loss/len(train_set):5f} \n"
            f.write(result)
        end_time = datetime.now()
        time = "Execution time : {}".format(end_time - start_time)
        f.write(time)
        f.close()


def run_model_quantized():
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    with open("reports/quantization/training_quantized.txt", "w") as f:
        start_time = datetime.now()
        for e in range(6):
            running_loss = 0
            print("We are starting the", e, "epochs")
            for i, (images, labels) in enumerate(train_set):
                optimizer.zero_grad()

                images = transform(images)  # kornia transformations

                log_ps = model_fp32_prepared(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i > 5:
                    break

            result = f"Finished epoch: {e+1} - Training loss: {running_loss/len(train_set):5f} \n"
            f.write(result)
        end_time = datetime.now()
        time = "Execution time : {}".format(end_time - start_time)
        f.write(time)
        f.close()


def run_model_scripted():
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    with open("reports/quantization/training_scripted.txt", "w") as f:
        start_time = datetime.now()
        for e in range(6):
            running_loss = 0
            print("We are starting the", e, "epochs")
            for i, (images, labels) in enumerate(train_set):
                optimizer.zero_grad()

                images = transform(images)  # kornia transformations

                log_ps = scripted_model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i > 5:
                    break

            result = f"Finished epoch: {e+1} - Training loss: {running_loss/len(train_set):5f} \n"
            f.write(result)
        end_time = datetime.now()
        time = "Execution time : {}".format(end_time - start_time)
        f.write(time)
        f.close()


if __name__ == "__main__":
    run_model()
    run_model_quantized()
    run_model_scripted()
