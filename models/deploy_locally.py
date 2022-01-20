from src.models.conv_nn import ConvNet
import torch

model = ConvNet(1024,512)

# And if it's already train
# model = model.load_state_dict(torch.load("models/trained_model.pt"))

script_model = torch.jit.script(model)
script_model.save('deployable_model.pt')