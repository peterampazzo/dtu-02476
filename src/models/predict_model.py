import torch
from conv_nn import ConvNet


def predict():
    print("Evaluating")

    model = ConvNet(1024, 512)
    model.load_state_dict(torch.load("models/trained_model.pt"))
    test_data = torch.load("data/processed/test.pt")
    test_set = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_set):

            print(f"Evaluating Batch {i}/{len(test_data)//64}")

            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

    accuracy = torch.mean(equals.type(torch.FloatTensor))

    print(f"Accuracy: {accuracy.item()*100}%")


if __name__ == "__main__":
    predict()
