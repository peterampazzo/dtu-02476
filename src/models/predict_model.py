import torch


def predict():
    print("Evaluating")

    model = torch.load("models/trained_model.pt")
    test = torch.load("data/processed/test.pt")
    test_set = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

    with torch.no_grad():
        for images, labels in test_set:
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

    accuracy = torch.mean(equals.type(torch.FloatTensor))

    print(f"Accuracy: {accuracy.item()*100}%")


if __name__ == "__main__":
    predict()
