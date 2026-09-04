import torch
import torch.nn as nn

class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Linear(2, 16)  # input: count + movement

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=2,
            dim_feedforward=32,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return torch.sigmoid(x)


# 🔥 Load model once
model = MiniTransformer()
model.eval()


def predict_risk(count, movement):
    x = torch.tensor([[count, movement]], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        score = model(x).item()

    return int(score * 100)