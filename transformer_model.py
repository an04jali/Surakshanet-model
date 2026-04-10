import torch
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
import numpy as np

# Load pretrained ViT
weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_action(frames):
    imgs = []

    # Take few frames (reduce load)
    selected_frames = frames[::2]  # take every 2nd frame

    for frame in selected_frames:
        img = transform(frame)
        imgs.append(img)

    imgs = torch.stack(imgs)  # shape: (N, 3, 224, 224)

    with torch.no_grad():
        outputs = model(imgs)  # (N, 1000)

    # Average prediction across frames
    avg_output = torch.mean(outputs, dim=0)

    score = torch.max(avg_output).item()

    # Dummy but stable logic
    if score > 0.6:
        return "SUSPICIOUS", round(score * 100, 2)
    else:
        return "NORMAL", round(score * 100, 2)