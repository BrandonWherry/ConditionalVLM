from torchvision import models
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, transforms, Normalize, InterpolationMode
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# model_save_path = './unsafe_models_SH/best_model_masked.pth'
# model_save_path = './unsafe_models_NSFW/best_model_masked.pth'
model_save_path = './unsafe_models_CB/best_model_masked.pth'

# data_path = '/workspace/adv_robustness/CSE/datasets/self_harm_masked/'
# data_path = '/workspace/adv_robustness/CSE/datasets/nsfw_masked/'
data_path = '/workspace/adv_robustness/CSE/datasets/cyberbullying_masked/'

# model_chkp_path = './unsafe_models_SH/best_model.pth'
# model_chkp_path = './unsafe_models_NSFW/best_model.pth'
model_chkp_path = './unsafe_models_CB/best_model.pth'

epochs = 10

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
    
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample, target

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGBA') if 'P' in img.getbands() else img), # Convert Palette images to RGBA
    transforms.Lambda(lambda img: img.convert('RGB')), # Convert RGBA to RGB
    transforms.Resize(232, interpolation=InterpolationMode.BILINEAR), # Resize to 232
    transforms.CenterCrop(224), # Center crop to 224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize
])

# Load all data
train_dataset = CustomImageFolder(root=data_path, transform=transform)

# dataset = CustomImageFolder(root=data_path, transform=transform)

# Determine lengths of splits
# total_size = len(dataset)
# train_size = int(total_size * 0.8)
# valid_size = int(total_size * 0.1)
# test_size = total_size - train_size - valid_size




# # Set the seed for reproducibility
# torch.manual_seed(0)

# # Create the data sets
# train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

# Create data loaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create the model
model = models.resnet50(pretrained=False)
last_linear_in_features = model.fc.in_features
model.fc = nn.Linear(last_linear_in_features, 2)  # Change the output to 2 classes

checkpoint_path = model_chkp_path  # replace with your checkpoint path
checkpoint = torch.load(checkpoint_path)

# Update model's state dictionary
model.load_state_dict(checkpoint)


total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=5e-6)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Validation function
def validate(model, valid_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), torch.nn.functional.one_hot(labels, num_classes=2).float().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(valid_loader)

# Training loop
best_valid_loss = float('inf') # Initialize with infinity
for epoch in range(epochs):
    model.train()
    train_progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{epochs}], Training')
    for data, target in train_progress_bar:
        data, target = data.to(device), target.to(device)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=2).float()

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target_one_hot)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_progress_bar.set_postfix({'Loss': loss.item()})
    
    # Validate after each epoch
#     valid_loss = validate(model, valid_loader, criterion)
#     print(f'Validation Loss: {valid_loss}')

    # Save the model if better validation loss is found
    if loss.item() < best_valid_loss:
        print("Saving best model...")
        torch.save(model.state_dict(), model_save_path)
        best_valid_loss = loss.item()

# torch.save(model.state_dict(), model_save_path)