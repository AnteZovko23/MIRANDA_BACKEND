import torch

# Load the saved model
model = torch.load('./VGG_Face.t7', map_location=torch.device('cpu'))

# Convert the saved model to the current PyTorch version
torch.save(model, 'vgg_face.pth')