# Step 1: Import necessary libraries and prepare the device (CPU or GPU)
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image normalization (using ImageNet means and std for pre-trained VGG19)
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

# Function to load and preprocess an image
def load_image(img_path, max_size=512, shape=None):
    """Load an image and convert it to a normalized tensor."""
    image = Image.open(img_path).convert('RGB')
    # Resize the image if needed
    if max_size is not None:
        if shape is not None:
            # If shape is specified (width, height), resize to it
            image = image.resize(shape, Image.LANCZOS)
        else:
            # Resize while maintaining aspect ratio so that the longer side is max_size
            w, h = image.size
            if max(w, h) > max_size:
                if w >= h:
                    new_h = int(max_size * h / w)
                    new_w = max_size
                else:
                    new_w = int(max_size * w / h)
                    new_h = max_size
                image = image.resize((new_w, new_h), Image.LANCZOS)
    # Convert image to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image_tensor = transform(image)[:3, :, :].unsqueeze(0)  # ensure [1,3,H,W] tensor
    return image_tensor

# Function to convert a tensor back to a numpy image for display
def tensor_to_image(tensor):
    """Convert a normalized tensor to a numpy image (in [0,1] range)."""
    image = tensor.detach().cpu().clone().squeeze(0)
    image = image.permute(1, 2, 0).numpy()  # CxHxW -> HxWxC
    # De-normalize using mean and std
    image = image * np.array(std) + np.array(mean)
    image = np.clip(image, 0, 1)
    return image

# Load the pre-trained VGG19 network for feature extraction
vgg = models.vgg19(pretrained=True).features  # use VGG19's feature extractor
for param in vgg.parameters():
    param.requires_grad = False  # freeze model parameters
vgg.to(device)
vgg.eval()

# Load content and style images
content_path = "path/to/your/content_image.jpg"  # TODO: replace with your content image path
style_path   = "path/to/your/style_image.jpg"    # TODO: replace with your style image path
content_image = load_image(content_path, max_size=512).to(device)
style_image   = load_image(style_path, max_size=512, shape=(content_image.size(3), content_image.size(2))).to(device)
# Resizing style to content's dimensions for better stylistic consistency

# Step 2: Extract features and define style/content losses
# Layers for content and style representation in VGG19
content_layer = 'conv4_2'
style_layers  = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

# Feature extraction function
def get_features(image, model):
    """Get features from specified layers of the model for the given image."""
    features = {}
    x = image
    conv_count = 0
    block = 1
    for layer in model.children():
        if isinstance(layer, torch.nn.Conv2d):
            conv_count += 1
            x = layer(x)
            layer_name = f'conv{block}_{conv_count}'
            if layer_name in style_layers or layer_name == content_layer:
                features[layer_name] = x
        elif isinstance(layer, torch.nn.ReLU):
            x = layer(x)  # apply ReLU activation
        elif isinstance(layer, torch.nn.MaxPool2d):
            x = layer(x)
            block += 1
            conv_count = 0
        elif isinstance(layer, torch.nn.BatchNorm2d):
            x = layer(x)
    return features

# Gram matrix for style representation
def gram_matrix(tensor):
    """Compute the Gram matrix of a tensor (for style features)."""
    _, n_filters, h, w = tensor.size()
    # Flatten the feature maps
    tensor = tensor.view(n_filters, h * w)
    # Compute Gram matrix as the dot product of feature map matrix with its transpose
    gram = torch.mm(tensor, tensor.t())
    return gram

# Get content and style features
content_features = get_features(content_image, vgg)
style_features   = get_features(style_image, vgg)
# Compute Gram matrices for style features
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}

# Initialize target image as content image clone (requires grad for optimization)
target = content_image.clone().requires_grad_(True).to(device)
# Alternatively, start from random noise:
# target = torch.randn_like(content_image).requires_grad_(True).to(device)

# Weights for content and style loss
content_weight = 1e0  # alpha (e.g., 1)
style_weight   = 1e6  # beta (e.g., 1e6)

# Optimizer to modify the target image
optimizer = torch.optim.Adam([target], lr=0.003)

# Optimization loop
num_steps = 300  # number of iterations (increase for higher quality)
for step in range(1, num_steps+1):
    optimizer.zero_grad()
    # Get current features for target image
    target_features = get_features(target, vgg)
    # Content loss: MSE between target and content features
    content_loss = torch.nn.functional.mse_loss(
        target_features[content_layer], content_features[content_layer]
    )
    # Style loss: MSE between target & style Gram matrices for each layer
    style_loss = 0
    for layer in style_layers:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram  = style_grams[layer]
        layer_loss  = torch.nn.functional.mse_loss(target_gram, style_gram)
        # Normalize style loss by number of elements to avoid size bias
        style_loss += layer_loss / (target_feature.shape[1] * target_feature.shape[2] * target_feature.shape[3])
    # Total loss
    total_loss = content_weight * content_loss + style_weight * style_loss
    # Backpropagate and optimize
    total_loss.backward()
    optimizer.step()
    # Print progress every 50 steps
    if step % 50 == 0:
        print(f"Step {step}: Content Loss = {content_loss.item():.4f}, "
              f"Style Loss = {style_loss.item():.4f}, Total Loss = {total_loss.item():.4f}")

# Step 3: Visualize the result
final_img = tensor_to_image(target)
plt.imshow(final_img)
plt.axis('off')
plt.title('Output Image')
plt.show()
