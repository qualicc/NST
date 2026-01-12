import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


content_path = "./ressources/land2.jpg"  # Replace with your content image path
style_path   = "./ressources/noc.jpg"    # TODO: replace with your style image path
output_path  = "./results/output.jpg"  # TODO: replace with your output directory


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

def load_image(img_path, max_size=512, shape=None):
    """Load an image and convert it to a normalized tensor."""
    image = Image.open(img_path).convert('RGB')
    if max_size is not None:
        if shape is not None:
            image = image.resize(shape, Image.LANCZOS)
        else:
            w, h = image.size
            if max(w, h) > max_size:
                if w >= h:
                    new_h = int(max_size * h / w)
                    new_w = max_size
                else:
                    new_w = int(max_size * w / h)
                    new_h = max_size
                image = image.resize((new_w, new_h), Image.LANCZOS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image_tensor = transform(image)[:3, :, :].unsqueeze(0)
    return image_tensor

def tensor_to_image(tensor):
    """Convert a normalized tensor to a numpy image (in [0,1] range)."""
    image = tensor.detach().cpu().clone().squeeze(0)
    image = image.permute(1, 2, 0).numpy()
    image = image * np.array(std) + np.array(mean)
    image = np.clip(image, 0, 1)
    return image

vgg = models.vgg19(pretrained=True).features  
for param in vgg.parameters():
    param.requires_grad = False  
vgg.to(device)
vgg.eval()

content_image = load_image(content_path, max_size=512).to(device)
style_image   = load_image(style_path, max_size=512, shape=(content_image.size(3), content_image.size(2))).to(device)

content_layer = 'conv4_2'
style_layers  = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

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
            x = layer(x)  
        elif isinstance(layer, torch.nn.MaxPool2d):
            x = layer(x)
            block += 1
            conv_count = 0
        elif isinstance(layer, torch.nn.BatchNorm2d):
            x = layer(x)
    return features

def gram_matrix(tensor):
    """Compute the Gram matrix of a tensor (for style features)."""
    _, n_filters, h, w = tensor.size()
    tensor = tensor.view(n_filters, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

content_features = get_features(content_image, vgg)
style_features   = get_features(style_image, vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}

target = content_image.clone().requires_grad_(True).to(device)

content_weight = 1e0 
style_weight   = 1e6

optimizer = torch.optim.Adam([target], lr=0.003)

num_steps = 300 
for step in range(1, num_steps+1):
    optimizer.zero_grad()
    target_features = get_features(target, vgg)
    content_loss = torch.nn.functional.mse_loss(
        target_features[content_layer], content_features[content_layer]
    )
    style_loss = 0
    for layer in style_layers:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram  = style_grams[layer]
        layer_loss  = torch.nn.functional.mse_loss(target_gram, style_gram)
        style_loss += layer_loss / (target_feature.shape[1] * target_feature.shape[2] * target_feature.shape[3])
    total_loss = content_weight * content_loss + style_weight * style_loss
    total_loss.backward()
    optimizer.step()
    # Print progress every 50 steps
    if step % 50 == 0:
        print(f"Step {step}: Content Loss = {content_loss.item():.4f}, "
              f"Style Loss = {style_loss.item():.4f}, Total Loss = {total_loss.item():.4f}")

final_img = tensor_to_image(target)
Image.fromarray((final_img * 255).astype(np.uint8)).save(output_path)
plt.imshow(final_img)
plt.axis('off')
plt.title('Output Image')
plt.show()
