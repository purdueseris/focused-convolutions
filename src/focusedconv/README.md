# focusedconv
Official library for the Focused Convolution, developed for public use by the High Efficiency, Low-Power Systems (HELPS) lab at Purdue University.

Use this library at your own risk. HELPS and Purdue University are not liable for any damages or consequences resulting from the use or modification of the library.

------------------------

## What's a focused convolution?
Normal CNNs operate convolutions on the entirety of the input image.
However, many input images have many pixels that are not very interesting (e.g. background pixels).
This means normal CNNs are wasting time and energy on those uninteresting pixels.

The Focused Convolution is designed to ignore any pixels that are outside the Area of Interest, focusing only on interesting pixels.
The weights and biases can be kept from the original CNN, allowing you to achieve the same accuracy while saving on energy and inference time.

## What's an fCNN?
By supplying an activation brightness threshold, you can filter the features from the early layers in your CNN, and then use that as an AoI for your later layers!
You can replace the later layers with focused convolutions but keep the weights and biases the same.
Thus, you avoid retraining, and pick up some energy and speed improvements! This modifed model is an "fCNN" - it maintains accuracy on par with the original CNN, but is faster on images with uninteresting pixels.

------------------------

## Usage
Focused Convolutions are designed as drop-in replacements for the standard `torch.nn.Conv2d` module from PyTorch.
You can use convenience functions from `focusedconv` to quickly convert CNNs into "fCNNs" - models that share the same weights and biases but can ignore pixels from outside an Area of Interest (AoI). Examples are shown below.

### Basic Example (fCNN)
First, choose an activation brightness threshold as described in our paper as `BRIGHTNESS_THRESHOLD`.
Next, choose the layers from your CNN that will be filtered by your threshold as `EARLY_LAYERS`.
Finally, use the `focusedconv.build_fcnn()` function to generate an fCNN that applies the threshold to the output of your early layers, and then uses the resulting Area of Interest (AoI) with compute-efficient focused convolutions later on.

```python
import copy
import focusedconv
from torchvision.models import vgg16, VGG16_Weights

original_vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval()
print("ORIGINAL VGG:", original_vgg)

NUM_TOP_LAYERS = 4 # Let's say you want to use the first 4 layers of VGG as EARLY_LAYERS
BRIGHTNESS_THRESHOLD = 90 # Let's say your activation brightness threshold is 90
EARLY_LAYERS = copy.deepcopy(original_vgg.features[0: NUM_TOP_LAYERS])
vgg16_remaining_layers = copy.deepcopy(original_vgg.features[NUM_TOP_LAYERS: ])

focused_vgg = copy.deepcopy(original_vgg)

# Produces fVGG with focused convolutions, allowing you to get similar accuracy as the original VGG,
# but with efficient focused convolutions for faster inference speed and reduced energy use
# Also generates an AoI object for you to use
focused_vgg, aoi = focusedconv.build_focused_model(focused_vgg, vgg16_top_layers, ACTIVATION_BRIGHTNESS_THRESHOLD, vgg16_remaining_layers)
print("fVGG with FOCUSED CONVOLUTIONS:", focused_vgg)
```

### Advanced Example (manual focused convolution usage)
If you want more finetuned control when using focused convolutions, you can specify your own AoI mask yourself before each inference.
The `focusedconv.focusify_all_conv2d()` function is designed to recursively run in place on a PyTorch `nn.Module`, replacing all `Conv2d` layers with `FocusedConv2d` layers.
Each layer points to your specified AoI, allowing you to change the AoI whenever you like; the focused convolutions will adjust their behavior accordingly. 

```python
import focusedconv
import copy
import torch
import torch.nn as nn

# Let's say you have this simple CNN that does 1x1 convolutions on 
# a 3-channel RGB image and produces a 1-channel grayscale image
cnn = nn.Sequential(
    nn.Conv2d(3, 1, (1,1)),
    nn.ReLU(),
)
print("Original CNN:", cnn)

# Make an area of interest (AoI) mask that only keeps the top left 25% of the image:
aoi_mask = torch.as_tensor([
    [1, 0],
    [0, 0]
])

# Initialize an AoI object with this mask
aoi = focusedconv.areaofinterest.AoI(aoi_mask)

# Produce a model that shares the weights and biases of the CNN
# but uses Focused Convolutions instead with the specified AoI mask
focused_cnn = copy.deepcopy(cnn)
focusedconv.focusify_all_conv2d(focused_cnn, aoi)
print("Focused CNN:", focused_cnn)

# Your input image
input_img = torch.rand([1, 3, 4, 4])

# AoI size is calculated as 25% of the image
print("AoI size is:",  aoi.calculate_aoi_size())
aoi.imshow() # Can be used to display the AoI

# Observe that the focused CNN only provides output in the
# top left corner of the output
print("Output of original CNN:\n", cnn(input_img))
print("Output of focused CNN:\n", focused_cnn(input_img))

# Change the AoI so that it now includes the entire top half of the image:
aoi.mask_tensor[0, 1] = 1

# Observe that the focused CNN's output now provides the entire top half
print("New AoI size is:", aoi.calculate_aoi_size())
print("Now, output of focused CNN:\n", focused_cnn(input_img)) 
```