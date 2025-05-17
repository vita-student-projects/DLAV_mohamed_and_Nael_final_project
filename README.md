# DLAV Project – Phase 1  
**Contributors**: Mohamed and Nael  

## Milestone 1: End-to-End Trajectory Prediction

In this phase, we developed an end-to-end planning model that predicts the future trajectory of an ego vehicle based on:

- **Trajectory history**
- **RGB camera image** of the current scene

The model takes these inputs and outputs a sequence of predicted waypoints representing the future path of the vehicle.

---

## Model Architecture

Our architecture combines convolutional, transformer, and feedforward components to process visual and historical motion data:

### 1. Convolutional Neural Network (CNN)
The CNN processes the RGB camera image and extracts hierarchical spatial features. These features are represented as a 3D output volume that encodes important scene context.

### 2. Vision Transformer (ViT)
The CNN output is passed into a Vision Transformer (designed from scratch, not pre-trained since the input is not the camera image). Through the self-attention mechanism, the ViT allows different parts of the scene to communicate. It then compresses the spatial features into a single **learned special token**, that serves as a compact summary of the scene.

### 3. Feature Fusion
We simply concatenate the learned token from the ViT with the trajectory history. 

### 4. MLP Decoder
The fused concatenated vector is passed through a Multi-Layer Perceptron (MLP) decoder. The decoder outputs the predicted future trajectory.

---

### Training

The model is trained using the Adam optimizer with an initial learning rate of `1e-3`. A learning rate scheduler (`ReduceLROnPlateau`) is used to automatically reduce the learning rate by a factor of 0.5 with patience of 1.

For data augmentation, the training images are transformed using `ColorJitter`. We were hesitant to use spatial transformations to the training images, as we could not apply equivalent transformations to the GT future trajectory.

The training objective is based on Average Displacement Error (ADE), which calculates the mean L2 distance between the predicted and ground truth trajectory points.


## (Failed) Experiments
We tried complexifying the above architecture, but this simple architecture always turned out to perform (slightly) better (on ADE):
- We replaced the MLP decoder with an LSTM to predict the future trajectory, but it did not improve (on ADE), potentially because of the
   autoregressive nature of the LSTM and error accumulation. We now plan to try with a non-autoregressive decoder. We could not do it         before the Milestone 1 deadline, unfortunately.
- Instead of concatenating the history, we tried passing it through our ViT jointly with the RGB image. The ViT was not pretrained, and the history and image had different positional embeddings to facilitate distinction. In the same idea of trying to make the history and image "communicate", we now plan to use a (transformer) encoder-decoder architecture instead to make the distinction between the history and image more explicit; this will also allow us to use a pre-trained ViT to better encode the image (independantly of the history). 
- We replaced the CNN with a pre-trained CNN (ResNet 50), and removed the last layer (layer 4) to increase the number of output channels and spatial dimensions of the output volume to leave some food for the ViT. The bottelneck here was our light ViT on top of the CNN, which forced us to pool/compact (too much, apparently) the output volume into patches that matched the ViT's embedding dimension. We now plan to try to make it converge for a larger ViT.

## Usage

To load and use the model in the notebook, run both cells:

```python
# Upload the model weights (phase1_model.pth)
from google.colab import files
uploaded = files.upload()

# Load the model
model = DrivingPlanner()
model.load_state_dict(torch.load("phase1_model.pth"))
```

## Milestone 2

### Overview

The objectives are similar to those of Milestone 1, except there are additional inputs, namely a depth and semantic segmentation map, however
we only use the depth map as an auxiliary task.

### Model architecture

### 1. Image encoder
We use a (pre-trained) resnet-18 backbone to encode the RGB input image.
### 2. History encoder
We use a transformer decoder, where the queries from the history tokens attend to the keys and values from the ouput image tokens. This cross-attention mechanism allows the model
to "look back-and-forth" between the history and image of the current scene, thus baking into the history tokens' representation some information from the image encoder output.
### 3. Trajectory decoder
Here we use a simple MLP decoder, since the hard work is done by the image and history encoders. Also, from experience from Milestone 1, an MLP decoder worked better for us than an autoregressive
deocder (specifically an LSTM in our case), even though we might have just not implemented it correctly.
### 4. Auxiliary task
We use an upsampling network to predict the depth map, and thus enhance the image representation. However, this didn't really lead to an improvement in performance (ADE).









