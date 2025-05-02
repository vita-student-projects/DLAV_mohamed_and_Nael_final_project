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
The CNN output is passed into a Vision Transformer. Through the self-attention mechanism, the ViT allows different parts of the scene to communicate. It then compresses the spatial features into a single **learned special token**, that serves as a compact summary of the scene.

### 3. Feature Fusion
We simply concatenate the learned token from the ViT with the trajectory history. 

### 4. MLP Decoder
The fused concatenated vector is passed through a Multi-Layer Perceptron (MLP) decoder. The decoder outputs the predicted future trajectory.

---

## (Failed) Experiments
We tried complexifying the above architecture, but this simple architecture always turned out to perform (slightly) better:
- We replaced the MLP decoder with an LSTM to predict the future trajectory, but it did not improve, potentially because of the
   autoregressive nature of the LSTM and error accumulation.
- Instead of concatenating the history, we tried passing it through our ViT jointly with the RGB image. The ViT was not pretrained, and the history and image had different positional embeddings to facilitate distinction. We plan to use a (transformer) encoder-decoder architecture instead to make the distinction more explicit, this will also allow us to use a pre-trained ViT to better encode the image (indivudually). We could not implement this before the Milestone 1 deadline unfortunately.
- We replaced the CNN with a pre-trained CNN (ResNet 50), and removed the last layer (layer 4) to increase the number of output channels and increase spatial dimensions to leave food for the ViT. This did not yield a better ADE, which means that we probably got close to what this architecture could express.








