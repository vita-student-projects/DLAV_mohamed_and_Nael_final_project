# DLAV project, phase 1
by Mohamed and Nael 
# Milestone 1
We implemented an end-to-end planning model that uses as inputs:
  - trajectory history
  - RGB camera image of the current scene
The model predicts the future trajectory of the vehicle.
# Architecture
Our model consists of a CNN that extracts hierarchical spatial features of the camera image, and returns a final ouput volume, that then goes through a ViT, which returns a learned special token. We then concatenate this learned token and the history, and finally, they jointly go through a final MLP decoder that ouputs the future/predicted waypoints.



