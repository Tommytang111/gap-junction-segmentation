Note: This branch is for CC cluster only, will eventually be deleted.

# Gap Junction Segmentation (GJS)
Welcome to the Gap Junction Segmentation Project! This repository contains the set of all codes required for training and utilizing customized deep learning models for the task of gap junction semantic segmentation. It also aids in the visualization and evaluation of model performance metrics and outputs. For the Gap Junction Segmentation Online Tool, please refer to the [Gap Junction Segmentation App](https://Github.com/Tommytang111/gap-junction-segmentation-app).

Gap junctions are specialized intercellular connections that facilitate direct communication between cells. Semantic segmentation models can be used to identify these proteins from electron microscopy images of C.elegans. The [Zhen Lab](https://zhenlab.com/) at the University of Toronto/Lunenfeld-Tanenbaum Research Institute investigates many aspects of neural circuits, including their electrical coupling and communication via gap junctions. Using specially-stained EM datasets from adult and dauer (alternative developmental pathway) C.elegans, we have developed a pipeline for creating 3D segmentation volumes of gap junctions from 2D EM slices. Such volumes enable the exploration of the electrical connectome of C.elegans, which answer questions related to the structure, function, and plasticity of electrical synapses in neural circuits.

### Project Structure

```
.
├── notebooks
│   ├── inference.ipynb
│   ├── sweep.ipynb
│   ├── testing.ipynb
│   ├── train.ipynb
│   └── utilities.ipynb
├── src
│   ├── inference.py
│   ├── models.py
│   ├── segment_dataset.py
│   ├── sweep.py
│   ├── train.py
│   ├── train_local.py
│   └── utils.py
└── README.md

2 directories, 13 files
```
