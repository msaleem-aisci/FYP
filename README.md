# TheHulkNet: Multi-Scale CNN for Crop Disease Classification (Version 1 Baseline)

**Research Team:** BlueMind

## 1. Abstract
TheHulkNet is a custom convolutional neural network designed from scratch to classify agricultural pathology across multiple crop species. The baseline architecture moves away from standard pre-trained transfer learning (e.g., ResNet) in favor of a custom multi-branch topology intended to capture varying receptive fields. It successfully classifies 13 distinct healthy and diseased states across four primary crops (Corn, Potato, Tomato, and Wheat), incorporating gradient-based explainability to provide visual justifications for its predictions.

## 2. Dataset and Preprocessing
The model is trained on a localized agricultural dataset containing 13 classes of crop leaves. 

* **Robust Data Augmentation:** To prevent overfitting and simulate real-world field conditions (lighting changes, camera angles, wind distortion), the training pipeline applies aggressive dynamic augmentation. This includes Random Resized Cropping, Affine Transformations (shear and translation), Color Jittering (brightness, contrast, saturation), and multi-axis flipping.
* **Resolution:** All inputs are standardized and normalized to a $256 \times 256$ spatial resolution to preserve the intricate structural details of foliar lesions.

## 3. Baseline Architecture (BranchedCNN)
The Version 1 architecture is engineered to extract features at multiple spatial scales simultaneously, mimicking early Inception-style topologies. 

* **Multi-Scale Branches:** The input tensor is fed into three parallel convolutional pathways utilizing different kernel sizes ($3 \times 3$, $5 \times 5$, and $7 \times 7$) and strides. This design theoretically allows the network to capture both fine-grained necrotic spotting (via smaller kernels) and broad structural decay (via larger kernels).
* **Batch Normalization:** Applied after every convolutional layer to stabilize the internal covariate shift and accelerate convergence.
* **Feature Fusion:** The outputs of the three branches are spatially standardized using `AdaptiveAvgPool2d(4, 4)`, flattened, and concatenated along the feature dimension before being passed into a heavily regularized (Dropout = 0.5) Dense classification head.

## 4. Interpretability (Saliency Maps)
To prevent the model from acting as a "black box," the pipeline implements gradient-based Saliency Mapping. By computing the derivative of the predicted class score with respect to the input image tensor (`input_tensor.grad`), the system generates a heat map highlighting the exact pixels (e.g., leaf lesions or rust spots) that drove the classification decision.

## 5. Results and Critical Evaluation
The baseline model achieves strong top-line accuracy but exhibits specific failure modes when subjected to Confusion Matrix analysis.

* **Inter-Species Success:** The model demonstrates near-perfect accuracy in distinguishing between crop species (e.g., it does not confuse Corn with Wheat).
* **Intra-Species Limitations:** The network struggles to differentiate between morphologically similar diseases within the same species. Most notably, it exhibits confusion between *Tomato Early Blight* vs. *Tomato Late Blight*, and *Potato Early Blight* vs. *Potato Late Blight*.

## 6. Limitations and Version 2 Roadmap
Analysis of the Version 1 architecture reveals three specific bottlenecks that cause the intra-species confusion. The ongoing development of Version 2 will address these directly:

1. **Information Loss via Pooling:** The aggressive `AdaptiveAvgPool2d(4, 4)` forces high-resolution branches to match low-resolution branches, destroying the fine-grained texture data required to differentiate early vs. late blight. *V2 will implement a shared hierarchical backbone before branching.*
2. **Naive Concatenation:** The dense layer currently treats all branch features equally. *V2 will introduce a learned Gated Fusion mechanism to dynamically weigh the importance of the $3 \times 3$ vs $7 \times 7$ branches based on the input.*
3. **Background Noise:** Standard convolutions process soil and leaf equally. *V2 will integrate a Convolutional Block Attention Module (CBAM) to spatially mask out the background and strictly focus on the pathological lesions.*
