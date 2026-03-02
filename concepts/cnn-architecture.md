# CNN Architecture — Design Framework


## Problem Structure

Given spatial grid input $X \in \mathbb{R}^{H \times W \times C}$ (e.g., image with height $H$, width $W$, $C$ channels), learn mapping to:
- Classification: $f: \mathbb{R}^{H \times W \times C} \to \mathbb{R}^k$ (class logits)
- Segmentation: $f: \mathbb{R}^{H \times W \times C} \to \mathbb{R}^{H \times W \times k}$ (per-pixel labels)
- Detection: $f: \mathbb{R}^{H \times W \times C} \to \mathbb{R}^{H' \times W' \times k}$ (heatmaps or bounding boxes)

Assumed structure:
- Locality: Nearby elements (pixels, grid cells) are correlated
- Stationarity: Same patterns appear at different locations (edges, textures repeat)
- Hierarchy: Low-level patterns (edges) compose into mid-level (textures) and high-level (objects)

Not assumed: Long range dependencies between distant regions (pixel at top-left directly depends on pixel at bottom-right)


## Inductive Bias: Locality + Weight Sharing

### Locality

Each output depends only on a local receptive field of the input.

Standard fully connected layer: Every output connects to every input.

Parameter count: $d \times h$ (where $h$ = number of outputs).

Convolutional layer: Each output connects only to a $k \times k$ window of inputs.
$$y_{i,j,c} = \sigma\left(\sum_{m,n,c'} w_{m,n,c',c} \cdot x_{i+m, j+n, c'} + b_c\right)$$
Parameter count: $k \times k \times C_{\text{in}} \times C_{\text{out}}$ (independent of spatial size $H, W$).

Why this helps: For images, pixel $(i,j)$ is strongly correlated with neighbors $(i \pm 1, j \pm 1)$, weakly correlated with distant pixels. Fully connected wastes capacity learning 

---

### Weight Sharing (Translation Equivariance)

Same filter applied at every spatial location. If input shifts, output shifts identically.

- If $f(x) = y$, then $f(T_{\delta} x) = T_{\delta} y$, where $T_{\delta}$ is spatial translation by $\delta$.

- Edge detector at top-left works equally well at bottom-right. No need to learn separate detectors for every position.

- Assumes patterns are position-independent. Fails if patterns are position-specific
---

### Connection to Optimization

CNNs reduce parameter count compared to MLPs:
- MLP on 224×224×3 image: $224 \times 224 \times 3 \times 128 \approx 19M$ parameters (first layer alone)
- CNN with 64 filters, 3×3 kernels: $3 \times 3 \times 3 \times 64 = 1,728$ parameters (first layer)

Smaller parameter count → smaller hypothesis class → better generalization (for spatial data where the bias is correct).

This is architectural regularization, like margin in SVM. Not added during optimization.


## Architectural Components

### Convolution

Operation: Slide $k \times k$ filter over spatial grid, compute dot product at each position.

Effect: Learns $C_{\text{out}}$ patterns (filters). Each filter activates where its pattern matches the input.

Receptive field: Each conv layer with $k \times k$ kernel adds $(k-1)/2$ to receptive field radius. Stack $L$ layers with 3×3 kernels → receptive field of $1 + L \cdot 2 = 2L + 1$.

---

### Pooling

Max pooling: For each $p \times p$ window, output the maximum value.
$$y_{i,j} = \max_{m,n \in \text{window}} x_{i \cdot p + m, j \cdot p + n}$$

Types: Max,Mean,Sum,Min,Stochastic,L_p,Global Max/Average (at the end of fc)

Effect: Downsamples spatial dimensions by factor $p$. $H \times W \to (H/p) \times (W/p)$.


1. Larger receptive field: After pooling by 2×, each pixel in the next layer sees 2× the input region.
2. Local translation invariance: Max pool over 2×2 → output unchanged if pattern shifts by 1 pixel within the window.
3. Computational efficiency: Smaller spatial size → fewer FLOPs in later layers.


- Spatial resolution: Cannot localize precisely after aggressive pooling. If you pool 224×224 → 7×7, output can only say pattern exists somewhere in this 32×32 region, not at a specific pixel.

Alternatives: Strided convolution (downsample without losing information to max operation), dilated convolution (larger receptive field without downsampling).



### Activation Functions

Mostly ReLU and variants are used.


- No saturation for positive activations → gradients dont vanish
- Computationally cheap
- Empirically: works well for natural images

- Leaky ReLU (allows small negative gradient), GELU (smooth approximation).



## Capacity Control

### Depth  - Number of Conv layers

Deeper networks learn hierarchical features:
- Layer 1: Edges, colors, simple textures
- Layer 2: Corners, combinations of edges
- Layer 3: Parts of objects (wheels, eyes, windows)
- Layer 4+: Whole objects

- Compositionality - A 3-layer network can represent functions that require exponentially many neurons in a 2-layer network (same principle as MLP, but stronger for spatial data).

Causes vanishing gradients for which we can use:
- Batch normalization: Normalize activations per layer → stabilizes gradient flow.
- Skip connections: Gradient has direct path via identity, doesnt vanish even if $f$ has small gradients.

---

### Width - Number of filters per layer

More filters → more patterns detected per layer.

Layer with 16 filters: Detects 16 different features (eg, 8 edge orientations + 8 texture types).

Layer with 64 filters: More representations, but more parameters

Trade-off: Width increases capacity without depth's gradient flow issues. But depth gives compositionality (features of features).

  - Increase width as depth increases. Early layers: 64 filters. Middle layers: 128-256. Late layers: 512-1024.

---

### Pooling Strategy

Aggressive pooling (pool after every layer): Small output (e.g., 7×7), large receptive field. Good for classification where we dont need spatial precision.

Conservative pooling (pool every 2-3 layers): Larger output  smaller receptive field. Good for segmentation (need pixel-level precision)

Skip connections (U-Net): Encoder pools aggressively (224 → 7), decoder upsamples (7 → 224), skip connections pass high-res features from encoder to decoder. Gets both: large receptive field (from deep encoder) + spatial precision (from skip connections).

---

### Regularization

Same mechanisms as MLP (L2, dropout, early stopping), plus:

Data augmentation: Random crops, rotations, flips, color jitter. Encodes task-invariant transformations (digit identity doesnt change if rotated +-15 degrees).

Batch normalization: Normalizes activations per mini-batch. Acts as implicit regularization (batch statistics add noise, similar to dropout).



## Failure Modes and Diagnostics

### Filters Look Like Noise

Symptom: After training, Conv1 filters have no structure (random pixel values).

Causes could be:
1. Task has no spatial structure - wrong model choice
2. Training failed (learning rate too low, bad initialization, insufficient epochs).

- Check task appropriateness first. If task is spatial, check learning rate, initialization, and training curves.

---

### Overfitting Despite Few Parameters

- Train loss low, test loss high, even though parameter count is small.

- CNNs inductive bias is wrong for the task. eg) Training CNN on synthetic data where patterns are not translation-invariant (top-left is always type A, bottom-right is always type B).

- Add data augmentation (if invariances should exist but arent in training data), or switch to position-dependent model ( add positional embeddings, or use MLP on flattened features).

---

### Poor Localization

Classification works (98% accuracy), but cant localize objects precisely (bounding boxes are off by 20 pixels).

Cause: Too much pooling. Final feature map is 7×7, but input is 224×224. Each cell in 7×7 corresponds to 32×32 region → +-16 pixel uncertainty.

- Use encoder-decoder with skip connections (U-Net), or reduce pooling frequency, or use dilated convolutions to increase receptive field without downsampling.

---

### Vanishing Gradients in Deep Networks

Same as MLP. Gradients in early layers → 0.

- Batch normalization + skip connections. Or reduce depth.



### Use with

1. Images: Natural images, medical scans, satellite imagery. Strong locality + translation invariance.

2. Spatial grids: Any data arranged on a 2D grid where neighbors are related. Examples: Game boards (Go, chess), spatial graphs embedded in 2D (molecules visualized as graphs).

3. Signals with local structure: 1D CNNs for audio (spectrograms), time series (if local patterns repeat).

---

### Dont use with

1. Tabular data: No spatial structure. Age and income are not neighbors in any meaningful sense.

2. Long-range dependencies: If pixel at (0,0) directly affects pixel at (100,100) with no intermediate locality, CNN's receptive field is inefficient. (Need attention - transformers).

3. Position-dependent patterns: If top-left has different patterns than bottom-right, weight sharing hurts. Add positional encodings or use position-dependent architectures.



## Connection to Prior Models

### MLP

CNN is MLP with:
- Sparse connectivity: Each neuron connects only to local window, not all inputs.
- Weight sharing: Same weights at all positions.

Trade-off: MLP is more flexible (can learn position-dependent patterns), CNN is more constrained (enforces translation equivariance).

For spatial data, CNN's constraint is correct → better generalization with fewer parameters.


## Optimization Geometry

### Loss Landscape

Same as MLP: non-convex, many local minima and saddle points.

Additional structure: Weight sharing creates more symmetries (permute filters → same function). More equivalent minima than MLP.

Empirical observation: Overparameterized CNNs (many more filters than "needed") have many good local minima. Underparameterized CNNs have fewer good minima, harder to optimize.

---

### Implicit Bias of SGD on CNNs

Same as MLP: SGD prefers flat minima (robust to perturbations).

CNN-specific: Batch normalization + data augmentation + weight sharing create strong implicit bias toward solutions that:
- Use low-frequency filters (smooth patterns, not high-frequency noise)
- Generalize across spatial positions (same filter activates on left and right sides of image)

---

## Summary: CNN Design Checklist

1. Data has spatial structure (images, grids) → CNN is appropriate.
2. Patterns are translation-invariant (same features at all locations) → weight sharing helps.
3. Start shallow (3-5 conv layers). Add depth only if hierarchy is needed.
4. Pooling strategy: Aggressive for classification, conservative for localization. Use U-Net if you need both.
5. Monitor filter visualizations: If Conv1 looks like noise, training failed or task is inappropriate.
6. Regularize: Data augmentation (rotation, crop, flip) is critical for small datasets. Batch norm is standard.
7. Diagnose failures: Same as MLP (underfit, overfit, vanishing gradients), plus check if filters learn structure.

