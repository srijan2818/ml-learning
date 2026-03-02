# MLP Architecture — Design Framework


## Problem Structure

Given arbitrary input $x \in \mathbb{R}^d$, learn mapping $f: \mathbb{R}^d \to \mathbb{R}^k$ (classification or regression).

No assumptions about input structure:
- Features can have arbitrary relationships
- No spatial locality structure
- No sequential ordering (time dependencies)
- No symmetries (rotation, translation invariance)

Hypothesis class: Compositions of affine maps + nonlinearities.

$$f(x) = W_L \sigma(W_{L-1} \sigma(\cdots \sigma(W_1 x + b_1) \cdots) + b_{L-1}) + b_L$$

Each layer: $x^{\ell} = \sigma(W^{\ell} x^{\ell-1} + b^{\ell})$



## Inductive Bias

None at the architectural level. Every input feature can interact with every hidden unit.

This is the MLP's strength and weakness:
- Universal approximator - given enough capacity (width & depth), can represent any continuous function
- No structure encoded → needs large capacity → prone to overfitting on small datasets

Contrast with structured models:
- SVM encodes margin bias (prefer large-margin separators)
- Decision trees encode axis-aligned splits
- CNNs encode locality + weight sharing

MLP's dont assume anything, they learn everything from the data

## Capacity Control

### Width  - Neurons Per Layer

More hidden units → more linear combinations of previous layer.

Layer $\ell$ with $n$ neurons computes $n$ functions: $h_i^{(\ell)} = \sigma(\sum_j w_{ij} h_j^{(\ell-1)})$

- Each neuron is a feature detector. More neurons → more features → richer representation

  
- More parameters → more capacity → overfitting risk if data is limited.

### Depth - Number of Layers

Deeper networks learn compositional features

- Layer 1: Linear combinations of input
- Layer 2: Combinations of layer 1 features
- Layer 3: Combinations of combinations

Some functions are exponentially easier to represent with depth vs width

Example: Parity function $f(x_1, \ldots, x_d) = x_1 \oplus x_2 \oplus \cdots \oplus x_d$ needs $O(2^d)$ neurons in 2 layers, but $O(d)$ neurons in $O(\log d)$ layers.

- Vanishing gradients. Gradient at layer 1 is product of Jacobians:
$$\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial h^{(L)}} \prod_{\ell=2}^{L} \frac{\partial h^{(\ell)}}{\partial h^{(\ell-1)}} \frac{\partial h^{(1)}}{\partial W^{(1)}}$$

If each Jacobian has spectral norm $< 1$, product shrinks exponentially with depth.

Solutions: ReLU (gradient 1 for positive inputs), skip connections (ResNet), or careful initialization.



### Regularization

Same mechanisms as before:

L2 (Weight Decay): $L_{\text{total}} = L_{\text{data}} + \frac{\lambda}{2} \sum_{\ell} \|W^{(\ell)}\|_F^2$

Biases toward small-norm solutions. From optimization perspective: shifts Hessian eigenvalues → improves conditioning.

L1: Promotes sparsity (some weights → 0). Rarely used in deep networks (breaks smooth optimization, hard to tune).

Dropout: Randomly zero activations during training. Forces network to not rely on any single neuron.
Equivalent to training an ensemble of $2^n$ subnetworks (where $n$ = number of neurons).

Early Stopping: Implicit regularization. Stop training when validation loss stops decreasing. Equivalent to path-dependent regularization (early iterations explore low-complexity regions, later iterations enter high-capacity region).



## Failure Modes and Diagnostics

### Underfit

- Train loss high, test loss high.

- Due to insufficient capacity or insufficient training.

<br>If its underfitting then:
- If loss is decreasing → train longer.
- If loss has plateaud → increase capacity

<br>So add neurons, add layers, or train longer.


### Overfit

Train loss low, test loss high. Large gap between train and test.

 Model memorizes training data instead of learning generalizable patterns.


- Gap grows over training → Overfitting during training.
- Gap exists from start → Model capacity too high for dataset size.

Reduce capacity (fewer/narrower layers), add regularization (L2, dropout), get more data, or stop training earlier.


### Vanishing Gradients

- Loss barely decreases. Gradient norms in early layers → 0.
  
- Repeated multiplication of Jacobians with eigenvalues < 1.

- Log $\|\frac{\partial L}{\partial W^{(\ell)}}\|$ for earlier layer. If early layers have norms $\sim 10^{-8}$ while late layers have norms $\sim 10^{-2}$, gradients are vanishing.

Use ReLU, add skip connections, or reduce depth.

---

### Exploding Gradients

- Loss oscillates or diverges. Gradient norms → $\infty$.

- Repeated multiplication of Jacobians with eigenvalues > 1, or learning rate too high.

- Log gradient norms. If they grow over training, gradients are exploding.

- Reduce learning rate, clip gradients ($\text{if } \|g\| > \text{threshold, scale } g \leftarrow g \cdot \frac{\text{threshold}}{\|g\|}$), or fix initialization.

### When to Use 

1. Tabular data (age, income, zip code, medical records): No spatial/sequential structure. Features can interact arbitrarily.

2. Small input dimensions ($d < 1000$): Fully connected layers are tractable.

3. Final classification layer after structured feature extraction: After CNN/RNN/Transformer extracts features, MLP maps to class logits.


### Not to use

1. Images ($28 \times 28 = 784$ dims or larger): No spatial inductive bias. MLP treats pixel $(i,j)$ and pixel $(i', j')$ as unrelated, even if theyre neighbors. Needs massive capacity. Use CNN instead.

2. Sequential (text, time series): No temporal ordering encoded. MLP sees "the cat sat" and "sat cat the" as different but related by arbitrary feature interactions. Use RNN/Transformer instead.

3. Very high dimensions ($d > 10^4$): Parameter count explodes ($d \times h$ per layer). Use dimensionality reduction (PCA, autoencoders) or structured architectures (CNN, attention).



## Comparison to previous models

### Logistic Regression

MLP with:
- No hidden layers ($L = 1$)
- Sigmoid activation at output
- Cross-entropy loss

Logistic regression is a 1-layer MLP. MLP generalizes this by adding hidden layers → learns features automatically instead of using hand-crafted features.

---

### Kernel SVM

Kernel methods map input to infinite-dimensional feature space $\phi(x)$, then learn linear separator in that space.

MLP maps input through hidden layers $h^{(1)}, h^{(2)}, \ldots$, learning the feature map jointly with the classifier.

Difference: 
- Kernel SVM: Feature map fixed (kernel chosen beforehand), only separator is learned.
- MLP: Feature map learned via gradient descent on end task.

Trade-off: 
- Kernel SVM: Strong guarantees (convex, global optimum), but feature map not optimized for task.
- MLP: Task-specific features, but non-convex (no guarantees, local minima).

---

### Decision Trees

Decision trees partition input space with axis-aligned splits.

MLPs partition input space with oblique hyperplanes (each neuron computes $\sigma(w^T x + b)$, which is a hyperplane in input space after applying $\sigma^{-1}$).

MLPs are more expressive (oblique splits can capture diagonal boundaries), but harder to interpret.



## Optimization Geometry

### Loss Landscape

MLP loss is non-convex with many local minima and saddle points.

- Symmetries in weight space. If you permute hidden units in a layer, you get the same function but different weights. This creates equivalent minima connected by permutation paths.

- No guarantee gradient descent finds global minimum.
- But empirically, local minima often have similar loss (especially in overparameterized networks).



### Overparameterization

Modern deep learning: Often $p \gg n$ (more parameters than training samples).

VC Theory: This should overfit catastrophically.

Why it doesnt: 
- Implicit bias of SGD: Stochastic gradients prefer flat minima (robust to perturbations).
- Inductive bias of architecture: Even with many parameters, MLP can only represent smooth functions (ReLU networks have bounded variation).

---

## Summary: MLP Design Checklist

1. Data has no structure (arbitrary feature interactions) → MLP is appropriate.
2. Input dimension small ($d < 1000$) → MLP is tractable.
3. Need universal approximation (no prior knowledge of function structure) → MLP is flexible.
4. Start shallow (1-2 hidden layers). Add depth only if needed.
5. Monitor gradients (log norms per layer). If early layers vanish, use ReLU or reduce depth.
6. Regularize (L2, dropout, early stopping) unless dataset is huge ($n \gg p$).
7. Diagnose failures: Underfit → add capacity. Overfit → reduce capacity or regularize. Vanishing gradients → fix activation or depth.
