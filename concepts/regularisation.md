# REGULARISATION

Goal is to introduce inductive bias to the model to converge towards a specific solution out of many available by constraining the hypothesis space to improve generalisation.

## TYPES OF REGULARISATION

### EXPLICIT REGULARISATION

Add a term to the objective:
<br><div align='center'>   $\min_w \hat{R}(w) + \lambda \Omega(w)$ </div>

where $\lambda \Omega(w)$ penalizes complexity.

## IMPLICIT REGULARISATION

Optimisation algorithm itself biases towards certain solutions wtihout explicit penalty.

- GD finds $\ell_2$ colsest minimum to initial point
- SGD biases towards flat minima
  
## EXPLICIT REGULARISTIONS - NORM PENALTY

### L2 REGULARISATION - RIDGE/WEIGHT DECAY

Adds penalty term proprotional to sum of squared weights, to encourage all the weights towards 0 but not set them to 0.

<br><div align='center'> $\min_w \hat{R}(w) + \frac{\lambda}{2}||w||_2^2$ </div>

- Since its a quadratic term, it appears in the Hessian as $H \rarr H + \lambda I$ , thus shifting all eigenvalues by $\lambda$ . This reduced the condition number $\kappa = \frac{\lambda_{max} + \lambda}{\lambda_{min} + \lambda}$ .
- The penalty gradient is  proportional to the weight . In eigenspace, small eigenvalues (flat directions) get proportionally larger relative boost from the $\lambda I$ term added to the Hessian, improving conditioning.
- Geometrically, it introduces a contraint circle over the contour of loss landscape which closes in on the valley, since its most likely we would hit some space not along axis since its a smooth surface, no weight becomes 0, they just shrink. 
- Shrinks weights towrads 0 but never exactly 0.

- Prefers solutions with small $\ell_2$ norm and among same empirical minima, picks one with smallest weights.

<br>Helps in :
- Ill-conditioned problems
- Overparameterized problems - many solutions, pick minimum weights one

<br> Fails when :
- True weights are sparse (they wont ever be set to 0)
- All features are equally important (uniform shirnkage wont help)

### L1 REGULARISATION - LASSO
Adds penalty term proportional to absolute sum of weights, forcing irrelevant weights to 0 allowing feature selection and sparse models helpful in high-dimensional data.

<br><div align='center'> $\min_w \hat{R}(w) + \lambda||w||_1$ </div>
- Penalty subgradient is constant which provides a non-vanishing pressure towards the origin. And since gradient is global /doesnt scale with weight, it can drive a weight to zero.
- It doesnt act directly in eigenspace, it simply mutes any feature whose contribution to loss reduction is less than $\lambda$.
- Geometrically, it introduces a diamond shaped contraint which has singularities (sharp corners) along axes. When GD hits these corners it sets some directions to 0 and induces sparsity.
- Prefers solutions with minimum absolute sum of weights, geometrically forces the optmisation paths towards the axes / kill unimportant weights.

<br> Helps in:
- Feature selection - Few features are relevant  
- Model compression
- Interpretability - completely remove influence of noise
  
<br> Fails when:
- Collinearity - when features are higly correlated, only one is arbitrarily set to 0
- When number of features > samples(n), it can select at select at most n variables.

### ELASTIC NET
To address the limitations of Lasso (specifically the instability with correlated features), Elastic Net combines both L1 and L2 penalties.

<br><div align='center'> $\min_w \hat{R}(w) + \lambda_1 ||w||_1 + \frac{\lambda_2}{2}||w||_2^2$ </div>

- L1 sparsity + L2 grouping
- Constraint shape is like a pointy circle, sharp vertices + curvature
- L2 component forces correlated features to have similar coefficients rather than one arbitrarily being set to 0.

This is looking like a high-tier study guide. Your summaries for L1 and L2 are technically dense and accurate. I'll complete the **Implicit Regularization** section (Layer 3) and the **Constraint vs. Penalty** section (Layer 2) to maintain the momentum and structural flow of your document.

---

## IMPLICIT REGULARISATION

The algorithm’s choice of path through the parameter space acts as a regularizer, selecting specific minima even when the loss function doesn't explicitly penalize them.

### GRADIENT DESCENT (GD) - $L_2$ BIAS

For overparameterized linear models, GD converges to the solution that is closest to the initialization point in Euclidean distance.

- In linearly separable classification with BCE/Log loss it converges towards max margin solution (Hard SVM)
- For neural networks GD biases towards low complexity functions in hypothesis space

### SGD NOISE - FLAT MINIMA BIAS

Stochasticity from mini-batches prevents the optimizer from settling in sharp regions of the loss landscape.

- Flat minima are stable under small perturbations. Since test data is essentially a perturbed version of training data, flat regions generalize significantly better.

- Not a universal rule, some sharp minima generalize well, if the sharpness is in irrelevatn direction.

### EARLY STOPPING - PATH CONTROL

Stopping at iteration $T$ is equivalent to $\ell_2$ regularization with time-varying $\lambda(t)$ that decreases as training progresses. Early $T$ = strong regularization, late $T$ = weak regularization.

- Weights typically start small and grow to capture complex details. Stopping early prevents the model from entering the high-complexity phase.
- In linear cases, early stopping is equivalent to  regularization where the penalty  is inversely proportional to the number of iterations.
### DROPOUT - ENSEMBLE REGULARIZATION

Randomly drops neurons (sets activations to 0) with probability $p$ during training.

- Implicit effect: Trains $2^n$ subnetworks (where $n$ = number of neurons)
- At test time: Use full network with scaled weights (multiply by $1-p$)  
- Inverted Dropout : Scale the activations up during training (divide by $1-p$) during training instead of test time.
- Prevents co-adaptation — forces each neuron to be useful independently
- Adaptive $\ell_2$ penalty on weights (approximate, not exact)

Helps: Deep networks, prevents reliance on specific neuron combinations  
Fails: Already small networks, adds noise without benefit

### DATA AUGMENTATION - INVARIANCE REGULARIZATION

Artificially expands training set by applying transformations (rotations, crops, noise) that preserve labels.

- Encodes prior knowledge about invariances 
- Changes the empirical distribution, not the loss function
- Forces model to learn robust features invariant to  transformations

Helps : Image tasks (rotation/translation invariance), low data regimes
<br>Fails: When transformations violate semantics (horizontal flip for left/right classification)

### BATCH NORMALIZATION - CONDITIONING + NOISE REGULARIZATION

Normalizes activations per mini-batch: $\hat{x} = \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma_{\text{batch}}^2 + \epsilon}}$


- Conditioning improvement: Rescales activations -> prevents gradient explosion/vanishing -> acts like adaptive preconditioning per layer
- Implicit noise: Mini-batch statistics vary -> adds noise to forward pass -> regularization similar to dropout

Helps:
- Stabilizes training (allows higher learning rates)
- Reduces sensitivity to initialization
- Adds stochasticity even in deterministic optimization

Fails:
- Very small batch sizes (statistics become too noisy)
- batch-dependent inference (online learning)

## CONSTRAINTS VS. PENALTIES

Regularization can be expressed in two mathematically dual ways:

### PENALTY FORM (UNCONSTRAINED)

<br><div align='center'>$\min_w \hat{R}(w) +\lambda \Omega(w)$</div>

 - A soft penalty where you pay a price for complexity.
 -  Standard in Deep Learning (e.g., Weight Decay) because its easy to optimize via backpropagation.

### CONSTRAINT FORM (IVANOV REGULARIZATION)
<br><div align='center'>$\min_w \hat{R}(w) \quad\text{s.t}\quad \Omega(w)\leq c$</div>

 -  A hard penalty on complexity. If the optimizer tries to leave the constrained space its projected back to the surface.
-  The solution occurs exactly where the loss contour becomes tangent to the constraint surface .



## MARGIN 

Regularization is also essentially margin maximisation in a way.

-  For linear classification :
  Margin ($\gamma$ = $\frac{1}{||w||}$)
   Maximizing margin  is equivalent to minimizing $||w||$.
-  SVM explicitly maximizes margin


---

### DECISION MAP
| **Problem** | **Regularization** | **Why** |
|-------------|-------------------|---------|
| Linear regression, ill-conditioned | L2  | Improves $\kappa$, stabilizes solution |
| High-dimensional sparse features | L1  | Feature selection, interpretability |
| Correlated features + sparsity | Elastic Net | L1 sparsity + L2 grouping |
| Deep networks, standard training | Weight decay + early stopping | Prevents overfitting, implicit flat minima bias |
| Classification with clear margin | SVM / max-margin | Explicit margin maximization |
| Small dataset, high capacity | Strong L2 + dropout + data aug | Explicit + implicit regularization |
| Large dataset, underparameterized | Minimal regularization | Data itself regularizes |

**Weight Decay vs L2 Regularization**:

- L2 penalty: Adds $\frac{\lambda}{2}\|w\|^2$ to loss → gradient becomes $\nabla L + \lambda w$
- Weight decay: Directly shrinks weights: $w \leftarrow w - \alpha(\nabla L + \lambda w)$

For SGD: These are identical (just absorb $\lambda$ into learning rate).

For Adam/RMSprop: They differ because adaptive scaling is applied:
- L2: Adaptive scaling affects the $\lambda w$ term → weaker regularization
- Weight decay: Shrinks weights independently of adaptive scaling → stronger regularization
Use weight decay for Adam to get proper regularization.