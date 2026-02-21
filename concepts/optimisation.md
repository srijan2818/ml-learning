# OPTIMISATION
(Just theoretical summaries no math here, just reference / summary style overview )

- Methods of finding the best parameters for a model so it can make accurate and reasonably fast decisions on unseen data.
- Based on currently available data, these parameters are chosen to be optimal with respect to a specific learning task<br>
The goal is to generalize a model , by making it learn underlying patters than just memorizing 

  
## Model Complexity Measure
- Capacity - size of hypothesis space / how complex functions can the model represent
- VC Dimension - largest number of points a model can shatter (classify in all possible ways). Higher VC dim = more complicated decision boundaries , but higher chance of overfitting
  
- Rademacher Complexity - How well a model class can fit random +/- 1 noise on a given dataset. High Rademacher complexity means it can fit pure noise, too complex. Lower means more general

## Optimisation Landscapes
1. Convex problems - simpler well behaved problems where any local improvement guaranteed leads to a better global solution - text classification
2. Non-convex problems - Highly complex and nonlinear, have local points that dont contribute to global best solution (stationary points) - image/speech recognition

## Optimisation Methods
- Operator - risk measure / function being minimized
- Geometry - shape of the search space and the path the algorithm takes to reach the solution
- Norm - measure distance / size of a step 
  - Euclidean distance - l2 norm
  - Simple/sparse models - l1 norm
  - based on local terrain - Hessian/Quadratic norm
- Eigenstrucutre - Scaling and curvature of the landscape 
   - Hessian matrix - steepness in all directions
   - Condition number - how stretched / narrow a valley is / degree of ill-conditioning
- Bias - Algorithms preference for a specific type of solution when many available 

## FIRST ORDER GEOMETRY - GLOBAL SCALAR STEP
###  Batch (Full) Gradient Descent 

- Operator :  Minimizes Empirical Risk by calculating the average loss across the entire dataset in each step.
- Geometry : Follows a direct route down the steepest slope of the error landscape
- Norm : Euclidean l2 norm
- Eigenstructure : speed entirely dependent on the condition number , ill-conditioned/high condition number /narrow bowl -> extremely slow
- Bias : GD converges to the global minimizer that is Euclidean-closest to the initialization point (without regularisation), On linear models it finds the max-margin solution, For neural networks towards low complexity functions.

In **full batch GD**, optmisation process is a direction-specific contraction of the error towards minimum. Geometrically the loss landscape is a valley (Hessian - shape , spectrum - set of eigenvals of H) where each direciton has a specific steepness(eigenvalue). Because GD uses a single step size for each direction(param), its trapped by the steepest direction (it must be small enough to not overshoot the walls of the largest eigenval / stability requires $\alpha < \frac{2}{\lambda_\text{max}}$), which forces the progress along the valley floor (smallest eigenval) to be very slow. This value is captured by condition number (dictates the number of iterations to converge). In overparameterized case itll go for directions to land on global minimizer l2-closest to starting point (its implicit bias). L2 regularisation mitigates these geometry issues by shifting the eigenvals(valley bases) upward, effectively reducing the ratio b/w steepest nd flattest directions (contraction (error reduction rate) -> 1000/10 = 100 to 1100/110=10)

###  Momentum  & Acceleration
- Operator - Minimizes loss by adding a velocity term to the gradient
- Geometry - creates persistent motion, update vector keeps moving in similar direction to previous step because of previous gradient's influence
- Norm - Euclidean
- Eigenstructure - dampens oscillation in high-curvature direction, model gains speed in the long flat directions of a valley
- Bias - towards consistency, assumes best direction forward is similar to the direction it was moving before

In **momentum**, the optimisation process transforms from a first-order walk into second order dynamical system with inertia nd acceleration. Geometrically this introduces eigenvalue-dependent (exponentially weighted average) damping. If the eigenval for a direction is high / steep the velocity from previous steps conunteracts current grad to prevent zigzag while in low eigenval / flat direction it gains acceleration. Iteration complexity now depends on $O(\sqrt{\kappa}\log{\frac{1}{\epsilon}})$. It still depends on the single global hyperparameter $\beta$, could be good for one EV but fail elsewhere. Also fails in stiff / non-uniform landscapes with fixed $\beta$. Momentum's implicit bias is kinetic and path-dependent, as accumulated energy could carry it past nearst minima to initial.

In **Nesterov Accelerated Gradient (NAG)**, the optimization process goes from reactive into anticipatory system with lookahead. Geometrically, this introduces curvature-aware damping by evaluating the gradient at the predicted future position rather than the current one. If a direction is steep ($\lambda_{\max}$), the lookahead gradient improves worst case convergence constant (adjusts step) providing stabilization against overshooting compared to standard momentum.
It achieves the provably optimal first-order iteration complexity of $O(\sqrt{\kappa}\log{1/\epsilon})$ by smoothing the trajectory near convergence (NAG would win the 99 to 99.999 race but not 0 to 90). However it still relies on a single global hyperparameter $\beta$, same issue as Momentum and GD with multi-scale. Its implicit bias is curvature-sensitive and path-dependent, favoring minima where Hessian changes smoothly

## DIAGONAL METRIC ADAPTION - PER-PARAMETER SCALING
### Adaptive Scaling (Adagrad, ADAM) - Individual tuning
- Operator - minimizes loss by assigning unique learning rate to each paramter
- Geometry - Operates in a distorted landscape where frequently updated parameters are slowed down and rarely updated ones are sped up
- Norm - diagonal scaling matrix
- Eigenstructure - Only accounts for the diagonal elements of the curvature, ignoring how parameters curve together
- Bias - Biased toward historical updates, parameters with small historical gradients are given a boost

In **AdaGrad**, the optimization process is per-parameter update system using a diagonal preconditioning matrix. Geometrically, this replaces the isotropic metric  (same scaling each direction) of GD with an axis-aligned scaling that independently stretches or compresses each coordinate based on its historical squared gradient accumulation. This method is optimal only when the directions of curvature (eigenvectors of the Hessian) are aligned with the coordinate axes , AdaGrad preconditions the valley into a more spherical landscape. However because the operator is strictly diagonal, it lacks the off-diagonal terms necessary to account for parameter correlations. If the loss landscape is rotated relative to the parameter axes, AdaGrad’s coordinate-wise rescaling fails to decouple the correlated dimensions, leaving the optimizer trapped by the true (off-diagonal) condition number. Furthermore, the monotonic accumulation ensures that the scaling only decreases eventually leading to a vanishing learning rate that stalls optimization regardless of the remaining distance to the minimum.



**RMSprop** improves upon AdaGrad's per-parameter step size scaling by replacing the cumulative gradient square sum with an exponential moving average. Its stil the same gradiant preconditioning, but adapted more to recent gradients than all using a "remembering" factor $\beta$ (typically 0.9-0.99) controlling the window size (1/1-$\beta$). This solves the vanishing gradient problem. Retains the same limitation as AdaGrad - strictly diagonal. However it introduces temporal adaptivity - different parameters can activate at different training phases introducing some sense of correlation and indefinite optimisation

**Adam** combines RMSprop's diagonal scaling with momentum using two separate timescales: fast momentum ($\beta_1 \sim 0.9$, ~10 steps) and slow second-moment estimation ($\beta_2 \sim 0.999$, ~1000 steps). Momentum is applied **after** diagonal rescaling—gradients get rescaled by $\sqrt{v_t}$ first, then momentum accumulates on these preconditioned gradients.This makes it robust to mixed per-parameter scaling and sparse gradients without vanishing learning rates in long runs. Still diagonal—can't capture off-diagonal correlations—but most adaptive among diagonal methods. However, the fast adaptive scaling can lock onto nearby sharp minima instead of exploring toward flatter regions.

##  FULL CURVATURE ADAPTION - OFF-DIAGONAL STRUCTURE
### Newton's method 
- Operator - Minimizes a quadratic approximation (simple parabola) of the local function at each step
- Geometry - Scale-invariant - always points towards the critical point (minima/maxima/saddle) regardless of how space is stretched
- Norm : Hessian norm 
- Eigenstructure : Relies on Hessian matrix to round out the search space
- Bias : biased towards local curvature, complex landscape can be approximated as simple parabola 

**Newton's Method** uses full curvature correction $H^{-1}$ instead of scalar or diagonal scaling. Does metric adaptation in eigenspace—each eigendirection gets rescaled by $1/\lambda_i$, which collapses condition number to 1 locally. For quadratics gives one-step convergence regardless of $\kappa$ . Captures full parameter correlations including off-diagonal terms, working in eigenvector space not parameter space.

Three problems though: cost is $O(d^2)$ storage and $O(d^3)$ inversion (impossible for millions of parameters), indefinite Hessian in nonconvex regions points uphill toward saddles, and stochastic gradients break the quadratic model with noise amplification. Also no bias toward flat minima—converges to nearest critical point in Hessian metric which could be sharp. Can't actually use it but explains why curvature matters and what approximations trade off.

**Gauss-Newton** fixes the indefinite Hessian issue by using $G = J^T J$ where $J$ is the Jacobian of residuals, instead of full $H$. Always positive semidefinite since $v^T G v = ||Jv||^2 \geq 0$, so always points downhill even in nonconvex regions. Works well when residuals are small or nearly linear—gives near-quadratic convergence like Newton by capturing curvature from linearized residuals.

Drops the $\sum r_i \nabla^2 r_i$ term from true Hessian though, so falls back to first-order speed when residuals get large or highly nonlinear. Conditioning gets squared: $\kappa(G) = \kappa(J)^2$, often needs Levenberg-Marquardt regularization ($G + \lambda I$) to fix this. Still $O(d^3)$ to invert so can't scale to big networks. Biases toward solutions reachable along nearly-linear residual paths.

**L-BFGS** builds low-rank approximation to $H^{-1}$ using only gradient history—no explicit Hessian computation. Stores last $m$ pairs (typically $m = 5$-$20$) of gradient changes $y_k = \nabla F_{k+1} - \nabla F_k$ and position changes $s_k = w_{k+1} - w_k$. Uses these to build $B_k^{-1} \approx H_k^{-1}$ with rank-2 updates satisfying $B_k s_k = y_k$ (approximate Hessian should match observed gradient changes). Storage drops to $O(md)$ instead of $O(d^2)$, each step costs $O(md)$.

Gets superlinear convergence for smooth problems with low effective rank—each stored pair captures one curvature direction, so $m$ pairs approximate Newton in an $m$-dimensional subspace of recent search directions. Works when few eigenvalues dominate and curvature changes slowly. Breaks under gradient noise since mini-batches corrupt the $B_k s_k = y_k$ condition, needs full batches. Only captures $m$ modes which isn't enough when curvature is full-rank across millions of dimensions. Good for deterministic problems or final refinement after SGD but doesn't scale to stochastic deep learning.

## INFORMATION GEOMETRY

**Natural Gradient** replaces Euclidean parameter-space descent with descent in **probability distribution space** using the Fisher Information Matrix $F(w) = \mathbb{E}[\nabla \log p \, \nabla \log p^T]$ as the metric. This measures step size in KL-divergence units, aiming for parameterization invariance— reparameterizing coordinates shouldn't change the optimization trajectory in distribution space.

For probabilistic models with cross-entropy loss, Fisher is the expected Gauss-Newton matrix $F = \mathbb{E}[J^T J]$, always positive semidefinite. Eigenvalues reflect information content: high eigenvalue means parameter strongly affects the distribution. This addresses ill-conditioning from parameterization choices by rescaling according to information geometry.

In practice : empirical Fisher from mini-batches is noisy and doesn't equal true Fisher, destroying the theoretical invariance. Requires $O(d^2)$ storage so practical methods use approximations (K-FAC for Kronecker-factored blocks, block-diagonal structure). Most relevant for models outputting distributions—classification, generative models, RL policies. For deterministic regression Fisher reduces to scaled Hessian with less benefit.
## **When to Use What**

**Deterministic convex smooth problems** (small-scale optimization, final refinement):
-> **L-BFGS** (superlinear convergence, no hyperparameter tuning)

**Large deep networks, stochastic training** (standard supervised learning):
-> **SGD + Momentum** (generalizes well, flat minima bias) or **Adam** (faster convergence, sparse gradients)

**Vision tasks** (CNNs, ResNets):
-> **SGD + Momentum** (better generalization than Adam empirically)

**NLP/Transformers** (sparse attention, extreme parameter scale differences):
-> **Adam** (handles sparsity and scale variation robustly)

**Sparse features** (word embeddings, one-hot encoded data):
-> **AdaGrad** (maintains high learning rates for rare features)

**Policy optimization / RL** (distribution outputs, sample efficiency matters):
-> **Natural Gradient / TRPO / PPO** (KL constraints prevent catastrophic updates)

**Ill-conditioned inverse problems** (PDE parameter estimation, physics-informed learning):
-> **Gauss-Newton** (if residuals structured) or **L-BFGS** (if deterministic full-batch)

**Final fine-tuning after SGD pretraining** (small-scale refinement):
-> **L-BFGS or Gauss-Newton** (switch to second-order when close to minimum)

**General heuristic**: Start with Adam for prototyping (robust to hyperparameters). If generalization matters and you have time, tune SGD+Momentum. If deterministic and small-scale, use L-BFGS.

--- 
### Coordinate Ascent 
- Operator - updates only one parameter at a time
- Geometry - moves along a single axis of the space
- Norm - l1 norm to introduce sparsity / set unimportant features to 0
- Eigenstructure - Ignores the interaction between different variables (cross-curves)
- Bias - towards structural simplicity, assumes independence of parameters/variables
  
### Noise reduction - SVRG (Stochastic Variance Reduced Gradient) , SAGA (Stochastic Average Gradient Algo)
- Operator - Uses memory corrected / variance reduced gradient 
- Geometry - transitions from noisy stochastic gradient to relatively stable linear descent near the solution
- Norm - Euclidean
- Eigenstructure -  Uses stored history to cancel out the noise in the data's curvature
- Bias -  towards finite-sum minimisation, it assumes the dataset is fixed and gradient info can be stored or periodically recalculated
### Stochastic Gradient Descent 
- Operator - Minimizes expected risk by using just one random sample (stochastic) at a time
- Geometry - Markovian manner operation - memoryless random walk, next step depends on current spot and random sample
- Norm - Euclidean norm
- Eigenstrucutre -Highly sensitive to noise as it sees a noisy gradient estimate
- Bias - biased  toward computational efficienc, it assumes that since data is redundant, a single sample is good enough to make progress

## Terms & Risk Measures

1. **Expected risk** - theoretical measure of a model's performance, average loss of a model across all possible data points. Ideal but unknown in practice<br>   $R(f) = \mathbb{E}_{(x,y) \sim P}[\mathcal{L}(f(x), y)]$ 

2. **Empirical Risk** - approx practical measure of expected risk, average loss of a model over a finite training set <br>
   $R_{\text{emp}}(f) = \frac{1}{n} \sum_{i=1}^n \mathcal{L}(f(x_i), y_i)$

3. **Generalisation gap** - The difference between the Expected Risk and Empirical Risk. A small gap means the model's performance on the training data is a reliable indicator of its future performance on unseen data
   
4. **Implicit bias** - A phenomenon where the optimization algorithm itself (not just the error formula) biases the model towards a certain type of solution - on certain datasets, GD naturally biases the model toward finding the widest possible margin between different classes of data
5. **Structural Risk Minimisation** (SRM) - finding the best model by balancing empirical performance with model complexity as the simplest version is more likely to contain the underlying pattern than a complex function adjusted to the noise in the data
6. **KL Divergence (Kullback-Leibler)** - Measures how much one probability distribution differs from another. $D_{KL}(P \| Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$. Non-symmetric (distance from P to Q != Q to P) and always non-negative.

7. **Fisher Information Matrix** - $F(w) = \mathbb{E}[\nabla \log p   \nabla \log p^T]$. Measures how much information the data carries about parameters. High Fisher values mean small parameter changes cause large distribution changes. Defines the metric for probability distributions where distance is measured by KL divergence: $D_{KL}(p(w) \| p(w + \delta w)) \approx \frac{1}{2} \delta w^T F(w) \delta w$. For classification with cross-entropy loss, Fisher equals the expected Gauss-Newton matrix.