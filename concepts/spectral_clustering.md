# Spectral Clustering

Spectral clustering exists because k-means assumes convex clusters in input space itself, spectral clustering moves the problem to a space defined by relationships between data points not euclidean distances

- Connectivity not centroids
- Graph structure not geometry  
- Pairwise relations not distributions

<br>
The objective is to cut edges while keeping strongly connected nodes together → combinatorial problem<br>
Since direct optimization is not feasible,  relax to a continuous problem<br>
Spectral methods introduce eigenvectors of the graph Laplacian as the new representation<br>
These eigenvectors define an embedding where proximity reflects graph connectivity not euclidean distance<br>
k-means is then run in this embedded space where clusters are now separated linearly

- Graph Laplacian eigenvectors never violate local connectivity but global optimum depends on scale parameters  
- Makes scale sensitivity possible when $\sigma$ is too small all points disconnect, too large everything merges → clustering meaningless

<br>
The original problem isn't solved directly, it's projected into a spectral embedding where geometry matches connectivity
<br><br>

**Spectral clustering learns a geometry from connectivity, then clusters in that learned space.**


---
## Math / Implementation:

**Build similarity graph**:
1) similarity function (RBF kernel $W_{ij} = \exp(\frac{-||x_i - x_j||^2}{2\sigma^2})$)
2) adjacency matrix $(W)$ stores pairwise similarities  
3) degree matrix $(D_{ii} = \sum_j W_{ij})$
   
<br>

**Graph Laplacian** →<br>      

 $L = D - W$ 

This penalizes assigning different values to strongly connected nodes

**For the embedding:**

- Compute first k eigenvectors of L with smallest eigenvalues  
- Each point gets a new representation using these eigenvectors as coordinates

Result would be an embedding where graph-connected points are close in euclidean space
<br> 

**Run k-means:**
- Apply standard k-means on the embedded points  
- Assign cluster labels from k-means back to original points

The scale parameter ($\sigma$ / $\epsilon$ / k) controls what counts as connected, stop according to eigengap 1 or fixed k