#  Theoretical Background

 **unsupervised learning**  focuses on uncovering hidden patterns or structures in data without using labeled outcomes. Since there are no ground truth labels, the goal is not prediction but rather discovering meaningful insights like whether the data naturally separates into subgroups or clusters. This analysis applies several unsupervised learning techniques to explore global patterns in mental health, suicide rates, and economic indicators using Principal Component Analysis (PCA), K-Means clustering, and Hierarchical Clustering.

## Principal Component Analysis

###  Principal Components

**Principal Component Analysis (PCA)** is a technique used to simplify complex datasets by identifying the directions in which the data varies the most. These directions are called **principal components**, and they are always at right angles to each other (orthogonal). The key idea is that while our data might exist in many dimensions (features), not all of them are equally useful. PCA helps us find the most "informative" directions those that capture the most variation so we can reduce the number of dimensions without losing much of the original information.

The **first principal component** captures the greatest variance in the data. The **second principal component** captures the next highest variance, but in a direction that's completely uncorrelated with the first, and so on. This continues for as many dimensions as the data has. Each principal component is defined by a **loading vector**, which tells us how much each original feature contributes to that component. These loadings can highlight important relationships between variables.

The **scores** are the coordinates of each data point in this new PCA space they’re just the original data projected onto the principal components. If we keep only the scores from the first few principal components, we get a low-dimensional version of the data that still keeps the most important patterns. This is useful for things like visualization, removing noise, filling in missing values, or creating new features for machine learning models.


###  Proportion of Variance Explained



An important consideration in principal component analysis is determining how much of the original data’s structure is retained when projecting observations onto a reduced number of principal components. This is addressed by calculating the **proportion of variance explained (PVE)** by each component. The PVE quantifies the fraction of the dataset’s total variance that is captured by a given principal component. Assuming the data has been centered (i.e., each feature has mean zero), the PVE for each component is obtained by dividing the variance of its associated scores by the total variance across all features. This metric helps assess the effectiveness of dimensionality reduction in preserving the essential patterns in the data.
 $$\frac{\sum_{i=1}^n s^2_{im}}{\sum_{j=1}^p \mathbb{V}(X_j)}=\frac{\sum_{i=1}^n \left(\hat{v}_m^\top\vec{x}_i\right)^2}{\frac{1}{n}\sum_{j=1}^p \sum_{i=1}^n x_{ij}^2}$$
The **cumulative proportion of variance explained** (CPVE) of the first $M$ 

The **cumulative proportion of variance explained (CPVE)** refers to the total variance accounted for by the first *M* principal components. It is calculated by summing the individual proportions of variance explained (PVE) for those components. This cumulative measure provides a way to evaluate how much of the dataset’s overall variability is retained when using a subset of the components. By examining the CPVE, one can make an informed decision about how many principal components to keep for tasks like dimensionality reduction or visualization, aiming to balance simplicity with sufficient data representation.


### Singular Value Decomposition



### Singular Value Decomposition ) in PCA
-

The singular value decomposition (SVD) is a powerful matrix factorization method used in unsupervised learning to identify structure in high-dimensional data. It plays a central role in principal component analysis (PCA) and low-rank matrix approximation. Given a data matrix $A \in \mathbb{R}^{m \times n}$, the full SVD factorizes it as $A = U \Sigma V^\top$, where $U$ and $V$ are orthonormal matrices and $\Sigma$ is a diagonal matrix containing the singular values. The reduced SVD drops the columns corresponding to zero singular values, retaining only the significant components. This allows us to construct a low-rank approximation $A_k = U_k \Sigma_k V_k^\top$, which is the best approximation of $A$ in terms of minimizing reconstruction error. In PCA, when the data is standardized, SVD directly gives the principal components and their associated variances. The rows of $V^\top$ provide the directions of maximum variance (the principal components), the diagonal values of $\Sigma$ determine the amount of variance captured by each component, and the product $U \Sigma$ gives the principal component scores, which represent the coordinates of the data points in the new space defined by the principal components. This makes SVD a core tool for dimensionality reduction, data visualization, and low-rank matrix recovery.
ional data.


##  Matrix Completion



In many real-world datasets, missing values are common and must be handled before applying most statistical learning methods. A simple but often inefficient solution is to drop any rows with missing data, which can result in significant information loss. Another common technique is mean imputation, where missing entries in a variable are replaced with the average of the observed values in that column. While easy to implement, this method ignores relationships between variables and can lead to biased estimates.

A more effective strategy, when data is missing at random, is **matrix completion**. This method uses the correlation structure among variables to infer missing values. Specifically, it leverages **principal components** to reconstruct a low-rank approximation of the original matrix, filling in missing entries based on the dominant patterns in the

**Step-by-Step Matrix Completion Algorithm**
m**

1. Start by creating a fully filled-in version of your data matrix $\tilde{X} \in \mathbb{R}^{n \times p}$, where:

   $$
   \tilde{x}_{ij} = 
   \begin{cases}
   x_{ij}, & \text{if } (i,j) \text{ is an observed value} \\
   \bar{x}_j, & \text{if } (i,j) \text{ is missing}
   \end{cases}
   $$

   Here, $\mathcal{O}$ is the set of observed entries, and $\bar{x}_j$ is the mean of all observed values in column $j$.

2. Repeat the following steps until the objective function no longer improves:

   * Use singular value decomposition (SVD) to calculate a rank-$M$ approximation of $\tilde{X}$:
     $\tilde{X}_M = U_M \Sigma_M V^\top_M$
   * Replace the missing entries in $\tilde{X}$ with the corresponding values from $\tilde{X}_M$
   * Calculate the reconstruction error (objective function):
     $\sum_{(i,j) \in \mathcal{O}} (x_{ij} - \tilde{x}_{ij})^2$

3. Once the error stops decreasing, return the completed matrix $\tilde{X}$.
tilde{X}$

##  Clustering Methods

**Clustering** is a foundational technique in unsupervised learning used to uncover underlying structure in data by identifying meaningful subgroups, or *clusters*. The central idea is to partition a dataset into distinct groups such that observations within the same cluster are highly similar to each other, while observations from different clusters are as dissimilar as possible. Clustering is commonly applied when there are no predefined labels, and the goal is to explore patterns or groupings based on feature similarity.


###  K-Means Clustering

**K-means clustering** is a widely used method in unsupervised learning for partitioning a dataset into a predetermined number of clusters. The central goal is to divide the data in such a way that the observations within each cluster are as similar as possible, while those in different clusters are distinct. This similarity is quantified by minimizing the overall **within-cluster variation**, which is the sum of the variations within each cluster:

$$
\sum_k W(C_k)
$$

The within-cluster variation $W(C_k)$ for a given cluster $C_k$ is commonly defined as the **within-cluster sum of squares** (WCSS) normalized by the number of observations in the cluster:

$$
W(C_k) = \frac{1}{\lvert C_k \rvert} \sum_{i, i' \in C_k} \lvert \vec{x}_i - \vec{x}_{i'} \rvert^2
$$

This captures how tightly grouped the observations are within each cluster.

> **K-means Clustering Algorithm**
>
> 1. Randomly assign each observation to one of the $K$ clusters to initialize the cluster labels.
> 2. Repeat until cluster assignments stabilize:
>
>    * Calculate the **centroid** of each cluster, which is the mean of the observations assigned to it.
>    * Reassign each observation to the cluster whose centroid is nearest.

This iterative algorithm converges to a *local minimum* of the within-cluster variation, meaning the final result depends on the initial random assignment. As such, it's standard practice to run the algorithm multiple times with different initializations and choose the clustering result that yields the lowest total within-cluster variation.

A key challenge in using K-means is deciding on the number of clusters, $K$. Since the algorithm will always return some clustering, it's important to assess whether the clusters reflect actual structure or are just artifacts of random variation. A common strategy is to evaluate cluster quality metrics, such as the WCSS or the **silhouette score**, across a range of $K$ values. Visual tools like the **elbow method** can then be used to select an optimal value of $K$ that balances the complexity of the model with the quality of clustering.


###  Hierarchical Clustering

**Hierarchical clustering** is a method used to uncover nested groupings in data without requiring the number of clusters $K$ to be specified in advance. Instead of finding a flat partitioning like K-means, hierarchical clustering builds a tree-like structure known as a **dendrogram**, which represents data grouping at every possible level of granularity. This approach allows analysts to explore clustering structures at multiple resolutions.

In a dendrogram, each observation starts as its own individual cluster, represented by the leaves of the tree. As we move up the tree, similar observations or groups are progressively merged into larger clusters. The height at which two branches fuse reflects how dissimilar those groups are: early (low) fusions indicate high similarity, while late (high) fusions suggest greater dissimilarity. By cutting the dendrogram at a specific height, we can extract a desired number of clusters. A higher cut yields fewer, broader clusters with more internal variation, while a lower cut results in a larger number of tighter, more homogeneous clusters. However, one limitation is that hierarchical clustering assumes nested structue—clusters formed at one level are always containein ihiger levelel cluste s—which may not align with the actual data structure.

The most widely used form is **agglomerative clustering**, a bottom-up approach that begins by treating each observation as its own cluster. At each step, the two most similar clusters are merged until all observations are combined into a single group.

> **Agglomerative Clustering Procedure**
>
> 1. Start with $n$ observations and a distance metric (e.g., Euclidean distance). Each observation begins in its own cluster.
> 2. For $k = n, n - 1, \dots, 2$:
>
>    * Identify the pair of clusters with the smallest dissimilarity and merge them. The level at which this fusion happens defines the height in the dendrogram.
>    * Recalculate the distances between the new cluster and the remaining ones.

A key part of agglomerative clustering is how we define the **dissimilarity between groups** of observations. This is governed by the concept of **linkage**, which determines how distances between clusters are computed. The most common linkage strategies include:

* **Complete Linkage**: Measures the maximum distance between observations in two clusters. It tends to create compact, spherical clusters.
* **Single Linkage**: Measures the minimum distance between observations in two clusters. This can lead to long, chain-like clusters.
* **Average Linkage**: Computes the average of all pairwise distances between points in the two clusters, balancing the influence of outliers.
* **Centroid Linkage**: Uses the distance between cluster centroids, though this can sometimes cause inversions in the dendrogram where clusters appear to merge at lower levels than they should.

The choice of linkage method significantly affects the shape of the resulting dendrogram and should be made based on the data characteristics and analysis goals. In practice, **complete** and **average linkage** are often preferred for producing interpretable and stable clusters.





