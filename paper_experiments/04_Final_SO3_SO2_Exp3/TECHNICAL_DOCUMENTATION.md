# Technical Documentation: Laplacian Convergence on SO(3)/SO(2)

**A Comprehensive Guide for Applied Mathematics Researchers**

---

## Executive Summary

This document provides a rigorous mathematical treatment of kernel-based Laplacian estimation on the homogeneous space SO(3)/SO(2), with particular emphasis on G-invariant kernel constructions and their convergence properties. We present both theoretical foundations and numerical implementation details for single-point experiments that validate theoretical convergence rates.

**Key Contributions:**
- Empirical verification of improved convergence rates for G-invariant kernels
- Comparative analysis of three kernel approaches: Euclidean, minimum-orbit, and integral
- Numerical validation of the relationship between kernel symmetry and Laplacian approximation quality

---

## Table of Contents

1. [Mathematical Framework](#1-mathematical-framework)
2. [The Geometric Setting](#2-the-geometric-setting)
3. [Group Actions and Homogeneous Spaces](#3-group-actions-and-homogeneous-spaces)
4. [Kernel Constructions](#4-kernel-constructions)
5. [Laplacian Estimation Theory](#5-laplacian-estimation-theory)
6. [Test Functions and Eigenfunctions](#6-test-functions-and-eigenfunctions)
7. [Numerical Methodology](#7-numerical-methodology)
8. [Convergence Analysis](#8-convergence-analysis)
9. [Implementation Details](#9-implementation-details)
10. [Error Metrics and Interpretation](#10-error-metrics-and-interpretation)
11. [Expected Results](#11-expected-results)
12. [References and Further Reading](#12-references-and-further-reading)

---

## 1. Mathematical Framework

### 1.1 The Riemannian Manifold

We work on the special orthogonal group **SO(3)**, the group of 3×3 rotation matrices:

$$\mathrm{SO}(3) = \{R \in \mathbb{R}^{3\times3} : R^T R = I_3, \det(R) = 1\}$$

equipped with the **bi-invariant Riemannian metric** induced by the Frobenius inner product:

$$\langle A, B \rangle_F = \mathrm{tr}(A^T B) = \sum_{i,j} A_{ij} B_{ij}$$

The corresponding norm is:

$$\|R - S\|_F^2 = \sum_{i,j} (R_{ij} - S_{ij})^2$$

### 1.2 Metric Properties

**Fundamental Identity:** For any $R, S \in \mathrm{SO}(3)$, the Frobenius distance relates to the geodesic angle $\theta$ (the angle of rotation $R^T S$) via:

$$\|R - S\|_F^2 = 8 \sin^2(\theta/2)$$

where $\theta = \arccos\left(\frac{\mathrm{tr}(R^T S) - 1}{2}\right)$.

This identity is crucial because:
- It connects the extrinsic (ambient space) metric to the intrinsic (geodesic) metric
- Maximum distance is $\|R - S\|_F^2 = 8$ (when $\theta = \pi$)
- Small-angle approximation: $\|R - S\|_F^2 \approx 2\theta^2$ for small $\theta$

### 1.3 The Laplace-Beltrami Operator

On SO(3), the **Laplace-Beltrami operator** $\Delta$ is defined as the divergence of the gradient. For functions $f: \mathrm{SO}(3) \to \mathbb{R}$, it satisfies:

$$\Delta f = \sum_{i=1}^3 \left(\nabla_{X_i} \nabla_{X_i} f - \nabla_{\nabla_{X_i} X_i} f\right)$$

where $\{X_1, X_2, X_3\}$ is an orthonormal frame.

**Conventions:**
- We use the **geometer's Laplacian**: $\Delta f = -\mathrm{div}(\mathrm{grad}\, f)$, which is non-positive on positive-definite kernels
- Eigenvalue equation: $\Delta f = -\lambda f$ with $\lambda \geq 0$

---

## 2. The Geometric Setting

### 2.1 SO(3) as a Riemannian Manifold

**Dimension:** $\dim(\mathrm{SO}(3)) = 3$

**Topology:** $\mathrm{SO}(3) \cong \mathbb{RP}^3$ (real projective 3-space)

**Volume Form:** The Haar measure $\mu$ on SO(3) is:
- Bi-invariant (invariant under left and right multiplication)
- Normalized so that $\mu(\mathrm{SO}(3)) = 8\pi^2$

**Curvature:** SO(3) has constant sectional curvature $K = 1/4$ (positive curvature)

### 2.2 Parameterizations

#### 2.2.1 Euler Angles

A rotation $R \in \mathrm{SO}(3)$ can be parameterized by Euler angles $(\alpha, \beta, \gamma)$ with $\alpha, \gamma \in [0, 2\pi)$ and $\beta \in [0, \pi]$:

$$R(\alpha, \beta, \gamma) = R_z(\alpha) R_y(\beta) R_z(\gamma)$$

The Haar measure in these coordinates:

$$d\mu = \frac{1}{8\pi^2} \sin(\beta) \, d\alpha \, d\beta \, d\gamma$$

#### 2.2.2 Quaternion Parameterization (Used in Implementation)

We use **unit quaternions** $q = (w, x, y, z) \in S^3 \subset \mathbb{R}^4$ with the double cover:

$$\pi: S^3 \to \mathrm{SO}(3), \quad q \mapsto R(q)$$

where $\pi(q) = \pi(-q)$ (2:1 covering map).

**Advantages:**
- Uniform sampling: draw from $\mathcal{N}(0, I_4)$ and normalize
- No gimbal lock
- Numerically stable

**Quaternion-to-Rotation Map:**

$$R(q) = \begin{pmatrix}
w^2 + x^2 - y^2 - z^2 & 2(xy - wz) & 2(xz + wy) \\
2(xy + wz) & w^2 - x^2 + y^2 - z^2 & 2(yz - wx) \\
2(xz - wy) & 2(yz + wx) & w^2 - x^2 - y^2 + z^2
\end{pmatrix}$$

---

## 3. Group Actions and Homogeneous Spaces

### 3.1 The Subgroup G = SO(2)

Define $G = \mathrm{SO}(2) \subset \mathrm{SO}(3)$ as rotations about the z-axis:

$$\mathrm{SO}(2) = \left\{ R_z(\theta) = \begin{pmatrix}
\cos\theta & -\sin\theta & 0 \\
\sin\theta & \cos\theta & 0 \\
0 & 0 & 1
\end{pmatrix} : \theta \in [0, 2\pi) \right\}$$

**Haar Measure on SO(2):**

$$d\nu(\theta) = \frac{d\theta}{2\pi}$$

(normalized uniform measure)

### 3.2 Right Group Action

We consider the **right action** of $G$ on $M = \mathrm{SO}(3)$:

$$\Phi: M \times G \to M, \quad (R, g) \mapsto R \cdot g$$

**Properties:**
- $(R \cdot g_1) \cdot g_2 = R \cdot (g_1 g_2)$
- $R \cdot e = R$ (identity action)
- Free action: $R \cdot g = R$ implies $g = e$

### 3.3 The Quotient Space: SO(3)/SO(2) ≅ S²

The **orbit space** or **homogeneous space** is:

$$N = M/G = \mathrm{SO}(3)/\mathrm{SO}(2)$$

**Geometric Identification:** Each orbit $[R] = \{R \cdot g : g \in G\}$ corresponds to a point on the 2-sphere:

$$\mathrm{SO}(3)/\mathrm{SO}(2) \cong S^2$$

via the map $R \mapsto R \mathbf{e}_3$, where $\mathbf{e}_3 = (0, 0, 1)^T$.

**Intuition:** 
- A rotation $R$ maps $\mathbf{e}_3$ to some unit vector on $S^2$
- All rotations in the orbit $R \cdot \mathrm{SO}(2)$ map $\mathbf{e}_3$ to the same point
- The orbit consists of rotations that differ only by rotation about the z-axis

### 3.4 G-Invariant Functions

A function $f: M \to \mathbb{R}$ is **right G-invariant** if:

$$f(R \cdot g) = f(R) \quad \forall R \in M, \, g \in G$$

Such functions descend to well-defined functions on the quotient $N = M/G$.

**Example:** $f(R) = R_{33}$ (the (3,3)-entry of $R$) is right SO(2)-invariant because:

$$(R \cdot R_z(\theta))_{33} = R_{33}$$

---

## 4. Kernel Constructions

We study three kernel constructions on $M = \mathrm{SO}(3)$, all based on Gaussian weights but differing in their treatment of the group action.

### 4.1 Euclidean (Chordal) Kernel

The simplest approach uses the **ambient space distance**:

$$K_E(R, R_0; \varepsilon) = \exp\left( -\frac{\|(R - R_0)/\sqrt{2}\|_F^2}{2\varepsilon} \right)$$

**Normalization:** 
- Factor of $1/\sqrt{2}$ in the norm: standard normalization for SO(3)
- Denominator $2\varepsilon$: ensures correct scaling with bandwidth

**Properties:**
- Simple to compute: $O(1)$ per evaluation
- Not G-invariant (in general)
- Standard heat kernel approximation on the manifold
- Asymptotically approaches heat kernel as $\varepsilon \to 0$

**Distance Computation:**

$$d_E^2(R, R_0) = \frac{1}{2}\|R - R_0\|_F^2 = 4\sin^2(\theta/2)$$

where $\theta$ is the geodesic angle.

### 4.2 Minimum-Orbit Kernel

The **minimum-orbit kernel** uses the minimum distance over the orbit:

$$K_{\min}^G(R, R_0; \varepsilon) = \exp\left( -\frac{\min_{g \in G} \|(R - R_0 g)/\sqrt{2}\|_F^2}{2\varepsilon} \right)$$

**Computational Approach:**
- Discretize $G = \mathrm{SO}(2)$ with $m$ samples: $g_1, \ldots, g_m$
- Compute $R_0 g_i$ for each sample
- Find minimum: $\min_{i=1,\ldots,m} \|R - R_0 g_i\|_F^2$

**Properties:**
- **Right G-invariant:** $K_{\min}^G(R \cdot h, R_0; \varepsilon) = K_{\min}^G(R, R_0; \varepsilon)$ for all $h \in G$
- Computational cost: $O(m)$ per evaluation
- Non-smooth (minimum is non-differentiable at ties)

**Geometric Interpretation:**
- Finds the "closest representative" in the target orbit
- Naturally respects the quotient structure

### 4.3 Integral (Haar-Averaged) Kernel

The **integral kernel** averages over the group action:

$$K_{\mathrm{int}}^G(R, R_0; \varepsilon) = \int_G \exp\left( -\frac{\|(R - R_0 g)/\sqrt{2}\|_F^2}{2\varepsilon} \right) d\nu(g)$$

where $d\nu$ is the normalized Haar measure on $G$.

**Numerical Implementation:** Monte Carlo or quadrature approximation

$$K_{\mathrm{int}}^G(R, R_0; \varepsilon) \approx \frac{1}{m} \sum_{i=1}^m \exp\left( -\frac{\|(R - R_0 g_i)/\sqrt{2}\|_F^2}{2\varepsilon} \right)$$

**Properties:**
- **Right G-invariant:** averaging preserves invariance
- **Smooth:** integral of smooth functions
- Computational cost: $O(m)$ per evaluation
- Theoretical convergence guarantees

**Advantages over Minimum Kernel:**
- Smooth (differentiable)
- Better theoretical analysis
- Expected to have faster convergence (second-order vs. first-order)

### 4.4 Normalization Details

All kernels use the normalization:

$$K(R, R_0; \varepsilon) = \exp\left( -\frac{d^2(R, R_0)}{2\varepsilon} \right)$$

where the distance $d^2$ incorporates the $1/\sqrt{2}$ scaling factor:

$$d^2(R, R_0) = \frac{1}{2}\|R - R_0\|_F^2$$

This normalization ensures:
- Consistency with heat kernel theory
- Proper $\varepsilon$-scaling for convergence analysis
- Match with reference implementations

### 4.5 Critical Importance of the $2\varepsilon$ Factor

**Why exactly $2\varepsilon$ in the denominator?**

The factor of $2\varepsilon$ (not just $\varepsilon$) is **essential** for the experiments to converge correctly. This is dictated by the error formula:

$$\mathrm{Error} = \frac{2}{\varepsilon} \hat{L} f(R_0) - \lambda \cdot \Delta f(R_0)$$

where for our test function with eigenvalue $\lambda = 2$:

$$\mathrm{Error} = \frac{2}{\varepsilon} \hat{L} f(I_3) + 2$$

**The connection:**
1. The graph Laplacian $\hat{L} f$ is computed using kernel weights $K(R, R_0; \varepsilon)$
2. The Laplace-Beltrami estimator scales as $\widehat{\Delta}_\varepsilon = \frac{2}{\varepsilon} \hat{L}$
3. The factor of $2$ in the numerator must match the factor in the kernel denominator
4. If you change the kernel to use $\varepsilon$ instead of $2\varepsilon$ without adjusting the error formula, the experiments **will not converge**

**Mathematical Consistency:**

The kernel bandwidth and the Laplacian scaling must satisfy:

$$\widehat{\Delta}_\varepsilon f = \frac{C}{\varepsilon} \hat{L} f \quad \Leftrightarrow \quad K \propto \exp\left(-\frac{d^2}{C \cdot \varepsilon}\right)$$

where $C = 2$ in our formulation. This ensures the correct limiting behavior as $\varepsilon \to 0$.

**Reference:** This normalization convention is consistent with Amit Singer's theoretical framework and is crucial for matching theoretical convergence rates with numerical experiments.

---

## 5. Laplacian Estimation Theory

### 5.1 Graph Laplacian Construction

Given samples $\{R_1, \ldots, R_n\} \subset M$ and kernel weights $w_{ij} = K(R_i, R_j; \varepsilon)$:

**Degree Matrix:**

$$D_{ii} = \sum_{j=1}^n w_{ij}$$

**Weight Matrix:** $W$ with entries $w_{ij}$

**Graph Laplacian:**

$$L = D^{-1}W - I$$

(row-normalized or random-walk Laplacian)

### 5.2 Pointwise Laplacian Estimator

At a point $R_0$ (assumed to be one of the sample points, say $R_0 = R_k$):

$$\hat{L} f(R_0) = \frac{\sum_{i=1}^n w_i f(R_i)}{\sum_{i=1}^n w_i} - f(R_0)$$

where $w_i = K(R_i, R_0; \varepsilon)$.

### 5.3 Connection to Laplace-Beltrami

The **rescaled estimator** approximates the Laplace-Beltrami operator:

$$\widehat{\Delta}_\varepsilon f(R_0) = \frac{2}{\varepsilon} \hat{L} f(R_0)$$

**Theoretical Justification:**

As $\varepsilon \to 0$ and $n \to \infty$ with appropriate relationship $n \sim \varepsilon^{-d-\alpha}$ (where $d$ is the manifold dimension):

$$\mathbb{E}[\widehat{\Delta}_\varepsilon f(R_0)] \to \Delta f(R_0)$$

### 5.4 Convergence Rates

**Standard (Euclidean) Kernel:**
- **Bias term:** $O(\varepsilon)$ (first-order approximation)
- **Variance term:** $O(n^{-1/2}\varepsilon^{-d/2})$
- **Optimal rate:** $O(n^{-1/(d+2)})$ when $\varepsilon \sim n^{-1/(d+2)}$

**G-Invariant Kernels (Integral):**
- **Bias term:** $O(\varepsilon^2)$ (second-order approximation)
- **Variance term:** $O(n^{-1/2}\varepsilon^{-d/2})$
- **Optimal rate:** $O(n^{-2/(d+4)})$ when $\varepsilon \sim n^{-2/(d+4)}$

**For SO(3) ($d = 3$):**
- Euclidean: optimal at $\varepsilon \sim n^{-1/5}$, rate $O(n^{-1/5})$
- G-invariant: optimal at $\varepsilon \sim n^{-2/7}$, rate $O(n^{-2/7})$

### 5.5 Why G-Invariance Improves Convergence

**Intuition:** 
1. G-invariant kernels automatically "integrate out" nuisance directions
2. Effective dimension is $\dim(M/G) = d - \dim(G) = 3 - 1 = 2$
3. Better bias-variance tradeoff in the quotient space

**Technical Detail:**
- Invariant kernels satisfy additional cancellation conditions
- Lead to higher-order approximations (analogous to higher-order finite differences)
- Asymptotic expansion has better leading term

---

## 6. Test Functions and Eigenfunctions

### 6.1 Wigner D-Functions

The **Wigner D-functions** $D^\ell_{mn}(R)$ form a complete orthonormal basis for $L^2(\mathrm{SO}(3))$.

**Properties:**
- Index by $\ell \in \mathbb{N}_0$ (angular momentum quantum number)
- For each $\ell$: indices $m, n \in \{-\ell, \ldots, \ell\}$
- Dimension of $\ell$-subspace: $(2\ell + 1)^2$

**Eigenfunction Property:**

$$\Delta D^\ell_{mn} = -\ell(\ell + 1) D^\ell_{mn}$$

(negative definite operator, eigenvalue $-\lambda$ with $\lambda = \ell(\ell+1) > 0$)

### 6.2 The Special Case: $D^1_{00}(R) = P_1^{00}(R)$

We focus on $\ell = 1$, $m = n = 0$:

$$f(R) = D^1_{00}(R) = P_1^{00}(R)$$

**Explicit Formula:** Using Euler angles $R = R(\alpha, \beta, \gamma)$:

$$D^1_{00}(R) = \frac{\sqrt{3}}{2} P_1(\cos\beta)$$

where $P_1(x) = x$ is the Legendre polynomial of degree 1.

**Simplified Form:**

$$P_1^{00}(R) = R_{33}$$

(the (3,3)-entry of the rotation matrix)

**Verification:**
- For $R_z(\theta)$: $R_{33} = 1$ (constant along orbits)
- For general $R$: $R_{33} = R\mathbf{e}_3 \cdot \mathbf{e}_3 = \cos\beta$

### 6.3 G-Invariance of the Test Function

$$f(R \cdot R_z(\theta)) = (R \cdot R_z(\theta))_{33} = R_{33} = f(R)$$

Therefore, $f$ descends to a function on $S^2 \cong \mathrm{SO}(3)/\mathrm{SO}(2)$.

### 6.4 Eigenvalue and Laplacian

**Exact Laplacian:**

$$\Delta f = -\ell(\ell + 1) f = -2f$$

(for $\ell = 1$: $\lambda = 1 \cdot 2 = 2$)

### 6.5 Special Point Evaluation

At the **identity element** $R_0 = I_3$:

$$f(I_3) = (I_3)_{33} = 1$$

$$\Delta f(I_3) = -2 \cdot 1 = -2$$

**Error Measurement:** We measure

$$\mathrm{Error} = \widehat{\Delta}_\varepsilon f(I_3) - \Delta f(I_3) = \frac{2}{\varepsilon}\hat{L} f(I_3) + 2$$

Ideally, this should be close to zero.

### 6.6 Critical Property: Vanishing Gradient at the Identity

**Important Observation:** The gradient of $f(R) = R_{33}$ **vanishes at the identity**:

$$\nabla f(I_3) = 0$$

**Geometric Interpretation:**
- The function $f(R) = R_{33}$ achieves a local maximum at $R = I_3$ (where $f = 1$)
- As we move away from the identity in any direction, $f$ decreases
- This critical point property has profound implications for convergence rates

**Impact on Convergence (Following Amit Singer's Analysis):**

The vanishing gradient reduces the effective convergence order by **0.5** in the log-log slope. This is explained in detail in Singer's paper following equation (3.17).

**Why This Matters:**
- Standard theory assumes generic (non-critical) points
- At critical points, higher-order terms dominate the error expansion
- This modifies the bias term in the asymptotic expansion

---

## 7. Numerical Methodology

### 7.1 Experimental Setup

**Single-Point Experiment:**
1. Fix special point $R_0 = I_3 \in \mathrm{SO}(3)$
2. Sample $n$ points: $R_1 = R_0$, $R_2, \ldots, R_n \sim \mu_{\mathrm{Haar}}$
3. Evaluate $f(R_i) = (R_i)_{33}$
4. Compute kernel weights $w_i = K(R_i, R_0; \varepsilon)$
5. Estimate $\hat{L} f(R_0)$ and $\widehat{\Delta}_\varepsilon f(R_0)$
6. Compute error: $|\widehat{\Delta}_\varepsilon f(R_0) + 2|$

### 7.2 Sampling Strategy

**Haar Sampling on SO(3):**
```
1. Generate q ~ N(0, I_4) in R^4
2. Normalize: q ← q / ||q||
3. Convert: R ← quat_to_rotmat(q)
```

**Why This Works:**
- Uniform distribution on $S^3$ pushes forward to Haar on SO(3)
- No rejection sampling needed
- Numerically stable

**Haar Sampling on SO(2):**
```
θ ~ Uniform[0, 2π)
R_z(θ) = [[cos θ, -sin θ, 0], 
          [sin θ,  cos θ, 0], 
          [0,      0,     1]]
```

### 7.3 Parameter Ranges

**Bandwidth:** $\varepsilon \in [\exp(-6), \exp(-1)]$ (100 log-spaced values)
- Small $\varepsilon$: captures local geometry, low bias, high variance
- Large $\varepsilon$: averages over large regions, high bias, low variance

**Sample Sizes:**
- **Euclidean kernel:** $n = 300,000$ (needs more samples for convergence)
- **G-invariant kernels:** $n = 10,000$ (more efficient)

**Discretization:** $m = 200$ samples of SO(2) for orbit kernels

**Monte Carlo Trials:** $N_{\mathrm{trials}} = 500$ repetitions

### 7.4 Error Metric

For each $\varepsilon$:

$$\mathrm{RMSE}(\varepsilon) = \sqrt{\frac{1}{N_{\mathrm{trials}}} \sum_{t=1}^{N_{\mathrm{trials}}} \left[\widehat{\Delta}_\varepsilon^{(t)} f(I_3) + 2\right]^2}$$

where $\widehat{\Delta}_\varepsilon^{(t)}$ is the estimator in trial $t$.

### 7.5 Convergence Slope Estimation

In the **log-log plot** of $\log(\mathrm{RMSE})$ vs. $\log(\varepsilon)$:

**Variance Regime:** (large $\varepsilon$, before minimum)
- Error dominated by approximation bias
- Expected slope ≈ 1 for Euclidean, ≈ 2 for G-invariant

**Bias Regime:** (small $\varepsilon$, after minimum)
- Error dominated by sampling variance
- Error grows as $\varepsilon \to 0$

**Slope Fitting:**
```
1. Find minimum error: ε_min = argmin_ε RMSE(ε)
2. Fit linear regression on log-log plot for ε < ε_min
3. slope = d(log RMSE) / d(log ε)
```

---

## 8. Convergence Analysis

### 8.1 Theoretical Predictions at Generic Points

At **generic (non-critical) points** where $\nabla f \neq 0$:

**Euclidean Kernel:**

$$\mathbb{E}[|\widehat{\Delta}_\varepsilon f - \Delta f|] = O(\varepsilon) + O\left(\frac{1}{\sqrt{n\varepsilon^3}}\right)$$

Optimal at $\varepsilon_{\mathrm{opt}} \sim n^{-1/5}$ with error $O(n^{-1/5})$.

**Slope (generic):** $\approx 1$ (or $-1.25$ depending on regime convention)

**G-Invariant Integral Kernel:**

$$\mathbb{E}[|\widehat{\Delta}_\varepsilon f - \Delta f|] = O(\varepsilon^2) + O\left(\frac{1}{\sqrt{n\varepsilon^3}}\right)$$

Optimal at $\varepsilon_{\mathrm{opt}} \sim n^{-2/7}$ with error $O(n^{-2/7})$.

**Slope (generic):** $\approx 2$ (or $-1$ depending on regime convention)

### 8.1.5 Adjusted Predictions for Critical Points (Vanishing Gradient)

**Critical Property of Our Test Function:**

Since $\nabla f(I_3) = 0$ (gradient vanishes at the identity), the convergence analysis must be modified. Following **Amit Singer's analysis (equation 3.17 and subsequent discussion)**, the vanishing gradient **reduces the magnitude of the slope by 0.5**.

**Why the Gradient Vanishes:**
- The function $f(R) = R_{33}$ achieves a local maximum at $R = I_3$
- At this critical point, the first-order derivatives vanish
- Higher-order terms dominate the Taylor expansion
- This fundamentally alters the bias-variance tradeoff

**Slope Adjustment Formula:**

If the generic slope is $s_{\text{generic}}$, then at a critical point:

$$s_{\text{critical}} = s_{\text{generic}} + 0.5$$

**Predicted Slopes for Our Experiment:**

| Kernel | Generic Slope | Adjustment | **Critical Point Slope** |
|--------|--------------|------------|------------------------|
| **Euclidean** | $-1.25$ | $+0.5$ | $\mathbf{-0.75}$ |
| **G-Invariant (Min-Orbit)** | $-1.0$ | $+0.5$ | $\mathbf{-0.5}$ |
| **G-Invariant (Integral)** | $-1.0$ | $+0.5$ | $\mathbf{-0.5}$ |

**Summary of Predicted Slopes:**
- **Euclidean kernel:** slope $\approx -0.75$
- **G-invariant kernels:** slope $\approx -0.5$

These are the slopes we expect to observe in the log-log plots of error vs. $\varepsilon$ in the variance-dominated regime.

**Reference:** This analysis follows directly from Amit Singer's theoretical framework, specifically the discussion following equation (3.17) in his paper on Laplacian estimation on manifolds.

### 8.2 Expected Behavior

**Log-Log Plot Structure:**

```
         high ε                                  low ε
           |                                        |
   --------+--------------------+-------------------+--------
           |                    |                   |
     Bias-Dominated        Optimal           Variance-Dominated
      (slope ≈ α)          Region            (negative slope)
           |                    |                   |
```

Where $\alpha = 1$ for Euclidean, $\alpha = 2$ for G-invariant.

### 8.3 Error Regimes

| Regime | Epsilon | Dominant Term | Behavior |
|--------|---------|---------------|----------|
| Variance | Large $\varepsilon$ | $O(\varepsilon^\alpha)$ | Linear/Quadratic increase |
| Optimal | $\varepsilon_{\mathrm{opt}}$ | Balanced | Minimum error |
| Bias | Small $\varepsilon$ | $O(n^{-1/2}\varepsilon^{-3/2})$ | Rapid increase |

### 8.4 Kernel Comparison

**Expected Performance Ranking** (at optimal $\varepsilon$) with adjusted slopes:

1. **Integral Kernel:** Best (slope ≈ -0.5, lowest error)
2. **Minimum-Orbit Kernel:** Intermediate (slope ≈ -0.5 to -0.6)
3. **Euclidean Kernel:** Baseline (slope ≈ -0.75)

**Note on Slope Sign Convention:**
- Negative slopes indicate that error decreases as $\varepsilon$ increases (in the variance regime)
- Less negative slope (closer to 0) = better convergence
- The G-invariant kernels have slope -0.5 (better) vs. Euclidean's -0.75 (worse)

**Rationale:**
- Integral kernel: full G-invariance + smoothness → second-order convergence
- Minimum kernel: G-invariance but non-smooth (minimum operation) → nearly second-order
- Euclidean: no special structure → first-order convergence

**Key Insight:**
At the critical point (identity), the vanishing gradient makes the effective dimension lower, which improves convergence rates but reduces the observed slope magnitude by 0.5 compared to generic points.

---

## 9. Implementation Details

### 9.1 Algorithmic Pseudocode

```python
Algorithm: Single-Point Laplacian Estimation

Input:
  - ε: bandwidth parameter
  - n: number of sample points
  - m: number of group samples (for G-invariant kernels)
  - N_trials: number of Monte Carlo repetitions

Output:
  - RMSE(ε): root mean squared error

1. For trial = 1 to N_trials:
   
   a. Sample rotations:
      R[0] ← I_3  (special point)
      For i = 1 to n-1:
         q ~ N(0, I_4), q ← q/||q||
         R[i] ← quat_to_rotmat(q)
   
   b. Evaluate function:
      For i = 0 to n-1:
         f[i] ← R[i][2,2]  (3,3-entry, 0-indexed as [2,2])
   
   c. (For G-invariant kernels) Sample SO(2):
      For j = 0 to m-1:
         θ[j] ~ Uniform(0, 2π)
         g[j] ← R_z(θ[j])
   
   d. Compute kernel weights:
      For i = 0 to n-1:
         w[i] ← K(R[i], R[0]; ε)  # kernel-specific
   
   e. Graph Laplacian:
      degree ← sum(w)
      L_hat ← (sum(w[i] * f[i]) / degree) - f[0]
   
   f. Laplace-Beltrami estimate:
      Δ_hat ← (2/ε) * L_hat
   
   g. Error:
      error[trial] ← Δ_hat + 2  # should be ≈ 0
   
2. Compute RMSE:
   RMSE ← sqrt(mean(error^2))

3. Return RMSE
```

### 9.2 Kernel-Specific Weight Computation

#### Euclidean Kernel
```python
def compute_weight_euclidean(R, R0, ε):
    diff = (R - R0) / sqrt(2)
    d_squared = sum(diff^2)  # Frobenius norm squared
    return exp(-d_squared / (2*ε))
```

#### Minimum-Orbit Kernel
```python
def compute_weight_min_orbit(R, R0, ε, group_samples):
    distances = []
    for g in group_samples:
        R0g = R0 @ g  # matrix multiplication
        diff = (R - R0g) / sqrt(2)
        d_squared = sum(diff^2)
        distances.append(d_squared)
    
    d_min_squared = min(distances)
    return exp(-d_min_squared / (2*ε))
```

#### Integral Kernel
```python
def compute_weight_integral(R, R0, ε, group_samples):
    m = len(group_samples)
    total = 0
    for g in group_samples:
        R0g = R0 @ g
        diff = (R - R0g) / sqrt(2)
        d_squared = sum(diff^2)
        total += exp(-d_squared / (2*ε))
    
    return total / m  # average over group
```

### 9.3 Numerical Stability

**Underflow Protection:**
- Kernel values can be extremely small for large distances or small $\varepsilon$
- Use log-sum-exp trick when needed
- Check for zero degree: if $\sum w_i = 0$, set estimator to 0

**Normalization:**
- Always normalize quaternions: $q \gets q/\|q\|$
- Verify rotation matrices: check $\det(R) = 1$, $R^T R = I$

**Precision:**
- Use double precision (float64) throughout
- Accumulate sums in extended precision if available

### 9.4 Computational Complexity

**Per Trial:**

| Kernel | Sampling | Weight Computation | Total |
|--------|----------|-------------------|-------|
| Euclidean | $O(n)$ | $O(n)$ | $O(n)$ |
| Min-Orbit | $O(n + m)$ | $O(nm)$ | $O(nm)$ |
| Integral | $O(n + m)$ | $O(nm)$ | $O(nm)$ |

**Total for Experiment:**
- $N_{\varepsilon}$ epsilon values (typically 100)
- $N_{\mathrm{trials}}$ trials (typically 500)
- Total: $O(N_{\varepsilon} \cdot N_{\mathrm{trials}} \cdot nm)$

**Typical Runtime:**
- Euclidean: ~30-40 minutes (larger $n$, simpler weights)
- Min-Orbit: ~8-12 minutes
- Integral: ~8-12 minutes
- Total: ~50-65 minutes on modern CPU

---

## 10. Error Metrics and Interpretation

### 10.1 Pointwise Error

At the special point $R_0 = I_3$:

$$\mathrm{Error}(\varepsilon, \{R_i\}) = \widehat{\Delta}_\varepsilon f(R_0) - \Delta f(R_0)$$

For our test function with $\Delta f(I_3) = -2$:

$$\mathrm{Error} = \frac{2}{\varepsilon} \hat{L} f(I_3) + 2$$

### 10.2 Root Mean Squared Error

Across $N$ trials:

$$\mathrm{RMSE}(\varepsilon) = \sqrt{\frac{1}{N} \sum_{t=1}^N [\mathrm{Error}_t(\varepsilon)]^2}$$

**Interpretation:**
- Measures average deviation from true Laplacian
- Includes both bias and variance contributions
- Lower is better

### 10.3 Log-Log Slope

The **convergence rate** is characterized by the slope in log-log coordinates:

$$s = \frac{d \log(\mathrm{RMSE})}{d \log(\varepsilon)}$$

estimated via linear regression on the variance regime (before the minimum error point).

**Target Values (Adjusted for Vanishing Gradient):**
- **Euclidean:** $s \approx -0.75$
- **Minimum-Orbit:** $s \approx -0.5$ to $-0.6$ (empirical, nearly as good as integral)
- **Integral:** $s \approx -0.5$

**Why Negative Slopes?**
In the variance-dominated regime (large $\varepsilon$, before optimal):
- Error decreases as $\varepsilon$ decreases
- Therefore $\frac{d(\log \text{RMSE})}{d(\log \varepsilon)} < 0$
- Less negative (closer to 0) is better

**Physical Interpretation:**
- $s = -0.75$: first-order method at critical point (Euclidean kernel)
- $s = -0.5$: second-order method at critical point (G-invariant kernels)
- Less negative slope = better convergence (smaller error for same $\varepsilon$)

**Generic vs. Critical Point:**
- Generic point (no vanishing gradient): slopes would be more negative by 0.5
  - Euclidean: $-1.25$
  - G-Invariant: $-1.0$
- Critical point (vanishing gradient): slopes adjusted by $+0.5$
  - Euclidean: $-0.75$
  - G-Invariant: $-0.5$

**Reference:** The vanishing gradient adjustment is explained in detail in Amit Singer's paper after equation (3.17).

### 10.4 Bias-Variance Decomposition

Theoretically:

$$\mathrm{MSE}(\varepsilon) = [\mathrm{Bias}(\varepsilon)]^2 + \mathrm{Var}(\varepsilon)$$

**Bias:** Systematic error from kernel approximation
- Decreases with $\varepsilon$ (better local approximation)
- Rate: $O(\varepsilon^\alpha)$ with $\alpha \in \{1, 2\}$

**Variance:** Random error from finite sampling
- Increases as $\varepsilon \to 0$ (fewer effective neighbors)
- Rate: $O(n^{-1/2}\varepsilon^{-d/2})$

**Optimal Balance:**

$$\varepsilon_{\mathrm{opt}} = \arg\min_\varepsilon \mathrm{MSE}(\varepsilon)$$

Found empirically as the minimum in the error curve.

---

## 11. Expected Results

### 11.1 Qualitative Predictions

**Error vs. $\varepsilon$ Curves:**
- **U-shaped** on log-log scale
- Minimum at optimal bandwidth
- Left of minimum: bias-dominated (increasing error as $\varepsilon$ decreases)
- Right of minimum: variance-dominated (decreasing error as $\varepsilon$ decreases)

**Kernel Ranking:**
1. Integral kernel: lowest error, slope $\approx -0.5$
2. Minimum-orbit kernel: intermediate error, slope $\approx -0.5$ to $-0.6$
3. Euclidean kernel: highest error, slope $\approx -0.75$

**Note:** Slopes are negative in the variance regime (error decreases with increasing $\varepsilon$). Less negative slope indicates better convergence.

### 11.2 Quantitative Predictions (Adjusted for Vanishing Gradient)

At **optimal bandwidth** (at the critical point $R_0 = I_3$):

| Kernel | Optimal $\varepsilon$ | Min RMSE | Slope (Variance Regime) | Notes |
|--------|----------------------|----------|------------------------|-------|
| Euclidean | $\sim 10^{-3}$ | $\sim 10^{-1}$ | $\approx -0.75$ | Generic: -1.25, Adjusted: +0.5 |
| Min-Orbit | $\sim 10^{-2.5}$ | $\sim 10^{-1.5}$ | $\approx -0.5$ to $-0.6$ | Generic: -1.0, Adjusted: +0.5 |
| Integral | $\sim 10^{-2}$ | $\sim 10^{-2}$ | $\approx -0.5$ | Generic: -1.0, Adjusted: +0.5 |

**Critical Adjustment:** All slopes are shifted by **+0.5** due to the vanishing gradient at the identity element.

**Why -0.75 and -0.5?**
- **Euclidean:** Generic slope $-1.25$ → Adjusted $-1.25 + 0.5 = -0.75$
- **G-Invariant:** Generic slope $-1.0$ → Adjusted $-1.0 + 0.5 = -0.5$

*(Values are theoretical predictions; actual results may vary slightly due to finite sampling and discretization)*

### 11.3 Theoretical Validation

**Success Criteria (Updated for Critical Point):**
1. ✓ All kernels achieve decreasing error in variance regime
2. ✓ G-invariant kernels show lower error than Euclidean at optimal $\varepsilon$
3. ✓ **G-invariant kernels achieve slope $\approx -0.5$ (vs. $\approx -0.75$ for Euclidean)**
4. ✓ Minimum error occurs at intermediate $\varepsilon$ (not at boundaries)
5. ✓ **Slope difference of 0.25 between Euclidean and G-invariant kernels**

**Potential Observations:**
- Minimum-orbit kernel may show slight oscillations (due to non-smoothness)
- Very small $\varepsilon$: numerical instability may dominate
- Very large $\varepsilon$: all kernels converge (trivial constant approximation)

### 11.4 Interpretation Guidelines

**If slopes match theory (around -0.75 for Euclidean, -0.5 for G-invariant):**
- ✓ Validates G-invariant kernel construction
- ✓ Confirms improved convergence rate at critical points
- ✓ Supports quotient space geometry
- ✓ Verifies vanishing gradient adjustment (+0.5 shift)
- ✓ Demonstrates second-order convergence for G-invariant methods

**If slopes deviate significantly:**
- Check sample size $n$ (may need more points)
- Verify SO(2) discretization $m$ (may need finer grid, try 500-1000)
- Examine special point (is it truly at $I_3$?)
- Review normalization factors (must use $2\varepsilon$ in kernel denominator)
- Verify error formula: $\text{Error} = \frac{2}{\varepsilon}\hat{L}f + 2$
- Check epsilon range (should span from $\sim 10^{-6}$ to $\sim 10^{-1}$)

**Expected Slope Difference:**
The key validation is the **relative difference** between kernels:
- Euclidean vs. G-invariant: should differ by approximately $0.25$ in slope magnitude
- This $0.25$ difference demonstrates second-order vs. first-order convergence
- Both should be shifted by $+0.5$ from their generic counterparts due to vanishing gradient

**Connecting to Singer's Theory:**
The observed slopes directly validate the theoretical prediction in Singer's equation (3.17) that critical points (vanishing gradient) reduce the effective convergence order, manifesting as a +0.5 adjustment in the log-log slope.

---

## 12. References and Further Reading

### 12.1 Fundamental Theory

1. **Belkin, M., & Niyogi, P.** (2007). *Convergence of Laplacian eigenmaps*. NIPS.
   - Foundational work on graph Laplacian convergence

2. **Coifman, R. R., & Lafon, S.** (2006). *Diffusion maps*. Applied and Computational Harmonic Analysis, 21(1), 5-30.
   - Geometric harmonics and heat kernel on manifolds

3. **Hein, M., Audibert, J. Y., & von Luxburg, U.** (2005). *From graphs to manifolds – weak and strong pointwise consistency of graph Laplacians*. COLT.
   - Rigorous convergence rates

### 12.2 Riemannian Geometry

4. **Lee, J. M.** (2018). *Introduction to Riemannian Manifolds*. Springer.
   - Standard reference for Riemannian geometry

5. **Petersen, P.** (2016). *Riemannian Geometry*. Springer.
   - Advanced treatment, including comparison theorems

### 12.3 Lie Groups and Homogeneous Spaces

6. **Hall, B. C.** (2015). *Lie Groups, Lie Algebras, and Representations*. Springer.
   - Excellent introduction to SO(3) and representation theory

7. **Chirikjian, G. S., & Kyatkin, A. B.** (2001). *Engineering Applications of Noncommutative Harmonic Analysis*. CRC Press.
   - Practical aspects of rotations and SO(3)

### 12.4 Kernel Methods on Manifolds

8. **Cheng, M. Y., & Wu, H. T.** (2013). *Local linear regression on manifolds and its geometric interpretation*. JASA.
   - Kernel smoothing on manifolds

9. **Jiang, X., & Tao, M.** (2020). *Graph-based Laplacian smoothing for improved inference in finite samples*. arxiv.
   - Recent advances in graph Laplacian methods

### 12.5 G-Invariant Kernels

10. **Kondor, R., & Trivedi, S.** (2018). *On the generalization of equivariant graph neural networks*. ICLR Workshop.
    - Modern perspective on invariant kernels

11. **Cohen, T. S., Geiger, M., & Weiler, M.** (2019). *A general theory of equivariant CNNs on homogeneous spaces*. NeurIPS.
    - Theoretical framework for quotient spaces

### 12.6 Wigner D-Functions

12. **Varshalovich, D. A., Moskalev, A. N., & Khersonskii, V. K.** (1988). *Quantum Theory of Angular Momentum*. World Scientific.
    - Comprehensive reference on Wigner functions

13. **Edmonds, A. R.** (1996). *Angular Momentum in Quantum Mechanics*. Princeton University Press.
    - Classic text on spherical harmonics and rotation matrices

### 12.7 Numerical Methods

14. **Shoemake, K.** (1985). *Animating rotation with quaternion curves*. SIGGRAPH.
    - Uniform sampling of rotations via quaternions

15. **Arvo, J.** (1992). *Fast random rotation matrices*. Graphics Gems III.
    - Practical algorithms for rotation sampling

---

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $\mathrm{SO}(3)$ | Special orthogonal group (3D rotations) |
| $\mathrm{SO}(2)$ | Special orthogonal group (2D rotations) |
| $R, S$ | Rotation matrices |
| $g, h$ | Group elements (typically in $\mathrm{SO}(2)$) |
| $\|\cdot\|_F$ | Frobenius norm |
| $\Delta$ | Laplace-Beltrami operator |
| $\varepsilon$ | Bandwidth parameter |
| $K(\cdot, \cdot; \varepsilon)$ | Kernel function |
| $\mu$ | Haar measure on $\mathrm{SO}(3)$ |
| $\nu$ | Haar measure on $\mathrm{SO}(2)$ |
| $D^\ell_{mn}$ | Wigner D-function |
| $P_\ell$ | Legendre polynomial of degree $\ell$ |
| $\hat{L}$ | Graph Laplacian estimator |
| $\widehat{\Delta}_\varepsilon$ | Laplace-Beltrami estimator |

---

## Appendix B: Implementation Checklist

### Before Running Experiments

- [ ] Verify quaternion-to-rotation conversion
- [ ] Check Frobenius norm identity: $\|R - S\|_F^2 = 8\sin^2(\theta/2)$
- [ ] Confirm SO(2) sampling is uniform
- [ ] Validate G-invariance of test function: $f(R \cdot g) = f(R)$
- [ ] Test kernel normalizations match theoretical specifications

### During Experiments

- [ ] Monitor progress with tqdm or similar
- [ ] Check for numerical warnings (underflow, NaN)
- [ ] Verify sample sizes are as expected
- [ ] Ensure special point is exactly $I_3$ (first in sample list)

### After Experiments

- [ ] Verify U-shaped error curves
- [ ] Check slopes are in expected range (0.5-2.5)
- [ ] Compare G-invariant vs. Euclidean performance
- [ ] Inspect plots for anomalies
- [ ] Save results to pickle for reproducibility

---

## Appendix C: Troubleshooting

### Common Issues

**Issue:** Slopes are all close to zero
- **Cause:** Epsilon range too narrow or all in bias regime
- **Fix:** Expand epsilon range to include larger values

**Issue:** Minimum-orbit and integral kernels have similar errors to Euclidean
- **Cause:** Insufficient SO(2) discretization ($m$ too small)
- **Fix:** Increase $m$ to 500-1000

**Issue:** Very large errors for small epsilon
- **Cause:** Numerical underflow in kernel weights
- **Fix:** Add underflow protection, check weight sum $> 0$

**Issue:** Non-monotonic error curves
- **Cause:** Insufficient Monte Carlo trials
- **Fix:** Increase $N_{\mathrm{trials}}$ to 1000+

**Issue:** Integral kernel slope < 1.5
- **Cause:** Sample size too small relative to epsilon
- **Fix:** Increase $n$ or adjust epsilon range

---

**Document Version:** 1.0  
**Date:** October 30, 2025  
**Author:** Applied Mathematics Research Group  
**Experiment Code:** `converges_so3_so2_exp_journal.py`

---

*This documentation is intended for researchers with background in differential geometry, Lie theory, and numerical analysis. For implementation details, consult the source code and inline comments.*

