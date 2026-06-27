# Quick Reference Card - Convergence Experiments on SO(3)/SO(2)

**Version:** Journal-Ready v1.0  
**Date:** October 30, 2025  
**Location:** `successful_exp3/`

---

## 🎯 Predicted Convergence Slopes

### At Critical Point (Identity, Vanishing Gradient)

| Kernel | Slope | Why? |
|--------|-------|------|
| **Euclidean** | **-0.75** | Generic -1.25 + vanishing gradient adjustment +0.5 |
| **Min-Orbit** | **-0.5 to -0.6** | Generic -1.0 + vanishing gradient adjustment +0.5 |
| **Integral** | **-0.5** | Generic -1.0 + vanishing gradient adjustment +0.5 |

**Key Point:** Less negative = better convergence!

### Why the Adjustment?

- The test function $f(R) = R_{33}$ has $\nabla f(I_3) = 0$ (vanishing gradient)
- This reduces the effective convergence order
- Adjustment: **+0.5** to all slopes (from Singer's equation 3.17)

---

## 🔧 Critical Implementation Details

### Kernel Normalization (MUST USE THIS!)

```python
K(R, R0, ε) = exp(-||R - R0/√2||²_F / (2*ε))
```

**Why 2ε?** Because error formula is:
```python
Error = (2/ε) * L̂f + 2
```

The factor of 2 in numerator MUST match kernel denominator!

**If you change this, experiments won't converge!**

### Error Computation

```python
# Compute graph Laplacian
L_hat = (sum(w_i * f_i) / sum(w_i)) - f(R0)

# Scale to Laplace-Beltrami
Delta_hat = (2/ε) * L_hat

# Error (should be ≈ 0)
error = Delta_hat + 2  # Since Δf = -2f and f(I3) = 1
```

---

## 📊 Expected Results

### Log-Log Plot Appearance

```
  Error
   |
   |     \
   |      \___    <- Minimum at optimal ε
   |          \___
   |              \___
   |__________________|_____ log(ε)
   
   Variance      Optimal     Bias
   Regime        Region      Regime
   (slope ≈ -0.5             (slope > 0)
    or -0.75)
```

### Slope Fitting

- **Fit on variance regime only** (before minimum)
- Regression: slope = d(log Error) / d(log ε)
- Expected difference: Euclidean vs G-invariant ≈ 0.25

---

## 🧮 Mathematical Framework

### The Manifold
- $M = SO(3)$ with Frobenius metric
- $G = SO(2)$ acting on the right
- Quotient: $M/G \cong S^2$

### Test Function
- $f(R) = R_{33} = P_1^{00}(R)$
- Eigenvalue: $\lambda = 2$ (since $\ell = 1$)
- **Critical property:** $\nabla f(I_3) = 0$

### Distance Formula
$$\|R - S\|_F^2 = 8\sin^2(\theta/2)$$

where $\theta$ is the geodesic rotation angle.

---

## 🚀 Running Experiments

### Quick Start
```bash
cd successful_exp3/
./run_experiment.sh
```

### Direct Execution
```bash
python converges_so3_so2_exp_journal.py
```

### Replot Only
```bash
python replot_from_pickle.py
```

---

## 📖 Documentation Files

| File | Purpose | Size |
|------|---------|------|
| **TECHNICAL_DOCUMENTATION.md** | Complete mathematical theory | 37 KB |
| **README.md** | Usage guide | 4 KB |
| **INDEX.md** | Quick reference | 6 KB |
| **CHANGES_AND_IMPROVEMENTS.md** | What's new | 7 KB |

---

## ✅ Validation Checklist

After running experiments, verify:

- [ ] Euclidean slope ≈ -0.75
- [ ] G-invariant slopes ≈ -0.5
- [ ] Slope difference ≈ 0.25
- [ ] U-shaped error curves
- [ ] Minimum at intermediate ε
- [ ] G-invariant error < Euclidean error

---

## 📚 Key References

1. **Vanishing Gradient Effect:** Amit Singer's paper, after equation (3.17)
2. **Kernel Normalization:** Section 4.5 of TECHNICAL_DOCUMENTATION.md
3. **Slope Predictions:** Section 8.1.5 of TECHNICAL_DOCUMENTATION.md

---

## ⚠️ Common Pitfalls

1. **Wrong kernel normalization:** Using ε instead of 2ε → won't converge
2. **Wrong error formula:** Not including the 2/ε factor → wrong slopes
3. **Fitting on wrong regime:** Fitting on bias regime → incorrect slopes
4. **Insufficient SO(2) samples:** m < 200 → G-invariant kernels degrade
5. **Too few trials:** N < 100 → noisy slope estimates

---

## 🎓 For Applied Math Researchers

### What This Validates

1. **G-invariant kernels improve Laplacian estimation** on quotient spaces
2. **Second-order convergence** (slope -0.5) vs. first-order (slope -0.75)
3. **Vanishing gradient effect** reduces slope magnitude by 0.5
4. **Theoretical predictions match numerical experiments**

### Novel Contributions

- Explicit convergence rates at critical points
- Quantitative validation of Singer's theory
- Practical demonstration of quotient space geometry
- G-invariant kernel construction on SO(3)/SO(2)

---

**Last Updated:** October 30, 2025  
**Maintained By:** Applied Mathematics Research Group






