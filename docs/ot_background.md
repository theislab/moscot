# Optimal Transport (OT) in a nutshell

Optimal transport ({term}`OT`) is a mathematical framework that finds the most efficient way to transform one probability distribution into another, minimizing a cost function that depends on the distance between points. In the context of single-cell genomics, the distributions are set of cells, which we want to map onto each other. Realigning sets of cells is prevalent in single-cell genomics due to the destructive nature of sequencing technologies.
The solution of an ({term}`OT`) problem is given by a {term}`transport matrix` $\mathbf{P} \in \mathbb{R}_{+}^{n \times m}$ where $\mathbf{P}_{i,j}$ describes the amount of mass that is transported from a cell $x_i$ in the source cell distribution to a cell $y_j$ in the target cell distribution.

In practice, we solve a regularized ({term}`entropic regularization`) formulation of {term}`OT` due to computational and statistical reasons. The first kind of {term}`OT` problem we consider is a {term}`linear problem`, which considers the scenario when both cell distributions live in the same space:

```{math}
\begin{align*}
    \mathbf{L_C^{\varepsilon}(a,b) \overset{\mathrm{def.}}{=} \min_{P\in U(a,b)} \left\langle P, C \right\rangle - \varepsilon H(P).}
\end{align*}
```

Here, $\varepsilon$ is the {term}`entropic regularization`, and $\mathbf{H(P) \overset{\mathrm{def.}}{=} - \sum_\mathnormal{i,j} P_\mathnormal{i,j} \left( \log (P_\mathnormal{i,j}) - 1 \right)}$ is the discrete entropy of a coupling matrix.

:::{figure} figures/Kantorovich_couplings_sol.jpeg
:align: center
:alt: Kantorovich couplings.
:class: img-fluid

Continuous and discrete couplings between measures $\alpha, \beta$. Figure from {cite}`peyre:19`.
:::

## Gromov-Wasserstein (GW)

When the two cell distributions lie in different spaces, we are concerned with the {term}`quadratic problem`.
Here, we assume that two matrices $\mathbf{D \in \mathbb{R}^\mathnormal{n \times n}}$ and $\mathbf{D' \in \mathbb{R}^\mathnormal{m \times m}}$
quantify similarity relationships between cells within the respective distribution.

:::{figure} figures/GWapproach.jpeg
:align: center
:alt: Gromov-Wasserstein approach.
:class: img-fluid

Gromov-Wasserstein approach to comparing two metric measure spaces. Figure from {cite}`peyre:19`.
:::

The {term}`Gromov-Wasserstein` problem reads

```{math}
\begin{align*}
    \mathrm{GW}\mathbf{((a,D), (b,D'))^\mathrm{2} \overset{\mathrm{def.}}{=} \min_{P \in U(a,b)} \mathcal{E}_{D,D'}(P)}
\\
    \textrm{where} \quad \mathbf{\mathcal{E}_{D,D'}(P) \overset{\mathrm{def.}}{=} \sum_{\mathnormal{i,j,i',j'}} \left| D_{\mathnormal{i,i'}} - D'_{\mathnormal{j,j'}} \right|^\mathrm{2} P_{\mathnormal{i,i'}}P_{\mathnormal{j,j'}}}.
\end{align*}
```

In practice, we solve a formulation incorporating {term}`entropic regularization`.

## Fused Gromov-Wasserstein (FGW)

{term}`Fused Gromov-Wasserstein` is needed in cases where a data point, e.g. a cell, of the source distribution
has both features in the same space as the target distribution ({term}`linear term`) and features in a
different space than a data point in the target distribution ({term}`quadratic term`).

:::{figure} figures/FGWadapted.jpg
:align: center
:alt: Fused Gromov-Wasserstein distance.
:class: img-fluid

Fused Gromov-Wasserstein distance incorporates both feature and structure aspects of the source and target measures.
Figure adapted from {cite}`vayer:20`.
:::

The FGW problem is defined as

```{math}
\begin{align*}
    \mathrm{FGW}\mathbf{(a,b,D,D',C) \overset{\mathrm{def.}}{=} \min_{P \in U(a,b)} E_{D,D',C}(P)}
\\
    \textrm{where} \quad \mathbf{E_{D,D',C}(P) \overset{\mathrm{def.}}{=} \sum_{\mathnormal{i,j,i',j'}} \left( (1-\alpha)C_\mathnormal{i,j} + \alpha  \left| D_{\mathnormal{i,i'}} - D'_{\mathnormal{j,j'}} \right|^\mathrm{2} \right) P_{\mathnormal{i,i'}}P_{\mathnormal{j,j'}}}
\end{align*}
```

Here, $D$ and $D'$ are distances defined on the incomparable part of the source space and target space, respectively. $C$ quantifies the distance in the shared space. $\alpha \in [0,1]$ determines the influence of both terms.

## Unbalanced OT

When we would like to automatically discard cells (e.g. due to apoptosis or sequencing biases) or increase the influence of cells (e.g. due to proliferation)
we can add a penalty for the amount of mass variation using Kullback-Leibler divergence defined as

```{math}
\begin{align*}
    \mathrm{KL}\mathbf{(P|K) \overset{\mathrm{def.}}{=} \sum_\mathnormal{i,j} P_\mathnormal{i,j} \log \left( \frac{P_\mathnormal{i,j}}{K_\mathnormal{i,j}} \right) - P_\mathnormal{i,j} + K_\mathnormal{i,j}}.
\end{align*}
```

In the {term}`linear problem`, this results in the minimisation

```{math}
\begin{align*}
   \mathbf{L_C^{\lambda}(a,b) =  \min_{\tilde{a},\tilde{b}}  L_C(a,b) + \lambda_1 KL(a,\tilde{a}) + \lambda_2 KL(b,\tilde{b})} \\
   \mathbf{= \min_{P\in \mathbb{R}_+^\mathnormal{n\times m}} \left\langle C,P \right\rangle + \lambda_1 KL(P\mathbb{1}_\mathnormal{m}|a) + \lambda_2 KL(P^\top\mathbb{1}_\mathnormal{m}|b)}
\end{align*}
```

where $(\lambda_1, \lambda_2)$ controls how much mass variations are penalized as opposed to transportation of the mass. Here, $\lambda \in [0, \inf]$. Instead, we use the parameter

$\tau = \frac{\lambda}{\lambda + \varepsilon} \in [0,1]$

such that $\tau_a=\tau_b=1$ corresponds to the balanced setting, while a smaller $\tau$ allows for more deviation from the initial distribution. For the {term}`quadratic problem`, the objective is adapted analogously.

Now you are set to explore use cases in our {doc}`/notebooks/tutorials/index`.
