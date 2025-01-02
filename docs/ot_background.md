# Optimal Transport (OT) in a nutshell

Optimal transport ({term}`OT`) is a general problem in mathematics that has powerful applications in single-cell genomics analysis, especially in the context of spatial and multi-modal data.\
The problem that OT aims to solve is minimizing some measure of distance $L$ between two distributions, e.g. sets of cells.
The solution is encoded using a {term}`transport matrix` $\mathbf{P} \in \mathbb{R}_{+}^{n \times m}$ where $\mathbf{P}_{i,j}$ describes the amount of mass that is transported from data point $x_i$ in row $i$ to data point $y_j$ in column $j$.

The regularized {term}`linear OT` problem reads: 

```math
\begin{align*}
    \mathbf{L_C^{\varepsilon}(a,b) \overset{\mathrm{def.}}{=} \min_{P\in U(a,b)} \left\langle P, C \right\rangle - \varepsilon H(P).}
\end{align*}
```

where $\varepsilon$ is the {term}`entropic regularization`, and $\mathbf{H(P) \overset{\mathrm{def.}}{=} - \sum_\mathnormal{i,j} P_\mathnormal{i,j} \left( \log (P_\mathnormal{i,j}) - 1 \right)}$ is the discrete entropy of a coupling matrix.

## Gromov-Wasserstein (GW)

When the data points (e.g. cells) from source and target distributions lie in different metric spaces, 
we only assume that two matrices $\mathbf{D \in \mathbb{R}^\mathnormal{n \times n}}$ and $\mathbf{D' \in \mathbb{R}^\mathnormal{m \times m}}$
quantify similarity relationships between data points within the respective distribution.\

:::{figure} figures/GWapproach.jpeg
:align: center
:alt: Gromov-Wasserstein approach.
:class: img-fluid

Gromov-Wasserstein approach to comparing two metric measure spaces. Figure credit: {cite}`peyre:19`.
:::

The {term}`Gromov-Wasserstein` problem reads

```math
\begin{align*}
    \mathrm{GW}\mathbf{((a,D), (b,D'))^\mathrm{2} \overset{\mathrm{def.}}{=} \min_{P \in U(a,b)} \mathcal{E}_{D,D'}(P)}
\end{align*}
\\
\begin{align*}
    \textrm{where} \quad \mathbf{\mathcal{E}_{D,D'}(P) \overset{\mathrm{def.}}{=} \sum_{\mathnormal{i,j,i',j'}} \left| D_{\mathnormal{i,i'}} - D'_{\mathnormal{j,j'}} \right|^\mathrm{2} P_{\mathnormal{i,i'}}P_{\mathnormal{j,j'}}}.
\end{align*}
```

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

```math
\begin{align*}
    \mathrm{FGW}\mathbf{(a,b,D,D',M) \overset{\mathrm{def.}}{=} \min_{P \in U(a,b)} E_{D,D',M}(P)}
\end{align*}
\\
\begin{align*}
    \textrm{where} \quad \mathbf{E_{D,D',M}(P) \overset{\mathrm{def.}}{=} \sum_{\mathnormal{i,j,i',j'}} \left( (1-\alpha)M_\mathnormal{i,j} + \alpha  \left| D_{\mathnormal{i,i'}} - D'_{\mathnormal{j,j'}} \right|^\mathrm{2} \right) P_{\mathnormal{i,i'}}P_{\mathnormal{j,j'}}}
\end{align*}
```

Here, $D$ and $D'$ are distances in the structure spaces, $M$ - the distance in the feature space,
and $\alpha \in [0,1]$ is the tradeoff between the feature and the structure costs.

## Unbalanced OT

In cases that require allowing to ignore any outliers or skip points that don’t have a satisfactory mapping,
we can add a penalty for the amount of mass variation using some divergence $D_{\varphi}$
and get the minimization of an OT distance between approximate measures
 ```math
\begin{align*}
    \mathbf{L_C^{\tau}(a,b) =  \min_{\tilde{a},\tilde{b}}  L_C(a,b) + \tau_1 D_{\varphi}(a,\tilde{a}) + \tau_2 D_{\varphi}(b,\tilde{b})} \\
    \mathbf{= \min_{P\in \mathbb{R}_+^\mathnormal{n\times m}} \left\langle C,P \right\rangle + \tau_1 D_{\varphi}(P\mathbb{1}_\mathnormal{m}|a) + \tau_2 D_{\varphi}(P^\top\mathbb{1}_\mathnormal{m}|b)}
\end{align*}
```
where $(\tau_1, \tau_2)$ controls how much mass variations are penalized as opposed to transportation of the mass.
