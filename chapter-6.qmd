## 6.1 Estimation through empirical equations {#section-1}

The likelihood methods developed so far begin by specifying a full probability model. Estimating equations begin with the particular feature of a distribution that defines the target. A mean, for example, is characterized by $E(Y-\theta_0)=0$ without specifying a density for $Y$. This chapter develops the large-sample theory needed to turn such identities into estimates, standard errors, and confidence intervals. It then adapts the equations to missing observations by weighting and augmentation.

The logical order matters. First establish that the population equation identifies the intended target. Next show that its empirical solution converges to that target. Finally derive a linear approximation that accounts for all fitted nuisance parameters. A mean-zero estimating function alone does not guarantee a unique or consistent empirical root.

Specifying an entire joint distribution can be demanding when observations are multivariate or longitudinal. It can also impose assumptions unrelated to the scientific target: estimating a mean, for example, does not inherently require choosing every feature of a density. Estimating equations make the required moment restrictions explicit and use those restrictions directly for inference. Their robustness is consequently specific rather than absolute. An estimator can remain valid when unused aspects of the distribution are misspecified, while still depending on the mean, selection, and identification assumptions encoded in its equation.

::::: math-passage
Maximum likelihood supplies a familiar starting example. Let $p_Y(Y;\theta)$ be the full-data density and $l(Y;\theta)=\log p_Y(Y;\theta)$ its log density. For an independent sample, the MLE maximizes the sum of these contributions and, at an interior differentiable maximum, solves the score equation:

::: math-block
$$
\begin{aligned}
\widehat\theta&=\arg\max_\theta \sum_{i=1}^n l(Y_i;\theta)\\
&=\operatorname{Root}_\theta\sum_{i=1}^n \dot l(Y_i;\theta),
\end{aligned}
$$
:::

The single-observation score is

::: math-block
$$
\dot
l(Y;\theta)=\frac{\partial}{\partial\theta}l(Y;\theta)
$$
:::

This representation suggests replacing the likelihood score by another function whose expectation vanishes at the true parameter. Before doing so, we develop the probability-limit and linearization arguments needed to understand the resulting estimator. As with likelihood itself, an arbitrary root is not enough: its relation to the intended target must be established.
:::::

## 6.2 Convergence and stochastic order {#section-2}

::::: math-passage
Distinguish the true parameter $\theta_0$ from a generic candidate value $\theta$. The score $\dot l(Y;\theta)$ is a random function of the candidate, whereas $\dot
l(Y;\theta_0)$ is that function evaluated at the truth. Unless a different distribution is explicitly indicated, every expectation is taken under the true law of $Y$. Thus, for a function $g$,

::: math-block
$$
Eg(Y)=\int g(y)p_Y(y;\theta_0)d\nu(y).
$$
:::

If the function itself has a parameter argument, that argument does not change the distribution used for the expectation:

::: math-block
$$
Eg(Y;\theta)=\int
g(y;\theta)p_Y(y;\theta_0)d\nu(y).
$$
:::

This distinction is essential in a population estimating equation. The candidate parameter varies inside the estimating function, while the true data-generating law remains fixed.
:::::

:::: math-passage
Recall that a sequence of random variables $A_n$ converges in probability to a constant $a$, i.e.,

::: math-block
$$
\begin{equation}\tag{6.1}\label{eq:consistent_def}
A_n\to_p a,\end{equation}
$$
:::

if $\operatorname{Pr}(|A_n-a|>\epsilon)\to
0$ for every $\epsilon>0$. In other words, $A_n$ is eventually concentrated in arbitrarily small neighborhoods of $a$. An estimator $\widehat\theta_n$ is consistent, if $\widehat\theta_n\to_p\theta_0$. Consistency is the basic requirement for validity of an estimator. Asymptotic order: $A_n=o_p(B_n)$ if $A_n/B_n\to_p 0$. $A_n=O_p(B_n)$ if $A_n/B_n$ is asymptotically bounded (in probability), i.e., for every $\epsilon>0$, there exists $M$ such that $\liminf_n\operatorname{Pr}(|A_n/B_n|\leq
M)>1-\epsilon$.
::::

:::::: math-passage
So, in particular, $\widehat\theta_n\to_p\theta_0$ is the same as $\widehat\theta_n=\theta_0+o_p(1)$. We say $A_n$ converges in distribution (or weakly) to a random variable $A$, denoted as

::: math-block
$$
A_n\leadsto A,
$$
:::

if the distribution function of $A_n$ converges point-wise to that of $A$ (at the continuity points of the latter). Thus, if the weak convergence holds, the distribution of $A$ can be used to approximate that of $A_n$. In estimation of $\theta_0$, if

::: math-block
$$
\begin{equation}\tag{6.2}\label{eq:asym_norm}
\sqrt n(\widehat\theta_n-\theta_0)\leadsto
N(0,\sigma^2),\end{equation}
$$
:::

then, the distribution of $\widehat\theta_n$ can be approximated by

::: math-block
$$
N(\theta_0,n^{-1}\sigma^2),
$$
:::

where $\sigma^2$ is typically replaced by some estimator $\widehat\sigma^2_n$.
::::::

Consistency and asymptotic unbiasedness are different properties. Consistency is convergence in probability; unbiasedness concerns an expectation. Convergence in probability does not imply convergence of expectations without additional control of the tails. For example, an estimator equal to $n$ with probability $1/n$ and zero otherwise converges in probability to zero but has expectation one. Throughout this chapter, "consistent" is used in its probability-limit sense.

The notation $a^{\otimes2}$ means $aa^T$ for a column vector. Unless stated otherwise, the asymptotic covariance is that of $\sqrt n(\widehat\theta-\theta_0)$. Divide it by $n$ to estimate the sampling covariance of $\widehat\theta$ itself. Confusing those two scales produces a standard error wrong by a factor of $\sqrt n$.

## 6.3 Linearization and influence functions {#section-3}

:::: math-passage
The route to asymptotic normality is a linear approximation. Slutsky's lemma says that adding a term converging to zero in probability does not alter a limiting distribution: if $A_n=B_n+o_p(1)$ and $B_n\leadsto A$, then $A_n\leadsto A$. We therefore seek a representation of the estimator as an average of independent contributions plus a negligible remainder:

::: math-block
$$
\begin{equation}\tag{6.3}\label{eq:linearize}\begin{aligned}
\sqrt
n(\widehat\theta_n-\theta_0)&=n^{-1/2}\sum_{i=1}^n\Psi(Y_i;\theta_0)+o_p(1)\\
&\leadsto N(0,E\Psi(Y;\theta_0)^{\otimes 2}),
\end{aligned}\end{equation}
$$
:::

The mean-zero contribution $\Psi(Y;\theta_0)$ is the influence function. Once this representation and a finite second moment have been established, the central limit theorem gives the limiting normal law in the second line. The same representation also identifies the asymptotic covariance, so it connects the proof of normality directly to standard-error estimation.
::::

:::: math-passage
Another advantage of taking the linearization step is that it provides a natural estimator for the asymptotic variance, namely

::: math-block
$$
n^{-1}\sum_{i=1}^n\Psi(Y_i;\widehat\theta_n)^{\otimes
2}.
$$
:::

When $\widehat\theta_n$ can be linearized as in ($\ref{eq:linearize}$), we say $\widehat\theta_n$ is asymptotically linear. The MLE in a regular parametric model can be shown to be asymptotically linear and its influence functions can be exhibited.
::::

:::: math-passage
An influence function expresses the leading contribution of one observation to an estimator. If

::: math-block
$$
\widehat\theta-\theta_0=\frac1n\sum_i\Psi(O_i)+o_p(n^{-1/2}),\qquad
E\Psi=0,\quad E\|\Psi\|^2<\infty,
$$
:::

the central limit theorem gives covariance $E(\Psi\Psi^T)/n$. A direct estimate is $n^{-2}\sum_i(\widehat\Psi_i-\overline\Psi)(\widehat\Psi_i-\overline\Psi)^T$. Centering is harmless asymptotically and useful when estimated influence values do not sum exactly to zero. For independent subjects with repeated observations, $O_i$ is the whole subject record; the summation is over subjects, not over individual visits treated as independent.
::::

## 6.4 Likelihood as an estimating equation {#section-4}

::::: math-passage
Consider the function $g(D;\theta)\equiv
n^{-1}\sum_{i=1}^n\dot l(Y_i;\theta)$ as a (data-dependent) function of $\theta$. We use Taylor expansion of $g(D;\widehat\theta_n)$ around $\theta_0$:

::: math-block
$$
n^{-1}\sum_{i=1}^n\dot
l(Y_i;\widehat\theta_n)=n^{-1}\sum_{i=1}^n\dot
l(Y_i;\theta_0)+(\widehat\theta_n-\theta_0)n^{-1}\sum_{i=1}^n\ddot
l(Y_i;\theta_0)+o_p(\widehat\theta_n-\theta_0).
$$
:::

By definition of MLE, the left hand side is $0$. So, after algebraic manipulation,

::: math-block
$$
\begin{equation}\tag{6.4}\label{eq:linear_mle}\begin{aligned}
\sqrt n(\widehat\theta_n-\theta_0)&=-\left(n^{-1}\sum_{i=1}^n\ddot
l(Y_i;\theta_0)\right)^{-1}\frac{1}{\sqrt n}\sum_{i=1}^n\dot
l(Y_i;\theta_0)+o_p\left(\sqrt n(\widehat\theta_n-\theta_0)\right)\\
&=-\left\{E\ddot l(Y;\theta_0)\right\}^{-1}\frac{1}{\sqrt
n}\sum_{i=1}^n\dot l(Y_i;\theta_0)+o_p\left(\left|\sqrt
n(\widehat\theta_n-\theta_0)\right|+1\right).
\end{aligned}\end{equation}
$$
:::
:::::

:::::: math-passage
Note that $E\dot l(Y;\theta_0)=0$ and that

::: math-block
$$
-E\ddot l(Y;\theta_0)=E\dot
l(Y;\theta_0)^{\otimes 2}.
$$
:::

Hence,

::: math-block
$$
\sqrt n(\widehat\theta_n-\theta_0)=\frac{1}{\sqrt
n}\sum_{i=1}^n\widetilde l(Y_i;\theta_0)+o_p(1),
$$
:::

where

::: math-block
$$
\widetilde l(Y;\theta_0)=\left(E\dot
l(Y;\theta_0)^{\otimes 2}\right)^{-1}\dot l(Y;\theta_0)
$$
:::

is the **efficient influence function** for $\theta$ at $\theta_0$. The reason $\widetilde l(Y;\theta_0)$ is called efficient influence function is because a regular asymptotically linear estimators for $\theta_0$ is statistically efficient, i.e., having the smallest variance, if and only if its influence function is $\widetilde
l(Y;\theta_0)$.
::::::

:::::: math-passage
Denote

::: math-block
$$
\mathcal I(\theta_0)=E\dot
l(Y;\theta_0)^{\otimes 2}
$$
:::

as the information of $\theta$ based on one observation. Then the asymptotic variance for the MLE $\widehat\theta_n$ is

::: math-block
$$
\mathcal I(\theta_0)^{-1}E\dot
l(Y;\theta_0)^{\otimes 2}\mathcal I(\theta_0)^{-1}=\mathcal
I(\theta_0)^{-1},
$$
:::

which can be estimated by

::: math-block
$$
\left(n^{-1}\sum_{i=1}^n\dot
l(Y;\widehat\theta_n)^{\otimes 2}\right)^{-1}\mbox{ or
}-\left(n^{-1}\sum_{i=1}^n\ddot
l(Y;\widehat\theta_n)\right)^{-1}
$$
:::
::::::

::::::: math-passage
In the second step we have replaced

::: math-block
$$
-\left(n^{-1}\sum_{i=1}^n\ddot
l(Y_i;\theta_0)\right)^{-1}\frac{1}{\sqrt n}\sum_{i=1}^n\dot
l(Y_i;\theta_0)
$$
:::

by

::: math-block
$$
-\left\{E\ddot
l(Y;\theta_0)\right\}^{-1}\frac{1}{\sqrt n}\sum_{i=1}^n\dot
l(Y_i;\theta_0)+o_p(1)
$$
:::

. Denote this as replacing $A_nB_n$ by $AB_n+o_p(1)$. This replacement is justified because by the law of large numbers (and continuous mapping theorem), $A_n=A+o_p(1)$, and also $B_n=O_p(1)$ because it is weakly convergent to a multivariate normal. So

::: math-block
$$
A_nB_n=(A+o_p(1))B_n=AB_n+o_p(1)O_p(1)=AB_n+o_p(1).
$$
:::

In general, if $B_n$ is not asymptotically bounded (in probability), then,

::: math-block
$$
A_nB_n\neq AB_n+o_p(1),
$$
:::

even if the difference between $A_n$ and $A$ is $o_p(1)$.
:::::::

:::: math-passage
The displayed scalar Taylor expansions have the following column-vector form. With score $s_i(\theta)$ and Hessian $H_i(\theta)=\partial
s_i/\partial\theta^T$,

::: math-block
$$
0=P_ns(\theta_0)+\{P_nH(\theta_0)\}(\widehat\theta-\theta_0)+o_p(\|\widehat\theta-\theta_0\|).
$$
:::

If the estimator is consistent, the Hessian converges uniformly near the truth, and $EH(\theta_0)$ is nonsingular, the remainder can be absorbed after establishing the $n^{-1/2}$ rate. This gives $\Psi=-\{EH\}^{-1}s$. Under a correctly specified regular likelihood the information identity gives $-EH=E(ss^T)$. Under misspecification the equality can fail, so a sandwich covariance is needed even for a likelihood maximizer targeting a pseudo-true parameter.
::::

## 6.5 General estimating equations {#section-5}

:::::: math-passage
We call a function $m(Y;\theta)$ a valid **estimating function** if

::: math-block
$$
Em(Y;\theta_0)=0.
$$
:::

Given an estimating function $m(Y;\theta)$, we can derive an $M$ estimator $\widehat\theta_n$ based on solving

::: math-block
$$
n^{-1}\sum_{i=1}^nm(Y_i;\widehat\theta_n)=0,
$$
:::

that is,

::: math-block
$$
\widehat\theta_n=\operatorname{Root}_\theta\left\{n^{-1}\sum_{i=1}^nm(Y_i;\theta)\right\}.
$$
:::

Clearly, the MLE corresponds to the $M$ estimator with $m(Y;\theta)=\dot l(Y;\theta)$.
::::::

::::: math-passage
Valid estimating functions (i.e., mean zero at true value of parameter) generally lead to valid $M$ estimators. The reasoning is as follows. By the law of large numbers, we have that, for all $\theta$,

::: math-block
$$
n^{-1}\sum_{i=1}^nm(Y_i;\theta)\to_p
Em(Y;\theta).
$$
:::

Therefore one expects that

::: math-block
$$
\widehat\theta_n=\operatorname{Root}_\theta\left\{n^{-1}\sum_{i=1}^nm(Y_i;\theta)\right\}\to_p\operatorname{Root}_\theta
Em(Y;\theta)=\theta_0.
$$
:::

Hence unbiasedness of the estimating function is the starting point. Consistency also requires identification, a suitable uniform law of large numbers, and selection of an appropriate root.
:::::

### Conditions behind consistency

Let $M(\theta)=Em(Y;\theta)$. A useful consistency argument assumes uniform convergence $\sup_{\theta\in\Theta}\|P_nm(\theta)-M(\theta)\|\to_p0$, continuity of $M$, and separation of its root: outside every neighborhood of $\theta_0$, $\|M(\theta)\|$ is bounded away from zero on the relevant parameter set. If the selected empirical root approximately solves the equation, it cannot remain outside that neighborhood. Pointwise laws of large numbers alone do not establish this conclusion. A function identically zero is unbiased at every parameter value but identifies nothing.

## 6.6 Linear regression and sandwich variance {#section-6}

::::: math-passage
Let $Y$ and $Z$ denote the response and the covariates, respectively. Suppose

::: math-block
$$
\begin{equation}\tag{6.5}\label{eq:linear_reg}
Y=\beta^{\mathrm{T}}Z+\epsilon,  E[\epsilon\mid
Z]=0.\end{equation}
$$
:::

But we do not specify the conditional distribution of $\epsilon$. Then, a moment's thought shows that the following estimating function

::: math-block
$$
m_h(Y,Z;\beta)=h(Z)(Y-\beta^{\mathrm{T}}Z)
$$
:::

is valid for any function $h$ of $Z$. Choice of the weight function $h(Z)$ does not affect the validity of the $M$ estimator as long as the model assumption ($\ref{eq:linear_reg}$) is satisfied.
:::::

However, it does affect the asymptotic efficiency of the estimator through its influence function, as we will see. The particular choice of $h_0(Z)=Z$ make $m_{h_0}(Y,Z;\beta)$ proportional to the score function of the regression model with independent normal errors, so that this $M$ estimator is equivalent to the MLE and is thus efficient when the underlying error distribution is normal. When the error distribution is not normal, the $M$ estimator associated with $m_{h_0}(Y,Z;\beta)$ is still valid, but need not be efficient. Hence, it is called **locally efficient** at normal errors.

:::::: math-passage
The derivation of asymptotics for the $M$ estimator is essentially the same as the MLE, only with $\dot l(Y;\theta)$ replaced by $m(Y;\theta)$. Specifically, we have

::: math-block
$$
\sqrt
n(\widehat\theta_n-\theta_0)=-\left\{E\dot
m(Y;\theta_0)\right\}^{-1}\frac{1}{\sqrt
n}\sum_{i=1}^nm(Y_i;\theta_0)+o_p(1),
$$
:::

where $\dot
m(Y;\theta)=\frac{\partial}{\partial\theta}m(Y;\theta)$. So the influence function for $\widehat\theta_n$ is

::: math-block
$$
-\left\{E\dot m(Y;\theta_0)\right\}^{-1}
m(Y;\theta_0).
$$
:::

The asymptotic variance is

::: math-block
$$
\left\{E\dot
m(Y;\theta_0)\right\}^{-1}Em(Y;\theta_0)^{\otimes 2}\left\{E\dot
m(Y;\theta_0)^{\mathrm{T}}\right\}^{-1}.
$$
:::
::::::

:::::: math-passage
Note that in this case, there is no reason to believe there is any relationship between $-E\dot
m(Y;\theta_0)$ and $Em(Y;\theta_0)^{\otimes 2}$. So, an estimator for the asymptotic variance is the following **sandwich estimator**

::: math-block
$$
\left\{n^{-1}\sum_{i=1}^n\dot
m(Y_i;\widehat\theta_n)\right\}^{-1}\left\{n^{-1}\sum_{i=1}^nm(Y_i;\widehat\theta_n)^{\otimes
2}\right\}\left\{n^{-1}\sum_{i=1}^n\dot
m(Y_i;\widehat\theta_n)^{\mathrm{T}}\right\}^{-1}.
$$
:::

In Example 1, the influence function for the $M$ estimator is

::: math-block
$$
\left\{Eh(Z)Z^{\mathrm{T}}\right\}^{-1}h(Z)(Y-\beta_0^{\mathrm{T}}Z).
$$
:::

The sandwich estimator for asymptotic variance is

::: math-block
$$
\left\{n^{-1}\sum_{i=1}^nh(Z_i)Z_i^{\mathrm{T}}\right\}^{-1}n^{-1}\sum_{i=1}^n(Y_i-\widehat\beta_n^{\mathrm{T}}Z_i)^2h(Z_i)^{\otimes
2}\left\{n^{-1}\sum_{i=1}^nZ_ih(Z_i)^{\mathrm{T}}\right\}^{-1}.
$$
:::
::::::

::::: math-passage
For $m(Y,Z;\beta)=h(Z)(Y-\beta^TZ)$, the derivative is $-h(Z)Z^T$. Hence the two minus signs cancel and the influence function is

::: math-block
$$
\Psi_\beta=\{Eh(Z)Z^T\}^{-1}h(Z)(Y-\beta_0^TZ).
$$
:::

For ordinary least squares $h(Z)=Z$, let $A=E(ZZ^T)$. Then

::: math-block
$$
\operatorname{avar}\{\sqrt
n(\widehat\beta-\beta_0)\}=A^{-1}E\{ZZ^T\operatorname{Var}(Y\mid
Z)\}A^{-1}
$$
:::

. This remains valid under heteroscedasticity. The familiar homoscedastic expression $\sigma^2A^{-1}$ requires constant conditional variance. Within the class of linear unbiased estimating functions, variance weighting $h(Z)=Z/\operatorname{Var}(Y\mid Z)$ is optimal under suitable nonsingularity conditions. Robust variance estimation protects against variance misspecification, not against an incorrect conditional mean.
:::::

## 6.7 Generalized estimating equations {#section-7}

::::: math-passage
Suppose each $Y_i=(Y_{i1},\cdots,Y_{iq})^{\mathrm{T}}$ consists of responses repeatedly measured on one subject. The covariates are a $q\times p$ matrix $Z_i=(Z_{i1},\cdots, Z_{iq})^{\mathrm{T}}$, where each $Z_{ij}$ is a $p$-dimensional vector. We want to model the conditional mean of $Y_{ij}$ as a function of $Z_{ij}$ using the generalized linear models:

::: math-block
$$
E[Y_{ij}\mid
Z_{ij}]=\mu(\beta_0^{\mathrm{T}}Z_{ij}),
$$
:::

where $\mu$ is a known inverse link function. In matrix form, this is

::: math-block
$$
\begin{equation}\tag{6.6}\label{eq:gee_model}
E[Y_i\mid Z_i]=\mu(Z_i\beta_0),\end{equation}
$$
:::

where the function $\mu(\cdot)$ operates component-wise.
:::::

:::: math-passage
However, the marginal regression models ($\ref{eq:gee_model}$) do not full specify the joint distribution of $Y_i$, whose components are likely to be correlated since they are measured on one individual. One option is include random effects (see GLMM in §3.4) to induce correlation and derive the MLE. If the random effects model are wrongly specified, inference can be invalid. An alternative is to use estimating equations based on the marginal regression models ($\ref{eq:gee_model}$) only. This approach is called the generalized estimating equation (GEE). Specifically, note that for any $p\times q$ weight matrix $H(Z)$, the estimating function

::: math-block
$$
H(Z)\{Y-\mu(Z\beta)\}
$$
:::

is valid.
::::

::::: math-passage
Although the $M$ estimator is valid no matter what $H$ is, it is most efficient if $H$ is chosen to be

::: math-block
$$
Z^{\mathrm{T}}\operatorname{Diag}\{\dot\mu(Z\beta)\}\operatorname{Var}(Y\mid
Z;\beta)^{-1}.
$$
:::

Since the structure $\operatorname{Var}(Y\mid Z;\beta)$ is unknown, it is convenient to choose a working model assuming independent components of $Y$. So the estimating equation becomes

::: math-block
$$
\sum_{i=1}^n\sum_{j=1}^q
Z_{ij}\dot\mu(\beta^{\mathrm{T}}Z_{ij})\operatorname{Var}(Y_{ij}\mid
Z_{ij};\beta)^{-1}\left(Y_{ij}-\mu(\beta^{\mathrm{T}}Z_{ij})\right)=0.
$$
:::

Note that the left hand side is equal to the score function under the independence working model (see §3.1).
:::::

::::: math-passage
Hence the estimator from GEE with the independence working assumption is valid as long as the regression models ($\ref{eq:gee_model}$) hold, that is, even when the independence assumption is not true, and is locally efficient when the independence assumption does hold. The sandwich estimator for the asymptotic variance is $A_n^{-1}\Omega_nA_n^{-1}$, where

::: math-block
$$
A_n=n^{-1}\sum_{i=1}^n\sum_{j=1}^q
Z_{ij}^{\otimes
2}\dot\mu(\widehat\beta_n^{\mathrm{T}}Z_{ij})^2\operatorname{Var}(Y_{ij}\mid
Z_{ij};\widehat\beta_n)^{-1},
$$
:::

and

::: math-block
$$
\Omega_n=n^{-1}\sum_{i=1}^n\left\{\sum_{j=1}^q
Z_{ij}\dot\mu(\widehat\beta_n^{\mathrm{T}}Z_{ij})\operatorname{Var}(Y_{ij}\mid
Z_{ij};\widehat\beta_n)^{-1}\left(Y_{ij}-\mu(\widehat\beta_n^{\mathrm{T}}Z_{ij})\right)\right\}^{\otimes
2}.
$$
:::
:::::

### The subject-level sandwich for GEE

:::: math-passage
Let $\mu_i(\beta)$ be the vector of marginal means, $D_i=\partial\mu_i/\partial\beta^T$, and $V_i$ a working covariance. Write the subject contribution as $U_i=D_i^TV_i^{-1}(Y_i-\mu_i)$. Then

::: math-block
$$
\begin{gathered}\widehat
A=\frac1n\sum_iD_i^TV_i^{-1}D_i\\[6pt]
\widehat B=\frac1n\sum_i U_iU_i^T\\[6pt]
\widehat{\operatorname{Var}}(\widehat\beta)=\frac1n\widehat
A^{-1}\widehat B\widehat A^{-T}.\end{gathered}
$$
:::

The residual in a nonlinear mean model is $Y_i-\mu_i(\beta)$, not $Y_i-Z_i\beta$. The working-independence double sum runs over the $q$ repeated responses. Validity assumes independent subjects and the stated conditional mean model; if time-dependent covariates invalidate that conditional mean interpretation, a sandwich covariance does not repair it. With missing responses, ordinary unweighted GEE generally needs stronger conditions than MAR. The weighting construction below addresses selection explicitly.
::::

## 6.8 Inverse probability weighting {#section-8}

:::: math-passage
Suppose $m(Y;\theta)$ is a valid estimating function. With missing data, the full-data estimating function is not applicable because $Y$ is not completely observed. Let $R=1$ if $Y$ is fully observed and $R=0$ if otherwise. In this chapter, we will always assume that data are MAR. The complete case analysis using the estimating function $m(Y;\theta)$ is to solve

::: math-block
$$
n^{-1}\sum_{i=1}^nR_im(Y_i;\theta)=0
$$
:::

for $\theta$. However, the CC analysis is generally biased, i.e., $E\{Rm(Y;\theta_0)\}\neq 0$, because the complete cases need not be a random sample from the population unless the data are MCAR.
::::

::::: math-passage
The inverse probability weighting (IPW) technique is a general approach to correcting the selection bias in the complete cases. It originates from the Horvitz-Thompson estimator in survey analysis to account for different sampling proportions. Specifically, suppose for each subject $i$, we know the probability of observing the full data $Y_i$, that is, the selection probability, denoted as $\pi_i={\Pr}(R_i=1\mid
Y_i)$. Then, the bias can be corrected by weighting each complete case with $\pi_i^{-1}$, so the estimating equation becomes

::: math-block
$$
n^{-1}\sum_{i=1}^n\pi_i^{-1}R_im(Y_i;\theta)=0.
$$
:::

The IPW estimating function is valid because

::: math-block
$$
E\left\{\pi_i^{-1}R_im(Y_i;\theta_0)\right\}=E\left\{\pi_i^{-1}E\left[R_i\mid
Y_i\right]m(Y_i;\theta_0)\right\}=Em(Y_i;\theta_0)=0.
$$
:::
:::::

The derivation of asymptotics of the resulting $M$ estimator follows the full-data case closely except for weighting each subject by $\pi_i^{-1}R_i$. If missingness is not by design but by chance, as is the case in observational studies, then we typically do not know the selection probabilities. The obvious solution is to use a parametric model to estimate the selection probabilities and then weight each complete case by the inverse of the estimated selection probability. For simplicity, we first look at the case with two-levels of missingness, that is, one level is the full data $Y$, indicated by $R=1$; the other is the partial data $Y_{\mathrm{obs}}$, indicated by $R=0$. We use $X$ to denote $Y_{\mathrm{obs}}$, that is, $X=Y_{\mathrm{obs}}$.

:::: math-passage
By the MAR assumption, the selection probability $\operatorname{Pr}(R=1\mid Y)$ is a function only of $X$. Denote this function as $\pi(X)$. In order to estimate the function $\pi(\cdot)$, we postulate a parametric model

::: math-block
$$
\begin{equation}\tag{6.7}\label{eq:pi_model}
\operatorname{Pr}(R=1\mid X)=\pi(X;\psi_0),\end{equation}
$$
:::

where $\psi$ is a finite-dimensional parameter. Note that $(R_i, X_i), i=1,\cdots,
n$ are fully observed (keep in mind that $X$ is a component, or more generally a function, of $Y$, so $X$ is observed in both incomplete and complete cases). So, an estimator $\widehat\psi_n$ of $\psi_0$ can be easily computed using the MLE based on the binary regression model ($\ref{eq:pi_model}$). Popular choices for model ($\ref{eq:pi_model}$) include logistic regression and probit regression models.
::::

:::: math-passage
After obtaining the estimate $\widehat\psi_n$, the selection probability for subject $i$ can be estimated by $\pi(X_i;\widehat\psi_n)$, giving rise to the following IPW estimating equation

::: math-block
$$
\begin{equation}\tag{6.8}\label{eq:ipw_ee}
n^{-1}\sum_{i=1}^n\frac{R_i}{\pi(X_i;\widehat\psi_n)}m(Y_i;\theta)=0.\end{equation}
$$
:::

for $\theta$. Two natural questions arise:
::::

1.  Is the $M$ estimator $\widehat\theta_n$ resulting from solving ($\ref{eq:ipw_ee}$) consistent to $\theta_0$?

2.  :::: math-passage
    If yes, is the asymptotic distribution (or equivalently, the influence function) of $\widehat\theta_n$ the same as the $M$ estimator from the estimating equation

    ::: math-block
    $$
n^{-1}\sum_{i=1}^n\frac{R_i}{\pi(X_i;\psi_0)}m(Y_i;\theta)=0?
$$
    :::
    ::::

::::: math-passage
The answer to the first question is yes, and that to the second is no. First, it is easy to see that the estimating function $\frac{R}{\pi(X;\psi_0)}m(Y;\theta)$ is valid in the sense that

::: math-block
$$
E\left\{\frac{R}{\pi(X;\psi_0)}m(Y;\theta_0)\right\}=0
$$
:::

by iterative conditional expectation. Then, under some extra regularity conditions, we expect that

::: math-block
$$
\begin{aligned}
n^{-1}\sum_{i=1}^n\frac{R_i}{\pi(X_i;\widehat\psi_n)}m(Y_i;\theta)&=n^{-1}\sum_{i=1}^n\frac{R_i}{\pi(X_i;\psi_0)}m(Y_i;\theta)+o_p(1)\\
&\to_pE\left\{\frac{R}{\pi(X;\psi_0)}m(Y;\theta)\right\},
\end{aligned}
$$
:::

so that $\widehat\theta_n$, the root for the left hand side goes in probability to $\theta_0$, the root for the far right hand side.
:::::

:::: math-passage
Under MAR, $E(R\mid Y)=\pi(X)$. Thus

::: math-block
$$
E\left\{\frac{R}{\pi(X)}m(Y;\theta_0)\right\}
=E\left[\frac{E(R\mid
Y)}{\pi(X)}m(Y;\theta_0)\right]=Em(Y;\theta_0)=0.
$$
:::

This calculation also reveals positivity: a zero denominator in a population region with relevant full-data contributions cannot be repaired by a weighting algorithm. For finite variance one needs an appropriate integrability condition, such as $E\{\|m(Y;\theta_0)\|^2/\pi(X)\}<\infty$. Very small probabilities can cause instability without violating strict positivity. Weight truncation may improve numerical behavior but changes the estimating equation and can introduce bias; it should be described and assessed as a separate modeling decision.
::::

## 6.9 Estimated nuisance parameters {#section-9}

::::: math-passage
We now treat the second question about asymptotic distribution of $\widehat\theta_n$, which we frame in a more general setting of $M$ estimation in the presence of an estimated nuisance parameter. Let $\{m(Y;\theta,\eta): \eta\}$ be a class of estimating functions for $\theta_0$ indexed by an unknown nuisance parameter $\eta$. We call it a valid class of estimating functions for $\theta_0$ if

::: math-block
$$
Em(Y;\theta_0,\eta_0)=0.
$$
:::

In our problem, $\eta=\psi$, and the class of estimating functions for $\theta$ is

::: math-block
$$
\left\{
\frac{R}{\pi(X;\psi)}m(Y;\theta):\psi\right\}.
$$
:::

Clearly, this is a valid class by previous arguments.
:::::

:::: math-passage
Now, we consider the following estimation equation for $\theta$

::: math-block
$$
\begin{equation}\tag{6.9}\label{eq:nuisance_ee}
n^{-1}\sum_{i=1}^nm(Y_i;\theta,\widehat\eta_n)=0,\end{equation}
$$
:::

where $\widehat\eta_n$ is a consistent estimator for $\eta_0$. Denote the $M$ estimator from ($\ref{eq:nuisance_ee}$), the solution to ($\ref{eq:nuisance_ee}$), as $\widehat\theta_n$. We want to derive the asymptotic distribution, or equivalently, the influence function, of $\widehat\theta_n$. To this aim, we treat $n^{-1}\sum_{i=1}^nm(Y_i;\theta,\eta)$ as a function of the bivariate $(\theta,\eta)$ and use (first-order) Taylor expansion around $(\theta_0,\eta_0)$ as similarly conducted in the standard case.
::::

:::: math-passage
We have that

::: math-block
$$
\begin{aligned}
n^{-1}\sum_{i=1}^nm(Y_i;\widehat\theta_n,\widehat\eta_n)&=n^{-1}\sum_{i=1}^nm(Y_i;\theta_0,\eta_0)\\
& +(\widehat\theta_n-\theta_0) n^{-1}\sum_{i=1}^n\dot
m_1(Y_i;\theta_0,\eta_0)\\
& +(\widehat\eta_n-\eta_0) n^{-1}\sum_{i=1}^n\dot
m_2(Y_i;\theta_0,\eta_0)\\
&
+o_p\left(|\widehat\theta_n-\theta_0|+|\widehat\eta_n-\eta_0|\right),
\end{aligned}
$$
:::

where $\dot
m_1(Y;\theta,\eta)=\partial m(Y;\theta,\eta)/\partial\theta$ and $\dot m_2(Y;\theta,\eta)=\partial
m(Y;\theta,\eta)/\partial\eta$.
::::

::::: math-passage
Note that, by definition, $n^{-1}\sum_{i=1}^nm(Y_i;\widehat\theta_n,\widehat\eta_n)=0$. By similar re-arrangements and arguments as in the standard case, we have that

::: math-block
$$
\begin{equation}\tag{6.10}\label{eq:linear1}\begin{aligned}
\sqrt n(\widehat\theta_n-\theta_0)&=-\left\{E\dot
m_1(Y;\theta_0,\eta_0)\right\}^{-1}\Bigg\{\frac{1}{\sqrt
n}\sum_{i=1}^nm(Y_i;\theta_0,\eta_0)
\\
&+E\dot m_2(Y;\theta_0,\eta_0)\sqrt
n(\widehat\eta_n-\eta_0)\Bigg\}+o_p(1).
\end{aligned}\end{equation}
$$
:::

Suppose $\widehat\eta_n$ is linearized as

::: math-block
$$
\sqrt n(\widehat\eta_n-\eta_0)=\frac{1}{\sqrt
n}\sum_{i=1}^n\Psi_\eta(Y_i;\eta_0)+o_p(1).
$$
:::
:::::

::::: math-passage
Then, ($\ref{eq:linear1}$) can be further linearized as

::: math-block
$$
\begin{equation}\tag{6.11}\label{eq:linear2}\begin{aligned}
\sqrt n(\widehat\theta_n-\theta_0)&=-\left\{E\dot
m_1(Y;\theta_0,\eta_0)\right\}^{-1}\frac{1}{\sqrt n}\sum_{i=1}^n\Bigg\{
m(Y_i;\theta_0,\eta_0)
\\
&+V(\theta_0,\eta_0)\Psi_\eta(Y;\eta_0)\Bigg\}+o_p(1).
\end{aligned}\end{equation}
$$
:::

where $V(\theta_0,\eta_0)=E\dot
m_2(Y;\theta_0,\eta_0)$. So the influence function for $\widehat\theta_n$ is

::: math-block
$$
\begin{aligned}
-\left\{E\dot m_1(Y;\theta_0,\eta_0)\right\}^{-1}\Bigg\{
m(Y;\theta_0,\eta_0)
+V(\theta_0,\eta_0)\Psi_\eta(Y;\eta_0)\Bigg\}.
\end{aligned}
$$
:::
:::::

::::: math-passage
The first term of the previous display is the influence function of the $M$ estimator with estimating equation $m(Y;\theta,\eta_0)$, and the second term represents the "influence" of estimating $\eta_0$. A sandwich estimator for the asymptotic variance can be similarly constructed as

::: math-block
$$
\begin{aligned}
&\left\{n^{-1}\sum_{i=1}^n\dot
m_1(Y_i;\widehat\theta_n,\widehat\eta_n)\right\}^{-1}n^{-1}\sum_{i=1}^n\Big\{m(Y_i;\widehat\theta_n,\widehat\eta_n)\\
& +\widehat
V_n(\widehat\theta_n,\widehat\eta_n)\Psi_\eta(Y_i;\widehat\eta_n)\Big\}^{\otimes
2}\left\{n^{-1}\sum_{i=1}^n\dot
m_1(Y_i;\widehat\theta_n,\widehat\eta_n)^{\mathrm{T}}\right\}^{-1},
\end{aligned}
$$
:::

where

::: math-block
$$
\widehat
V(\theta,\eta)=n^{-1}\sum_{i=1}^n\dot m_2(Y_i;\theta,\eta).
$$
:::
:::::

### A block-equation derivation

::::: math-passage
For column vectors define $A=E(\partial
m/\partial\theta^T)$ and $C=E(\partial m/\partial\eta^T)$. If a nuisance estimator has influence function $\Psi_\eta$, the Taylor expansion gives

::: math-block
$$
\Psi_\theta=-A^{-1}\{m(O;\theta_0,\eta_0)+C\Psi_\eta(O)\}.
$$
:::

Equivalently, stack $m$ with the nuisance estimating equation $s(O;\eta)$ and invert the block triangular derivative matrix

::: math-block
$$
\begin{pmatrix}A&C\\0&B\end{pmatrix},\qquad
B=E\frac{\partial s}{\partial\eta^T}.
$$
:::

Its upper-right block is $-A^{-1}CB^{-1}$, while $\Psi_\eta=-B^{-1}s$. This is often the simplest implementation: fit both models, evaluate all subject-level stacked contributions, and compute one sandwich covariance. The nuisance contribution may decrease or increase variance; treating estimated weights as known generally gives the wrong answer.
:::::

## 6.10 The influence function of IPW {#section-10}

::::::: math-passage
In our IPW estimation equation setting, we have that $\eta=\psi$,

::: math-block
$$
m(Y;\theta,\psi)=\frac{R}{\pi(X;\psi)}m(Y;\theta),
$$
:::

$\dot
m_1(Y;\theta,\psi)=\frac{R}{\pi(X;\psi)}\dot m(Y;\theta)$, and

::: math-block
$$
\dot
m_2(Y;\theta,\psi)=-\frac{R}{\pi(X;\psi)^2}m(Y;\theta)\dot\pi(X;\psi)^{\mathrm{T}}.
$$
:::

If the model $\pi(X;\psi)$ is assumed to be a logistic regression model, then

::: math-block
$$
\pi(X;\psi)=\frac{\exp(\psi^{\mathrm{T}}X)}{1+\exp(\psi^{\mathrm{T}}X)},
$$
:::

and the (efficient) influence function of $\widehat\psi_n$ is given by

::: math-block
$$
\Psi_\psi(R,X;\psi)=\left[E\{\pi(X;\psi)(1-\pi(X;\psi))XX^{\mathrm{T}}\}\right]^{-1}\{R-\pi(X;\psi)\}X.
$$
:::
:::::::

:::: math-passage
For a logistic response model with design vector $X$ including an intercept, let $s_\psi=(R-\pi)X$ and $I_\psi=E\{\pi(1-\pi)XX^T\}$. Then $\Psi_\psi=I_\psi^{-1}(R-\pi)X$. In the IPW equation, the cross derivative is the *expectation*

::: math-block
$$
C=-E\left\{\frac{R}{\pi(X)^2}m(Y;\theta_0)\dot\pi(X)^T\right\}.
$$
:::

It is a fixed population matrix multiplying the random nuisance influence function. Substituting the observation-level integrand for this expectation would produce a different and generally incorrect influence function.
::::

## 6.11 Estimating a partially observed mean {#section-11}

In Chapter 1 we looked at an example of estimating the mean of an outcome, i.e., weight, after an intervention program. The post-intervention weight was not observed on those who did not participate in the program. But it was assumed that whether the subject chose to participate, and thus, whether the variable of interest was observed, depends totally on the person's pre-intervention weight. Denote $Y=(X, Y_2)$, where $X$ and $Y_2$ are pre- and post-intervention weights, respectively. We thus want to estimate $\theta:=EY_2$. Let $R=1$ if $Y_2$ is observed, and $R=0$ if otherwise. By MAR assumption, $\operatorname{Pr}(R=1\mid Y)$ is only a function of $X$.

:::::: math-passage
For the purpose of estimating the selection probability, we use a logistic regression model

::: math-block
$$
\operatorname{Pr}(R=1\mid
X)=\frac{\exp(\psi_0^{\mathrm{T}}\widetilde
X)}{1+\exp(\psi_0^{\mathrm{T}}\widetilde X)},
$$
:::

where $\widetilde X=(1, X)$. Denote $\widehat\psi_n$ as the MLE for $\psi_0$. A natural estimator for $\theta$ had the full data been available is $n^{-1}\sum_{i=1}^n Y_{2i}$, which the $M$ estimator for the estimating function

::: math-block
$$
m(Y;\theta)=Y_2-\theta.
$$
:::

Thus the IPW estimating equation is

::: math-block
$$
n^{-1}\sum_{i=1}^n\frac{R_i}{\pi(X_i;\widehat\psi_n)}(Y_{2i}-\theta)=0.
$$
:::
::::::

::::: math-passage
This estimating equation has an explicit solution

::: math-block
$$
\widehat\theta_n=\frac{n^{-1}\sum_{i=1}^n\frac{R_i}{\pi(X_i;\widehat\psi_n)}Y_{2i}}{n^{-1}\sum_{i=1}^n\frac{R_i}{\pi(X_i;\widehat\psi_n)}}.
$$
:::

By the theory developed above, the influence function can be shown to be

::: math-block
$$
\Psi_\theta=\frac{R}{\pi(X;\psi_0)}(Y_2-\theta_0)-c^{\mathrm{T}}I^{-1}\{R-\pi(X;\psi_0)\}\widetilde
X,
$$
:::

Here $c$ and $I$ are the population moments defined immediately below. A variance estimator for $\widehat\theta_n$ can be constructed accordingly.
:::::

The ratio estimator is the solution of the weighted mean equation and is often called the normalized or Hájek estimator. With known weights its influence function is $R(Y_2-\theta_0)/\pi(X)$, whereas the unnormalized Horvitz--Thompson mean $P_n(RY_2/\pi)$ has influence function $RY_2/\pi-\theta_0$. They target the same mean but have different first-order behavior.

::::: math-passage
For fitted logistic weights, set

::: math-block
$$
c=E\{(1-\pi(X))(Y_2-\theta_0)\widetilde
X\},\qquad I=E\{\pi(X)(1-\pi(X))\widetilde X\widetilde X^T\}.
$$
:::

The normalized estimator's influence function is

::: math-block
$$
\Psi=\frac
R{\pi(X)}(Y_2-\theta_0)-c^TI^{-1}(R-\pi(X))\widetilde X.
$$
:::

Indeed $C=-c^T$ and $A=-E(R/\pi)=-1$. The second term subtracts the projection on the fitted response score in this correctly specified parametric setting. Estimate $c$ using the observed weighted expression $P_n\{R(1-\widehat\pi)(Y_2-\widehat\theta)\widetilde
X/\widehat\pi\}$ and estimate $I$ using all subjects. The sampling variance is estimated from these full influence values divided by $n$.
:::::

## 6.12 Monotone dropout {#section-12}

:::: math-passage
For monotone missingness, the IPW $M$ estimator works similarly as in the two-level case, i.e., by inversely weighting each complete case by its selection probability. The extra difficulty lies in modeling the missingness mechanism, and hence the selection probability. Let the full data be $Y=(Y_1,\cdots, Y_q)$. Let $R=j$, $j=1,\cdots, q$, if

::: math-block
$$
Y_{\mathrm{obs}}=(Y_1,\cdots, Y_j).
$$
:::

So, the indicator for a complete case is $I(R=q)$. We want to model this probability given the full data while assuming MAR. This can be completed in the following ways.
::::

::::: math-passage
We model the conditional selection probabilities

::: math-block
$$
\begin{equation}\tag{6.12}\label{eq:missing_hazard}
\operatorname{Pr}(R=j\mid R\geq j, Y)=\lambda_j(\widetilde
Y_j;\psi_{j0}),\end{equation}
$$
:::

where $\widetilde Y_j=(Y_1,\cdots, Y_j)$, $j=1,\cdots, q-1$. Note that the left hand side being only a function of $\widetilde
Y_j$ is justified by MAR. Denote the selection probability as $\pi(Y;\psi_0)=\operatorname{Pr}(R=q\mid
Y)$, where $\psi_0=(\psi_{10},\cdots,\psi_{q0})$. It can be shown that

::: math-block
$$
\pi(Y;\psi_0)=\prod_{j=1}^{q-1}\left\{1-\lambda_j(\widetilde
Y_j;\psi_{j0})\right\}
$$
:::

It is convenient to use binary regression models such as logistic regression to model ($\ref{eq:missing_hazard}$). The MLEs for $\widehat\psi_{jn}$ are based on "complete cases", i.e., subjects for which $\widetilde Y_j$ is observed.
:::::

::::: math-passage
To see this, note that those with some components of $\widetilde Y_j$ missing are excluded by the conditioning event $\{R\geq j\}$. Then, the selection probability can be estimated by

::: math-block
$$
\pi(Y;\widehat\psi_n)=\prod_{j=1}^{q-1}\left\{1-\lambda_j(\widetilde
Y_j;\widehat\psi_{jn})\right\}.
$$
:::

Then, the IPW $M$ estimator goes by solving

::: math-block
$$
n^{-1}\sum_{i=1}^n\frac{I(R_i=q)}{\pi(Y_i;\widehat\psi_n)}m(Y_i;\theta)=0
$$
:::

for $\theta$. Though more complicated, the asymptotic variance for $\widehat\theta_n$ can be derived similarly as in the two-level case.
:::::

The dropout model is fitted on a succession of risk sets. At stage $j$, include every subject with $R\geq j$ and use $I(R=j)$ as the binary dropout response. Calling these subjects "complete cases" means complete through stage $j$, not necessarily complete through the final visit. The product of the conditional retention probabilities is the probability of reaching the last visit. Stacking the stage-specific score equations with the final weighted estimating equation accounts for uncertainty in all the fitted hazards.

## 6.13 Augmentation and double robustness {#section-13}

We use Example 1 of §6.2 to compare the IPW approach with the MLE in dealing with missing data. Recall that the full data are $Y=(X,Y_2)$, where $X$ is always observed and $Y_2$ is possibly missing ($R=1$: $Y_2$ is observed; $R=0$, $Y_2$ is missing). We have assumed that the $Y_2$ are MAR, that is, the probability of missingness depends on $X$. The interest is in estimating the mean of $Y_2$. So we call $X$ an **auxiliary** variable. The MLE approach requires a joint model for the full data $Y=(X, Y_2)$, say, bivariate normal, and $EY_2$ is computed based on this model. The IPW approach requires a selection model, i.e., $\operatorname{Pr}(R=1\mid X)$, and $EY_2$ is computed based on inversely weighting the complete cases with the estimated selection probabilities.

:::: math-passage
When the joint model for $(X,
Y_2)$ is the bivariate normal, then it can be shown that the MLE (derived via an EM algorithm in chapter 1) for $EY_2$ is equivalent to

::: math-block
$$
n^{-1}\sum_{i=1}^n\left\{R_iY_{2i}+(1-R_i)\widehat
E(Y_{2i}\mid X_i)\right\},
$$
:::

where $\widehat E(Y_{2i}\mid X_i)$ is based on the conditional normal regression estimated from complete cases; the distribution of the fully observed covariate uses all subjects. In other words, the MLE seeks to "impute" the missing values of $Y_2$ using the auxiliary variable $X$ based on the their assumed joint distribution. The MLE would be biased if the posited joint model is incorrect. The IPW estimation differs from MLE in two respects, one in terms of modeling assumptions and the other in terms of the data used.
::::

For modeling assumptions, instead of the joint distribution of $(X, Y_2)$, the IPW specifies the conditional distribution $[R\mid X]$. The IPW estimator would be biased if the selection model is wrongly specified. Also, as a result of not positing any relationship between $X$ and $Y_2$, the IPW essentially uses information in the complete cases. There are two ways to improve the IPW estimator. First, we want to construct an estimator that is valid when either the model $[Y_2\mid X]$ or the model $[R\mid X]$ is true. Second, we want to exploit the association between $X$ and $Y_2$ to gain efficiency without compromising robustness.

:::: math-passage
Combining the outcome regression and the response model gives the augmented inverse-probability-weighted estimator:

::: math-block
$$
\widehat\mu_{\mathrm{DR}}=\frac1n\sum_{i=1}^n\left[\widehat
m(X_i)+\frac{R_i}{\widehat\pi(X_i)}\{Y_{2i}-\widehat
m(X_i)\}\right].
$$
:::

Here $\widehat
m(x)$ estimates $E(Y_2\mid
X=x)$, while $\widehat\pi(x)$ estimates $P(R=1\mid X=x)$. The derivation, variance, and extension to monotone missingness follow below.
::::

### Deriving the augmentation

:::::: math-passage
Let $m_0(x)=E(Y_2\mid X=x)$ and $\pi_0(x)=P(R=1\mid X=x)$. For arbitrary candidate functions $m$ and $\pi>0$, define the completed contribution

::: math-block
$$
H(O;m,\pi)=m(X)+\frac
R{\pi(X)}\{Y_2-m(X)\}.
$$
:::

The first term predicts an outcome for everyone. The second corrects the prediction using weighted observed residuals. Under MAR,

::: math-block
$$
E\{H(O;m,\pi)\mid
X\}=m(X)+\frac{\pi_0(X)}{\pi(X)}\{m_0(X)-m(X)\}.
$$
:::

Subtract $m_0(X)$ and average to obtain the exact bias identity

::: math-block
$$
E
H-\theta_0=E\left[\left\{1-\frac{\pi_0(X)}{\pi(X)}\right\}\{m(X)-m_0(X)\}\right].
$$
:::

It vanishes if $m=m_0$ or $\pi=\pi_0$. With consistent nuisance fits, an appropriate law of large numbers, and positivity, the empirical mean of $H$ is therefore consistent when either nuisance model is correct. "Doubly robust" does not mean unbiased in finite samples, consistent when both models are wrong, or protected against a failure of MAR.
::::::

### Variance and local efficiency

::::: math-passage
When both nuisance functions are consistently estimated with sufficient regularity, the influence function is

::: math-block
$$
\Psi_{\mathrm{DR}}=m_0(X)-\theta_0+\frac
R{\pi_0(X)}\{Y_2-m_0(X)\}.
$$
:::

The residual term has conditional mean zero, so its covariance with $m_0(X)-\theta_0$ is zero. Consequently

::: math-block
$$
E\Psi_{\mathrm{DR}}^2=\operatorname{Var}\{m_0(X)\}+E\left\{\frac{\operatorname{Var}(Y_2\mid
X)}{\pi_0(X)}\right\}.
$$
:::

This is the efficiency bound for the mean in the unrestricted MAR observed-data model. An empirical variance of fitted influence values divided by $n$ consistently estimates it under the corresponding conditions. If only one finite-dimensional nuisance model is correct, derivatives with respect to the other fitted model need not vanish; use the full stacked sandwich or another justified variance method. The simple influence formula should not be applied automatically at an arbitrary misspecified nuisance limit.
:::::

For flexible fitted functions, cross-fitting estimates nuisances on other folds before evaluating $H$ on held-out observations. Under positivity, finite moments, appropriate convergence, and suitable bounds, the leading bias is controlled by the product $\|\widehat
m-m_0\|_2\|\widehat\pi-\pi_0\|_2$. Requiring this product to be $o_p(n^{-1/2})$ is a common sufficient rate condition. Cross-fitting helps control dependence from model fitting; it does not make inaccurate nuisance estimates or inadequate overlap disappear.

### Augmented equations for a general target

:::: math-passage
The mean example extends to a full-data estimating function $u(Y;\theta)$. Define $a_\theta(X)=E\{u(Y;\theta)\mid X\}$. Solve

::: math-block
$$
P_n\left[\widehat a_\theta(X)+\frac
R{\widehat\pi(X)}\{u(Y;\theta)-\widehat a_\theta(X)\}\right]=0.
$$
:::

At the true parameter, the same conditional-expectation argument gives zero mean if either the response model or the conditional estimating-function model is correct. The derivative of the resulting population equation must still identify $\theta$. For $u=Y_2-\theta$, this reduces to the augmented mean estimator.
::::

## 6.14 Sequential augmentation for monotone missingness {#section-14}

:::: math-passage
Sequential augmentation extends the preceding construction to subjects followed over several visits. Let $A_j=I(R\geq j)$, with $A_1=1$, and $H_j=(Y_1,\ldots,Y_j)$. Define retention probabilities

::: math-block
$$
p_j(H_j)=P(A_{j+1}=1\mid
A_j=1,H_j),\quad G_1=1,\quad G_j=\prod_{k=1}^{j-1}p_k(H_k).
$$
:::

Assume sequential MAR: conditional on the observed history among those still followed, retention is independent of future full-data outcomes. Also assume the required probabilities are positive. The earlier hazard notation satisfies $p_j=1-\lambda_j$.
::::

::::: math-passage
For a terminal outcome $Y_q$, set $Q_q(H_q)=Y_q$ and recursively define

::: math-block
$$
Q_j(H_j)=E\{Q_{j+1}(H_{j+1})\mid
H_j\}=E(Y_q\mid H_j),\qquad j=q-1,\ldots,1.
$$
:::

Under sequential MAR these regressions can be estimated using subjects observed at the next stage, with the recursively constructed subsequent predictions as responses. A sequentially augmented estimator is

::: math-block
$$
\begin{gathered}\widehat\theta=P_n\left[\widehat
Q_1(H_1)+\sum_{j=1}^{q-1}\frac{A_{j+1}}{\widehat G_{j+1}}\{\widehat
Q_{j+1}(H_{j+1})-\widehat Q_j(H_j)\}\right]\\[6pt]
\widehat Q_q=Y_q.\end{gathered}
$$
:::

Every summand uses only information observed for subjects reaching its stage. For $q=2$ it is precisely the augmented inverse-probability-weighted mean.
:::::

### Two proofs of consistency

:::: math-passage
First suppose every retention model is correct, while the $Q_j$ are arbitrary integrable functions and $Q_q=Y_q$. Conditional on the full data, sequential MAR gives $E(A_{j+1}\mid Y)=G_{j+1}$. Therefore the expectation of each weighted increment equals $E(Q_{j+1}-Q_j)$. The sum telescopes:

::: math-block
$$
E Q_1+\sum_{j=1}^{q-1}E(Q_{j+1}-Q_j)=E Q_q=E
Y_q.
$$
:::

Next suppose every $Q_j$ is the true successive conditional expectation, while fitted retention models converge to positive functions $\widetilde p_j(H_j)$. Conditional on $H_j$ and being at risk, the next increment has mean zero even after the next-stage MAR selection. Its multiplier before that selection depends only on $H_j$. Thus every weighted increment has expectation zero and $EQ_1=EY_q$. These arguments establish the two global correctness cases. Stronger combinations of stage-specific robustness require additional careful conditions and are not inferred simply by counting models.
::::

When both sets of models are correct, subtract $\theta_0$ from the bracketed contribution to obtain the influence function under suitable regularity. Estimate its variance at the subject level, or use a stacked sandwich including all parametric nuisance fits when their first-order contributions do not cancel. Assess retention probabilities at every visit: a moderate per-visit dropout risk can compound into a very small final observation probability.

### A three-visit example and a practical sequence

:::: math-passage
With baseline $Y_1$, intermediate measurement $Y_2$, and final outcome $Y_3$, the contribution is

::: math-block
$$
Q_1(Y_1)+\frac{A_2}{p_1(Y_1)}\{Q_2(Y_1,Y_2)-Q_1(Y_1)\}
+\frac{A_3}{p_1(Y_1)p_2(Y_1,Y_2)}\{Y_3-Q_2(Y_1,Y_2)\}.
$$
:::

Fit the two retention models on their respective risk sets. Regress $Y_3$ on $(Y_1,Y_2)$ among subjects observed at visit 3 to obtain $\widehat Q_2$. Regress the resulting predictions on $Y_1$ among subjects reaching visit 2 to obtain $\widehat Q_1$. Evaluate the displayed contribution for each subject using only available increments, average it, and account for nuisance fitting in the variance. This example shows how intermediate observations contribute even when the final outcome is missing.
::::

### Exercises

Derive the bias identity for the general augmented equation and identify the derivative needed for its influence function. Next, verify the three-visit telescoping proof by expanding all three terms. Finally, compare the known-weight normalized and unnormalized mean estimators when $Y_2$ is constant: the normalized mean is exact whenever at least one outcome is observed, while the unnormalized mean still fluctuates with the realized weights. Explain why neither estimator is uniformly more efficient in every distribution.
