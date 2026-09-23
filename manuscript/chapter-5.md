## 5.1 The Bayesian perspective {#section-1}

Bayesian inference supplies the distributional counterpart to the optimization methods of Chapters 2--4. A likelihood describes how the observations depend on an unknown parameter; a prior completes a probability model for that parameter. The posterior then describes parameter uncertainty conditional on what was actually observed. With missing data, the unknowns include both parameters and unrecorded values. Their joint posterior is the central object, and the parameter posterior and predictive distribution are its two marginals.

This chapter first develops the complete-data calculations needed in that joint posterior. It then constructs data augmentation, uses the resulting predictive draws for multiple imputation, and explains why imputation requires both parameter uncertainty and residual uncertainty. Throughout, $\operatorname{Gamma}(a,b)$ uses shape $a$ and *rate* $b$, so its mean is $a/b$ and its variance is $a/b^2$.

:::: math-passage
Let $Y$ denote a complete observation with density $p_Y(y\mid\theta)$, and let $D=(Y_1,\ldots,Y_n)$ be an independent sample. Both frequentist and Bayesian analyses begin with the same sampling model and likelihood:

::: math-block
$$
L(D|\theta)=\prod_{i=1}^np_Y(Y_i|\theta).
$$
:::

The difference lies in how uncertainty about $\theta$ is represented. A frequentist analysis regards the true value $\theta_0$ as fixed and studies the behavior of an estimator across repeated samples. A Bayesian analysis specifies a prior density $q(\theta)$ and updates it after observing $D$. The prior expresses uncertainty about the parameter before these observations are incorporated.
::::

::::: math-passage
Bayes' theorem combines the likelihood and prior to give the posterior density:

::: math-block
$$
p(\theta|D)=\frac{L(D|\theta)q(\theta)}{\int
L(D|\theta')q(\theta')d\theta'},
$$
:::

The denominator normalizes the product so that the posterior integrates to one. It is also the marginal density of the observed dataset,

::: math-block
$$
L(D)=\int
L(D\mid\theta')q(\theta')\,d\theta'.
$$
:::

This integral often has no convenient closed form. Nevertheless, the unnormalized product $L(D\mid\theta)q(\theta)$ is enough to identify many familiar posterior families and to construct simulation algorithms. Once the posterior has been obtained, inference proceeds by summarizing its location, spread, and shape.
:::::

:::::: math-passage
Different posterior summaries answer different questions. A mean or median provides a measure of location, while the variance and quantiles describe uncertainty. For example, the posterior mean is

::: math-block
$$
E[\theta|D]=\int \theta
p(\theta|D)d\theta,
$$
:::

The posterior mode is the value at which the posterior density is largest:

::: math-block
$$
\theta^m=\arg\max_\theta p(\theta\mid
D).
$$
:::

Computing these summaries requires a specified prior. A common construction selects a parametric family,

::: math-block
$$
q(\theta)=q(\theta\mid \gamma),
$$
:::

in which $\gamma$ is a hyperparameter controlling features such as the prior mean or spread. Choices of $\gamma$ therefore belong to the model specification and should be distinguished from the unknown parameter $\theta$ about which inference is sought.
::::::

:::: math-passage
Bayesian estimators can also be studied from a repeated-sampling perspective. Consider the posterior mode. Because the marginal density of the data does not depend on the candidate value of $\theta$, its defining optimization can be written as

::: math-block
$$
\theta^m=\arg\max_\theta\frac{L(D|\theta)q(\theta)}{L(D)}=\arg\max_\theta
L(D|\theta)q(\theta),
$$
:::

In a regular model, the curvature of the log likelihood increases with the sample size because it is a sum of independent contributions. A fixed, sufficiently smooth prior then has diminishing influence near the likelihood maximum, and the posterior mode approaches the MLE. Consistency and efficiency follow under the same kind of identification and regularity conditions needed for the likelihood analysis, together with suitable prior support. The local expansion below makes those qualifications explicit.
::::

### What a large-sample approximation assumes

:::: math-passage
The normal approximation to a posterior is a local approximation, not a consequence of Bayes' formula alone. In a regular, identifiable finite-dimensional model, with the true parameter in the interior and a prior density positive and continuous near it, a second-order expansion gives

::: math-block
$$
\log\{L(\theta)q(\theta)\}=C-\tfrac12(\theta-\widehat\theta)^T\{nI(\theta_0)+o_p(n)\}(\theta-\widehat\theta)+o_p(1)
$$
:::

on $n^{-1/2}$ neighborhoods of the likelihood maximum. The resulting covariance is approximately $\{nI(\widehat\theta)\}^{-1}$. At boundaries, with nonidentifiability, or under substantial prior restrictions, this approximation need not hold. Bayesian credible intervals describe posterior probability; repeated-sampling coverage is an additional property to establish under assumptions.
::::

## 5.2 Conjugate posterior calculations {#section-2}

### Bernoulli observations and a beta prior

:::::: math-passage
Suppose $Y_1,\ldots,Y_n$ are conditionally independent Bernoulli observations with success probability $\theta$, and assign the prior $\theta\sim\operatorname{Beta}(\alpha,\lambda)$. The positive hyperparameters $\alpha$ and $\lambda$ control the prior distribution. Write $s=\sum_{i=1}^nY_i$ for the observed number of successes. The likelihood and prior density are, respectively,

::: math-block
$$
L(D\mid
\theta)=\theta^{s}(1-\theta)^{n-s},
$$
:::

::: math-block
$$
q(\theta)=\frac{\Gamma(\alpha+\lambda)}{\Gamma(\alpha)\Gamma(\lambda)}\theta^{\alpha-1}(1-\theta)^{\lambda-1}.
$$
:::

Multiplying them adds the exponents of $\theta$ and $1-\theta$. Terms independent of $\theta$ enter only the normalizing constant, leaving the posterior kernel

::: math-block
$$
p(\theta|D)\propto L(D|\theta)q(\theta)\propto
\theta^{s+\alpha-1}(1-\theta)^{n-s+\lambda-1}.
$$
:::
::::::

::::: math-passage
The exponents identify this kernel as a beta density, so the posterior belongs to the same family as the prior:

::: math-block
$$
\theta\mid D\sim
\mbox{Beta}\left(s+\alpha,n-s+\lambda\right),
$$
:::

Its normalized density follows directly from the beta normalizing constant:

::: math-block
$$
\begin{aligned}
p(\theta\mid
D)&=\frac{\Gamma(\alpha+\lambda+n)}{\Gamma(s+\alpha)\Gamma(n-s+\lambda)}\\
& \times\theta^{s+\alpha-1}(1-\theta)^{n-s+\lambda-1}.
\end{aligned}
$$
:::

This calculation illustrates why recognizing a kernel is useful. Instead of evaluating the marginal likelihood by integration, we identify a probability family whose integral is already known. The result supplies both the posterior distribution and its summaries, while making the contribution of the observed successes and failures explicit.
:::::

::::: math-passage
The beta--Bernoulli example can also be read as a decomposition of information. With $a=\alpha$, $b=\lambda$, and $s=\sum_iY_i$, the posterior mean is

::: math-block
$$
\frac{a+s}{a+b+n}=\frac{a+b}{a+b+n}\frac{a}{a+b}+\frac{n}{a+b+n}\overline
Y.
$$
:::

It averages a prior mean and an empirical mean with weights determined by their effective sample sizes. The posterior variance is

::: math-block
$$
\operatorname{Var}(\theta\mid
D)=\frac{(a+s)(b+n-s)}{(a+b+n)^2(a+b+n+1)}.
$$
:::

The next Bernoulli observation has success probability $(a+s)/(a+b+n)$ after integrating over the posterior. This predictive averaging will later be performed for missing observations rather than future observations.
:::::

::::: math-passage
Suppose $Y_1,\cdots, Y_n$ are a random sample from $N(\mu,\tau^{-1})$. First assume that $\tau$ is known and $\mu$ is unknown and that, as the prior for $\mu$,

::: math-block
$$
\mu\sim N(\mu_0,\tau_0^{-1}).
$$
:::

This gives

::: math-block
$$
\begin{aligned}
p(\mu|D,\tau)&\propto L(D|\mu,\tau)q(\mu)\\
&\propto\exp\left\{-\frac{\tau}{2}\sum_{i=1}^n(Y_i-\mu)^2\right\}\exp\left\{-\frac{\tau_0}{2}(\mu-\mu_0)^2\right\}\\
&\propto\exp\left\{-\frac{n\tau}{2}(\mu^2-2\overline
Y\mu)\right\}\exp\left\{-\frac{\tau_0}{2}(\mu^2-2\mu_0\mu)\right\}\\
&\propto\exp\left\{-\frac{n\tau+\tau_0}{2}\left(\mu-\frac{n\tau\overline
Y+\tau_0\mu_0}{n\tau+\tau_0}\right)^2\right\}.
\end{aligned}
$$
:::
:::::

:::::: math-passage
Hence,

::: math-block
$$
\mu\mid D,\tau \sim
N\left(\frac{n\tau\overline
Y+\tau_0\mu_0}{n\tau+\tau_0},(n\tau+\tau_0)^{-1}\right).
$$
:::

Next, assume that $\mu$ is known and $\tau$ is unknown and that

::: math-block
$$
\tau\sim\mbox{Gamma}(2^{-1}\delta_0,2^{-1}\gamma_0),
$$
:::

i.e., $q(\tau)\propto
\tau^{\delta_0/2-1}\exp(-\gamma_0\tau/2)$. The posterior distribution of $\tau$ is

::: math-block
$$
\begin{aligned}
p(\tau|D,\mu)&\propto L(D|\mu,\tau)q(\tau)\\
&\propto\tau^{n/2}\exp\left\{-\frac{\tau\sum_{i=1}^n(Y_i-\mu)^2}{2}\right\}\tau^{\delta_0/2-1}\exp(-\gamma_0\tau/2)\\
&\propto\tau^{(n+\delta_0)/2-1}\exp\left\{-\frac{\tau\left(\sum_{i=1}^n(Y_i-\mu)^2+\gamma_0\right)}{2}\right\}.
\end{aligned}
$$
:::
::::::

::::::: math-passage
It follows that

::: math-block
$$
\tau\mid
D,\mu\sim\mbox{Gamma}\left(\frac{n+\delta_0}{2},\frac{\sum_{i=1}^n(Y_i-\mu)^2+\gamma_0}{2}\right).
$$
:::

Finally, assume that both $\mu$ and $\tau$ are unknown. Specify the following prior $q(\mu,\tau)=
q(\mu\mid \tau)q(\tau)$, where

::: math-block
$$
\mu\mid \tau \sim
N(\mu_0,\tau^{-1}\tau_0^{-1}),
$$
:::

and

::: math-block
$$
\tau\sim
\mbox{Gamma}(2^{-1}\delta_0,2^{-1}\gamma_0).
$$
:::

It can be shown that

::: math-block
$$
p(\mu,\tau|D)=p(\mu|\tau,D)p(\tau|D)=\mbox{Normal}\times\mbox{Gamma}.
$$
:::
:::::::

### Completing the normal--gamma calculation

:::::: math-passage
Write $\kappa_0=\tau_0$ for the prior precision multiplier, reserving $\tau$ for the sampling precision. Define

::: math-block
$$
\kappa_n=\kappa_0+n,\quad
\mu_n=\frac{\kappa_0\mu_0+n\overline Y}{\kappa_n},\quad
a_n=\frac{\delta_0+n}{2},
$$
:::

::: math-block
$$
b_n=\frac12\left\{\gamma_0+\sum_i(Y_i-\overline
Y)^2+\frac{\kappa_0n}{\kappa_n}(\overline Y-\mu_0)^2\right\}.
$$
:::

To obtain these quantities, expand the two sums of squares and complete the square in $\mu$:

::: math-block
$$
\sum_i(Y_i-\mu)^2+\kappa_0(\mu-\mu_0)^2
=\kappa_n(\mu-\mu_n)^2+\sum_i(Y_i-\overline
Y)^2+\frac{\kappa_0n}{\kappa_n}(\overline Y-\mu_0)^2.
$$
:::

Consequently $\tau\mid
D\sim\operatorname{Gamma}(a_n,b_n)$ and $\mu\mid\tau,D\sim
N\{\mu_n,(\kappa_n\tau)^{-1}\}$. Integrating out $\tau$ gives a Student $t$ distribution for $\mu$ with $2a_n$ degrees of freedom, location $\mu_n$, and scale squared $b_n/(a_n\kappa_n)$. The predictive distribution for a new $Y$ is also Student $t$, with scale squared $b_n(\kappa_n+1)/(a_n\kappa_n)$. The extra term in the predictive scale is the future observation's residual variation.
::::::

## 5.3 Conjugacy, computation, and priors {#section-3}

For the first example, a beta prior on $\theta$ leads to a beta posterior for $\theta$. In the first scenario of the second example, a normal prior on $\mu$ yields a normal posterior for $\mu$. In the second scenario of the second example, a gamma prior for $\tau$ yields a gamma posterior for $\tau$. When the posterior distribution of a parameter is of the same family as the prior distribution, such prior distributions are called conjugate priors. Other conjugate priors include

::: {#tab:conjugate}
:::

::: {.table-scroll aria-label="Scrollable data table" tabindex="0"}
+-----------------------------------------+----------------------------+---+
| Likelihood                              | Conjugate prior            |   |
+:=======================================:+:==========================:+:=:+
| Poisson$(\theta)$                       | Gamma$(\delta_0,\gamma_0)$ |   |
+-----------------------------------------+----------------------------+---+
| Gamma$(\alpha,\lambda)$, $\alpha$ known | Gamma$(\delta_0,\gamma_0)$ |   |
+-----------------------------------------+----------------------------+---+
| Beta$(\alpha,1)$                        | Gamma$(\delta_0,\gamma_0)$ |   |
+-----------------------------------------+----------------------------+---+

: Examples of conjugate priors.
:::

Conjugacy is convenient, but many posterior distributions fall outside families that can be sampled directly. Gibbs sampling provides one way to handle this situation. As in the Monte Carlo E step of Chapter 3, the idea is to update one variable or block at a time, conditioning on the most recent values of the remaining components. Here the target is a parameter posterior rather than a conditional distribution of missing values. Each conditional update can use direct simulation when available; otherwise it needs an appropriate sampling method, such as adaptive rejection sampling for a log-concave density or a Metropolis--Hastings step.

Specifically, suppose $\theta=(\theta_1,\cdots,\theta_p)^{\mathrm{T}}$. To draw $\theta$ from the posterior distribution $[\theta\mid D]$:

1.  Initialize $\theta_1^{(0)},\cdots,\theta_p^{(0)}$;

2.  For $t=1,2,\cdot$, do

    1.  :::: math-passage
        ::: math-block
        $$
\theta_1^{(t)}\sim[\theta_1\mid
        \theta_2^{(t-1)},\theta_3^{(t-1)},\cdots,\theta_p^{(t-1)},
        D]
$$
        :::
        ::::

    2.  :::: math-passage
        ::: math-block
        $$
\theta_2^{(t)}\sim[\theta_2\mid
        \theta_1^{(t)},\theta_3^{(t-1)},\cdots,\theta_p^{(t-1)},
        D]
$$
        :::
        ::::

    3.  :::: math-passage
        ::: math-block
        $$
\ldots
$$
        :::
        ::::

    4.  :::: math-passage
        ::: math-block
        $$
\theta_p^{(t)}\sim[\theta_p\mid
        \theta_1^{(t)},\theta_2^{(t)},\cdots,\theta_{p-1}^{(t)},
        D]
$$
        :::
        ::::

Each step in 2 is completed by the adaptive rejection sampling, which requires only that we know the kernel of the conditional distribution. For the one-dimensional conditional distribution of $\theta_j$, the kernel is $L(D\mid \theta)q(\theta)$, viewed as a function of $\theta_j$. Because samples from the early iterations are not from the target posterior, it is common to discard these samples. The discarded iterations are often referred to as the "burn-in" period.

As an illustration, a run of $N=3000$ iterations might discard the first 1000 and retain the remaining 2000 for posterior summaries. These draws can be used to estimate means and other features, including a mode or a highest-density region when an appropriate density estimate is available. The numerical choices do not themselves establish convergence, however, and dependence between retained draws affects the precision of the summaries. This motivates a closer look at the transition and at the distinction between its invariant distribution and its mixing behavior.

### A Gibbs transition and its diagnostic limits

For two blocks, a sweep samples $\theta_1'\sim
p(\theta_1\mid\theta_2,D)$ and then $\theta_2'\sim
p(\theta_2\mid\theta_1',D)$. If the input has the posterior distribution, the output does too: multiplying the first conditional by the old marginal recovers the joint distribution, and replacing the second block by its conditional preserves that joint distribution again. This proves invariance. It does not prove that a chain started arbitrarily reaches that distribution quickly. Irreducibility, recurrence, and mixing must be considered separately.

A fixed burn-in length is only a starting choice; it must be assessed against the behavior of the chain. Examine trace plots from dispersed starting values, effective sample sizes, and Monte Carlo errors for the quantities to be reported. Adjacent retained draws remain dependent; thinning alone does not establish convergence. A conditional kernel is sufficient for Metropolis--Hastings ratios, but adaptive rejection sampling additionally needs the relevant log-concavity conditions.

### Prior information and parameterization

:::: math-passage
When little substantive prior information is available, one may seek a prior that changes slowly over the region favored by the likelihood. Such choices are often described as vague, flat, or noninformative; reference-prior constructions provide a more formal approach to related questions. These terms do not make a prior neutral in every parameterization. The simplest example is a constant density,

::: math-block
$$
q(\theta)\propto 1.
$$
:::

whose behavior depends on the parameter space and on the coordinate in which the density is declared constant. On an unbounded parameter space it need not integrate to a finite value, so posterior propriety must be checked separately.
::::

:::::: math-passage
This is an improper prior in the sense that

::: math-block
$$
\int q(\theta)=\infty.
$$
:::

Improper priors are sometimes useful because they may lead to proper posterior distributions, i.e.,

::: math-block
$$
\int L(\theta\mid
D)q(\theta)d\theta<\infty.
$$
:::

However, there is another drawback of using the "uniform" prior of $q(\theta)\propto 1$, which is that it depends on the parameterization. Suppose $\psi$ is another parameterization of $\theta$. Under the "uniform" prior for $\theta$, the prior for $\psi$ is

::: math-block
$$
q(\psi)\propto
\left|\det\frac{d\theta}{d\psi}\right|,
$$
:::

which is different from the "uniform" prior for $\psi$.
::::::

:::::: math-passage
An alternative choice for non-informative priors that overcomes the non-uniqueness issue is to use

::: math-block
$$
q(\theta)\propto \mathcal
I(\theta)^{1/2},
$$
:::

where $\mathcal
I(\theta)$ is the information for $\theta$ based on one observation. In the case that $\theta$ is a vector

::: math-block
$$
q(\theta)\propto \left\{\det\mathcal
I(\theta)\right\}^{1/2}.
$$
:::

For example, suppose that, given $\theta$, $Y_1,\cdots, Y_n$ are a random sample from $N(\theta,1)$. Then $\mathcal I(\theta)=1$. So the information-based non-informative prior coincides with the "uniform" prior, and

::: math-block
$$
p(\theta|D)\propto
L(D|\theta)\propto\exp\left(-\frac{n(\theta-\overline
Y)^2}{2}\right).
$$
:::
::::::

::::::: math-passage
So the posterior is proper and

::: math-block
$$
\theta\mid D\sim N(\overline Y, n^{-1}).
$$
:::

As another example, suppose that, given $\theta$, $Y_1,\cdots, Y_n$ are a random sample from $\mbox{Poisson}(\theta)$. Because $\mathcal I(\theta)=\theta^{-1}$, we take the prior to be

::: math-block
$$
q(\theta)\propto
\theta^{-1/2}.
$$
:::

This gives

::: math-block
$$
p(\theta|D)\propto
L(D|\theta)\theta^{-1/2}\propto\theta^{n\overline
Y-1/2}\exp(-n\theta)=\theta^{n\overline Y+1/2-1}\exp(-n\theta).
$$
:::

So

::: math-block
$$
\theta\mid D\sim\mbox{Gamma}(n\overline
Y+1/2,n).
$$
:::
:::::::

For a scalar transformation $\eta=g(\theta)$, a density transforms as $q_\eta(\eta)=q_\theta(\theta)|d\theta/d\eta|$. Fisher information transforms as $I_\eta=I_\theta(d\theta/d\eta)^2$, so Jeffreys' construction $q(\theta)\propto\sqrt{I_\theta}$ respects this change of coordinates. A flat density generally does not. Neither invariance nor the name "noninformative" guarantees a proper posterior. One must check that $\int
L(\theta)q(\theta)d\theta$ is finite before treating its normalized version as a probability density.

### From parameter inference to prediction

::::: math-passage
The posterior is also the basis for prediction. Suppose the observations $D=(Y_1,\ldots,Y_n)$ arise from $p_Y(y\mid\theta)$ and the prior is $q(\theta)$. Bayes' theorem gives

::: math-block
$$
p(\theta|D)=\frac{L(D|\theta)q(\theta)}{\int
L(D|\theta')q(\theta')d\theta'}\propto
L(D|\theta)q(\theta).
$$
:::

Now consider a future observation $Y$ that is independent of the recorded sample conditional on $\theta$. Its distribution is obtained by averaging the sampling density over parameter uncertainty:

::: math-block
$$
p(Y|D)=\int
p_Y(Y|\theta)p(\theta|D)d\theta,
$$
:::

This is the posterior predictive distribution. It includes both variation in a new outcome at a fixed parameter and uncertainty about that parameter. The same averaging will be central to multiple imputation, where the outcome to be predicted is an unrecorded part of an existing observation.
:::::

::::: math-passage
In the example of $N(\theta, 1)$ and $q(\theta)\propto 1$, we had that

::: math-block
$$
\theta\mid D\sim N(\overline Y,
n^{-1}).
$$
:::

Since for another $Y$ (independent of $D$) we have that $Y\mid \theta\sim N(\theta, 1)$, after some calculus, we find that

::: math-block
$$
Y\mid D\sim
N(\overline Y, 1+n^{-1}).
$$
:::

In general, the posterior distribution does not have an analytic form. So the following two steps are needed in order to draw Y from its predictive distribution. For $j=1,2,\cdots$,
:::::

1.  Draw $\theta^{(j)}$ from $[\theta\mid D]$ (possibly using Gibbs sampler);

2.  Draw $Y^{(j)}$ from $p_Y(Y\mid \theta^{(j)})$.

Then, $Y^{(1)},Y^{(2)},\cdots$ constitute a random sample from $[Y\mid
D]$.

Predictive simulation is a two-stage integration. A draw $\theta^{(k)}\sim p(\theta\mid D)$ followed by $Y^{*(k)}\sim
p(Y^*\mid\theta^{(k)})$ has marginal density $\int p(Y^*\mid\theta)p(\theta\mid
D)d\theta$. In the known-variance normal example, conditional predictive variance is $1$ and posterior mean variance is $1/n$. The law of total variance therefore gives $1+1/n$. Fixing $\theta$ at its posterior mean would discard the second term.

## 5.4 Data augmentation {#section-4}

:::: math-passage
The same reasoning applies to missing values, except that their conditional distribution must respect the observed part of the same record. Under MAR, distinct data and response parameters, and a factorized prior, the joint posterior can be written

::: math-block
$$
p(\theta,D_{\mathrm{mis}}\mid
D_{\mathrm{obs}})\propto
p(D_{\mathrm{obs}},D_{\mathrm{mis}}\mid\theta)q(\theta).
$$
:::

Under nonignorability the response-mechanism factor remains in this expression. Missing observations are therefore latent variables with a distribution, not unknown constants to be replaced once and then treated as data.
::::

::::: math-passage
With full data D, Bayesian inference is based on the posterior

::: math-block
$$
\begin{equation}\tag{5.1}\label{eq:full_bayes}
    [\theta\mid D].\end{equation}
$$
:::

if the posterior density $p(\theta\mid D)$ does not have a closed-form, the Gibbs sampler can be used to draw from ($\ref{eq:full_bayes}$). When missing data are present, denote $D=(D_{\mathrm{obs}},
D_{\mathrm{mis}})$. Then the Bayesian analysis is focused on $[\theta\mid D_{\mathrm{obs}}]$, the kernel of which usually does not have a closed form. Assuming data are MAR,

::: math-block
$$
\begin{aligned}
    p(\theta|D_{\mathrm{obs}})&\propto
L(D_{\mathrm{obs}}|\theta)q(\theta)\\
    &=\left(\int L(D|\theta)dD_{\mathrm{mis}}\right)q(\theta).
\end{aligned}
$$
:::
:::::

The obstacle is computational: the observed-data posterior may be difficult to evaluate even when complete-data calculations are straightforward. A two-block Gibbs construction connects the two problems. For a joint distribution $[A,B]$, alternating draws from $[A\mid B]$ and $[B\mid A]$ defines a chain with the joint distribution as its invariant law. Under suitable convergence conditions, retained pairs $(A^{(j)},B^{(j)})$ can therefore be used to approximate joint expectations, and the retained $A^{(j)}$ approximate the marginal law $[A]$. Applying this idea to parameters and missing values avoids the need to integrate the missing values out analytically.

This entails the following iterative sampling for drawing from $[\theta\mid D_{\mathrm{obs}}]$. First initialize $\theta=\theta^{(0)}$ (using, e.g., complete-case MLE). Then for the $(j+1)$th iteration:

1.  Draw $D_{\mathrm{mis}}^{(j+1)}$ from $[D_{\mathrm{mis}}\mid
    D_{\mathrm{obs}};\theta^{(j)}]$

2.  Draw $\theta^{(j+1)}$ from $[\theta\mid D_{\mathrm{obs}},
    D_{\mathrm{mis}}^{(j+1)}]$

where each step possibly entails one iteration of Gibbs sampling through individual components of the vector to be drawn. Observe that step 2 is the same as the sampling procedure in the full data scenario ($\ref{eq:full_bayes}$), which possibly involves a Gibbs sampler using the kernel $L(D^{(j+1)}\mid \theta)q(\theta)$, $D^{(j+1)}=(D_{\mathrm{obs}},D_{\mathrm{mis}}^{(j+1)})$. So, Bayesian analysis with missing data only adds one more layer of sampling to the full-data framework, that is step 1, sampling from conditional distribution of missing data given the observed data and the previous draw of the parameter.

Data augmentation thus treats the missing values as an additional unknown block. Conditional on a completed dataset, the parameter update is a familiar complete-data posterior calculation; conditional on the parameter, the missing-value update uses the same distribution that appears in a Monte Carlo E step. Its kernel is $L(D\mid\theta)$ viewed as a function of $D_{\mathrm{mis}}$. This correspondence explains the close connection with EM, but it also exposes a computational cost: the dimension of the missing-value block generally grows with the sample size. The two methods use that block differently, as the comparison below shows.

EM and data augmentation use the same conditional models for different purposes. EM replaces the complete-data log likelihood by its conditional expectation and optimizes over a candidate parameter. Data augmentation draws both missing values and parameters and targets a joint posterior. Replacing the parameter draw in data augmentation by a maximizer does not produce posterior draws; replacing an EM expectation by one random imputation introduces Monte Carlo error. The distinction concerns the target of the calculation, not merely its software implementation.

## 5.5 Bayesian regression with missing covariates {#section-5}

We specialize this general approach to the problem of regression analysis with covariates that are missing at random. Let the conditional distribution of response $Y$ given covariates $Z$ be $p(Y\mid Z;\beta)$, where $\beta$ is the parameter of interest. In order to fully specify the full-data likelihood, we need to postulate a model for covariate distribution. Let $p(Z\mid \alpha)$ denote the marginal density of $Z$, where $\alpha$ is a set of nuisance parameters. Write $\theta=(\beta,\alpha)$. For prior construction for $\theta$, it is convenient to decompose the prior as $q(\theta)=q(\beta\mid \alpha)q(\alpha)$, or even $q(\theta)=q(\beta)q(\alpha)$. The results typically will not vary much if the priors are chosen to be non-informative.

:::::: math-passage
The first option $q(\theta)=q(\beta\mid
\alpha)q(\alpha)$ is appealing when using the "information principle" to construct the prior. Specifically, denote $\mathcal I(\beta\mid Z)$ as the information of $\beta$ resulting from the conditional likelihood $p(Y\mid
Z;\beta)$, that is,

::: math-block
$$
\mathcal
I(\beta|Z)=\int\left(\frac{\partial \log
p(y|Z;\beta)}{\partial\beta}\right)^{\otimes 2}p(y|Z;\beta)dy.
$$
:::

Then, the marginal information of $\beta$ is

::: math-block
$$
\begin{equation}\tag{5.2}\label{eq:beta_info}
    \mathcal I(\beta|\alpha)=E[\mathcal I(\beta|Z)|\alpha]=\int\mathcal
I(\beta|z)p(z|\alpha)dz.\end{equation}
$$
:::

Then, using the information principle, we may choose

::: math-block
$$
q(\beta|\alpha)=\left\{\det\mathcal
I(\beta|\alpha)\right\}^{1/2}.
$$
:::
::::::

:::::: math-passage
Sometimes calculating $\mathcal
I(\beta\mid \alpha)$ through ($\ref{eq:beta_info}$) is difficult because the integral may not be analytic. In such cases, one has to construct non-informative priors by other means. Write $D_Y=(Y_1,\cdots, Y_n)$ and $D_Z=(Z_1,\cdots, Z_n)$. Denote

::: math-block
$$
L_{Y|Z}(D_Y|D_Z;\beta)=\prod_{i=1}^n
p(Y_i|Z_i;\beta),
$$
:::

and

::: math-block
$$
L_Z(D_Z|\alpha)=\prod_{i=1}^n
p(Z_i|\alpha).
$$
:::

This gives

::: math-block
$$
L(D|\theta)=L_{Y|Z}(D_Y|D_Z;\beta)L_Z(D_Z|\alpha).
$$
:::
::::::

:::: math-passage
Thus, the joint distribution of $(D,\theta)$ is

::: math-block
$$
L(D|\theta)q(\theta)=L_{Y|Z}(D_Y|D_Z;\beta)L_Z(D_Z|\alpha)q(\beta|\alpha)q(\alpha).
$$
:::

We describe the Gibbs sampling steps for $D_{\mathrm{mis}}, \beta,$ and $\alpha$ and the associated kernel functions for drawing from the conditional distributions. For simplicity, the iteration index is omitted. Write $Z_i=(Z_{i,\mathrm{obs}},Z_{i,\mathrm{mis}})$ and $Z_{i,\mathrm{mis}}$ is missing at random.
::::

1.  For $i=1,\cdots, n$, sample $Z_{i,\mathrm{mis}}$ from $[Z_{i,\mathrm{mis}}\mid Y_i,
    Z_{i,\mathrm{obs}};\theta]$, whose kernel is $p(Y_i|Z_{i,\mathrm{obs}},Z_{i,\mathrm{mis}};\beta)p(Z_{i,\mathrm{obs}},Z_{i,\mathrm{mis}}|\alpha)$.

2.  Sample $\beta$ from $[\beta\mid D;\alpha]$, whose kernel is $L_{Y|Z}(D_Y|D_Z;\beta)q(\beta|\alpha)$.

3.  Sample $\alpha$ from $[\alpha\mid D;\beta]$, whose kernel is $L_Z(D_Z|\alpha)q(\beta|\alpha)q(\alpha)$.

The factorization $p(Y,Z\mid\beta,\alpha)=p(Y\mid
Z,\beta)p(Z\mid\alpha)$ explains the two parameter updates. Given completed covariates, the regression likelihood supplies information about $\beta$ and the covariate likelihood supplies information about $\alpha$. If the prior is $q(\beta,\alpha)=q(\beta\mid\alpha)q(\alpha)$, the $\alpha$ update still contains $q(\beta\mid\alpha)$; it may not be omitted simply because $\beta$ is being held fixed. With missing covariates, their imputation conditional also contains the response likelihood. Ignoring the observed response while imputing a predictor can distort the regression association.

## 5.6 Normal regression: full conditional distributions {#section-6}

### Dimensions and a working prior

In the normal example let $W_i$ contain the $p$ stochastic covariates and put $Z_i=(1,W_i^T)^T$ in the response regression. Thus $\beta$ has $p+1$ entries, $W_i\sim N_p(\mu,V^{-1})$, and $Y_i\mid W_i\sim N(\beta^TZ_i,\tau^{-1})$. In the covariate-density formulas below the original notation $Z$ refers to its stochastic components $W$; an intercept has no nonsingular normal density.

:::: math-passage
The full conditional calculations use the specified working joint prior

::: math-block
$$
q(\beta,\tau,\mu,V)\propto\tau^{(p-1)/2}|V|^{-1},\qquad
\tau>0,\quad V\text{ positive definite}.
$$
:::

This improper prior must be accompanied by a posterior propriety check. It should not be described as the general multivariate Jeffreys prior. For $(\mu,V)$ in a $p$-variate normal model, using the unique entries of symmetric $V$ as coordinates, the information determinant is proportional to $|V|^{-p}$ and the joint Jeffreys density is proportional to $|V|^{-p/2}$. The $|V|^{-1/2}$ expression in the lecture calculation holds for that block only when $p=1$. Moreover, a factor depending on conditioning parameters cannot be discarded when defining a joint prior from improper conditional pieces. Specifying the working joint prior directly removes that ambiguity while preserving the following Gibbs example.
::::

:::: math-passage
Let

::: math-block
$$
Y=\beta^{\mathrm{T}}Z+\epsilon,
$$
:::

where $Z$ consists of 1 and a $p$-dimensional vector of continuous covariates and $\epsilon\sim
N(0,\tau^{-1})$. Assume that the last $p$ components of $Z$ follow multivariate normal with mean $\mu$ and variance $V^{-1}$. Write $\theta=(\beta,\tau,\mu,V)$. We have seen how to compute the MLE of $\theta$ using EM algorithm (with explicit E and M steps). Now we conduct Bayesian inference. For prior construction, we first use the information principle to construct non-informative priors for $(\beta,\tau)$ given $(\mu, V)$.
::::

:::::: math-passage
By straightforward calculation,

::: math-block
$$
\mathcal
I(\beta,\tau\mid Z)=\left(\begin{array}{cc}\tau Z^{\otimes
2}&0\\0&2^{-1}\tau^{-2}\end{array}\right),
$$
:::

so that

::: math-block
$$
\mathcal
I(\beta,\tau|\alpha)=\left(\begin{array}{cc}\tau E[Z^{\otimes 2}|\mu,
V]&0\\0&2^{-1}\tau^{-2}\end{array}\right).
$$
:::

Hence,

::: math-block
$$
\det \mathcal I(\beta,\tau|\alpha)\propto
\tau^{-2}\det\{\tau E[Z^{\otimes 2}|\mu, V]\}=\tau^{p-1}\det
E[Z^{\otimes 2}|\mu, V],
$$
:::

where we recall that $E[Z^{\otimes 2}\mid \mu, V]$ is a $(p+1)$-dimensional square matrix.
::::::

:::::: math-passage
It remains to calculate $\det E[Z^{\otimes
2}\mid \mu, V]$. Because we can write

::: math-block
$$
Z\sim
N\left\{\left(\begin{array}{c}1\\\mu\end{array}\right),\left(\begin{array}{cc}0&0\\0&V^{-1}\end{array}\right)\right\}
$$
:::

It is easy to see that

::: math-block
$$
E[Z^{\otimes 2}\mid
\mu,
V]=\left(\begin{array}{cc}1&\mu^{\mathrm{T}}\\\mu&\mu^{\otimes
2}+V^{-1}\end{array}\right).
$$
:::

To calculate its determinant, we use the general rule that

::: math-block
$$
\det\left(\begin{array}{cc}A&B\\C&D\end{array}\right)=\det(A)\det(D-CA^{-1}B).
$$
:::
::::::

:::::: math-passage
Therefore we have

::: math-block
$$
\det E[Z^{\otimes
2}|\mu, V]=|V|^{-1},
$$
:::

where $|A|=\det
A$ for any matrix $A$. Thus,

::: math-block
$$
q(\beta,\tau|\mu, V)\propto \{\det \mathcal
I(\beta,\tau|\mu,
V)\}^{\frac{1}{2}}\propto\tau^{\frac{p-1}{2}}|V|^{-\frac{1}{2}}.
$$
:::

For the multivariate normal block, $\det\mathcal I(\mu,V)\propto|V|^{-p}$ and its joint Jeffreys density is $|V|^{-p/2}$. Rather than multiplying improperly normalized conditional pieces, we specify the following working joint prior for this example:

::: math-block
$$
q(\beta,\tau,\mu, V)\propto
\tau^{\frac{p-1}{2}}|V|^{-1}.
$$
:::
::::::

:::: math-passage
The normal model permits explicit updates of the parameter blocks. Given the current parameter values, missing covariates are drawn using the conditional multivariate normal distribution already derived for the E step. With a completed dataset in hand, the remaining updates concern $\beta$, $\tau$, $\mu$, and $V$. To derive the regression update, let $\widehat\beta$ be the complete-data least-squares estimate and complete the square in the residual sum of squares:

::: math-block
$$
\begin{aligned}
    L_{Y|Z}(D_Y|D_Z;\beta,\tau)&\propto\tau^{n/2}\exp\left(-\frac{\tau\sum_{i=1}^n(Y_i-\beta^{\mathrm{T}}Z_i)^2}{2}\right)\\
    &\propto\tau^{n/2}\exp\left(-\frac{\tau\sum_{i=1}^n(Y_i-\widehat\beta^{\mathrm{T}}Z_i)^2}{2}\right)\\
    &\times
\exp\left(-\frac{\tau(\beta-\widehat\beta)^{\mathrm{T}}\sum_{i=1}^nZ_i^{\otimes
2}(\beta-\widehat\beta)}{2}\right).
\end{aligned}
$$
:::

The first exponential factor does not depend on the candidate regression coefficient. The second identifies the conditional normal kernel in $\beta$ and shows how the design matrix determines its posterior precision.
::::

::::: math-passage
Likewise,

::: math-block
$$
\begin{aligned}
    &L_Z(D_Z|\mu,V)\\
    \propto&|V|^{n/2}\exp\left(-\frac{\mbox{tr}\left\{\sum_{i=1}^n(Z_i-\mu)^{\otimes
2}V\right\}}{2}\right)\\
    =&|V|^{n/2}\exp\left(-\frac{\sum_{i=1}^n(Z_i-\overline
Z)^{\mathrm{T}}V(Z_i-\overline Z)+n(\mu-\overline
Z)^{\mathrm{T}}V(\mu-\overline Z)}{2}\right).
\end{aligned}
$$
:::

It can be seen that

::: math-block
$$
\begin{aligned}
    [\beta\mid D;\tau,\mu,V]&\propto
\exp\left(-\frac{\tau(\beta-\widehat\beta)^{\mathrm{T}}\sum_{i=1}^nZ^{\otimes
2}(\beta-\widehat\beta)}{2}\right)\\
    &\sim N\left\{\widehat\beta, \left(\tau\sum_{i=1}^n Z_i^{\otimes
2}\right)^{-1}\right\}.
\end{aligned}
$$
:::
:::::

::::: math-passage
Also,

::: math-block
$$
\begin{aligned}
    [\tau\mid D;\beta,\mu,V]&\propto
\tau^{\frac{n+p-1}{2}}\exp\left(-\frac{\tau\sum_{i=1}^n(Y_i-\beta^{\mathrm{T}}Z_i)^2}{2}\right)\\
    &\sim
\mbox{Gamma}\left(\frac{n+p+1}{2},\frac{\sum_{i=1}^n(Y_i-\beta^{\mathrm{T}}Z_i)^2}{2}\right).
\end{aligned}
$$
:::

Further,

::: math-block
$$
\begin{aligned}
    [\mu\mid D;\beta,\tau,V]&\propto\exp\left(-\frac{n(\mu-\overline
Z)^{\mathrm{T}}V(\mu-\overline Z)}{2}\right)\\
    &\sim N(\overline Z, n^{-1}V^{-1}).
\end{aligned}
$$
:::
:::::

Finally, to see what $[V\mid
D,\beta,\tau,\mu]$ is, we need the following definition of distribution of positive definite matrices.

> :::: math-passage
> **Definition 5.1 (Wishart Distribution).** A random positive definite $p\times p$ matrix $V$ follows Wishart$(\delta,\Sigma)$ if its density is given by
>
> ::: math-block
> $$
p(V|\delta,\Sigma)=2^{-\delta
> p/2}|\Sigma|^{\delta/2}\Gamma_p(\delta/2)^{-1}|V|^{\frac{\delta-p-1}{2}}\exp\left(-\mbox{tr}(\Sigma
> V)/2\right),
$$
> :::
>
> where $\Gamma_p$ is the $p$-variate gamma function.
> ::::

:::: math-passage
Then, we have that

::: math-block
$$
\begin{aligned}
[V|D,\beta,\tau,\mu]&\propto|V|^{n/2-1}\exp\left(-\frac{\mbox{tr}\left\{\sum_{i=1}^n(Z_i-\mu)^{\otimes
2}V\right\}}{2}\right)\\
    &\sim \mbox{Wishart}\left(n+p-1,\sum_{i=1}^n(Z_i-\mu)^{\otimes
2}\right).
\end{aligned}
$$
:::

All four conditional distributions of the components of the parameter have closed forms and can be directly sampled from.
::::

### Deriving and checking the full conditionals

:::: math-passage
Let $A=\sum_i Z_iZ_i^T$, $\widehat\beta=A^{-1}\sum_iZ_iY_i$, and $\operatorname{RSS}(\beta)=\sum_i(Y_i-\beta^TZ_i)^2$. Since

::: math-block
$$
\operatorname{RSS}(\beta)=\operatorname{RSS}(\widehat\beta)+(\beta-\widehat\beta)^TA(\beta-\widehat\beta),
$$
:::

the conditional distribution of $\beta$ is $N\{\widehat\beta,(\tau A)^{-1}\}$. Multiplying the likelihood power $\tau^{n/2}$ by the prior power $\tau^{(p-1)/2}$ gives the gamma shape $(n+p+1)/2$ for $\tau\mid\beta,D$, with rate $\operatorname{RSS}(\beta)/2$.
::::

:::: math-passage
Similarly, completing the square in $\sum_i(W_i-\mu)^TV(W_i-\mu)$ gives $\mu\mid V,D\sim N_p\{\overline
W,(nV)^{-1}\}$. The conditional density of $V$ is proportional to

::: math-block
$$
|V|^{(n-2)/2}\exp\{-\tfrac12\operatorname{tr}(S_\mu
V)\},\qquad S_\mu=\sum_i(W_i-\mu)(W_i-\mu)^T.
$$
:::

In the chapter's *rate-matrix* Wishart convention, this is $\operatorname{Wishart}(n+p-1,S_\mu)$. A software routine that expects a scale matrix instead requires $S_\mu^{-1}$. Its mean is $(n+p-1)S_\mu^{-1}$. The degrees of freedom must exceed $p-1$ and the rate matrix must be positive definite. Full-rank response designs, nonzero residual sums of squares, and sufficient covariate information are essential; the mere existence of formal conditional expressions does not establish propriety of the joint posterior with missing data.
::::

## 5.7 Generalized linear models and nonignorability {#section-7}

:::: math-passage
The preceding conjugate calculations rely on the quadratic structure of the normal likelihood. In a generalized linear model, the corresponding parameter conditionals usually have no familiar closed form. For a canonical link and $\phi=1$, for example, the response likelihood is

::: math-block
$$
\begin{aligned}
    L_{Y|Z}(D_Y|D_Z;\beta)&\propto\prod_{i=1}^n\exp\left(\beta^{\mathrm{T}}Z_iY_i-a(\beta^{\mathrm{T}}Z_i)\right)\\
    &=\exp\left(\beta^{\mathrm{T}}\sum_{i=1}^nZ_iY_i-\sum_{i=1}^na(\beta^{\mathrm{T}}Z_i)\right).
\end{aligned}
$$
:::

The nonlinear cumulant term makes it difficult to choose a prior $q(\beta\mid\alpha)$ that yields a recognizable posterior family. An information-based prior can also be difficult to compute because it requires an expectation over the covariate distribution. These difficulties affect computation, rather than the underlying posterior construction: the likelihood and prior still determine a conditional kernel that a suitable simulation method can use.
::::

:::::: math-passage
This is because in this case

::: math-block
$$
\mathcal
I(\beta\mid Z)=\dot\mu(\beta^{\mathrm{T}}Z)Z^{\otimes 2},
$$
:::

and for nonlinear link $\mu(\cdot)$, the expectation $E[\mathcal
I(\beta|Z)|\alpha]$ typically does not have a closed form. In such cases, it is convenient to take $q(\beta\mid \alpha)\propto 1$ if this results in proper posteriors. Then, Gibbs sampler generally needs to sweep through each univariate component of $\beta$ and $\alpha$. In doing so, notice that

::: math-block
$$
[\beta|D,\alpha]\propto
L_{Y|Z}(D_Y|D_Z;\beta)\propto
\exp\Big(\beta^{\mathrm{T}}\sum_{i=1}^nZ_iY_i-\sum_{i=1}^na(\beta^{\mathrm{T}}Z_i)\Big),
$$
:::

and

::: math-block
$$
[\alpha|D,\beta]\propto
L_Z(D_Z|\alpha)q(\alpha).
$$
:::
::::::

::::: math-passage
Nonignorable missingness introduces another component into the model. Let $R$ identify the observation pattern, and specify a response-mechanism model with parameter $\psi$:

::: math-block
$$
\pi(R\mid Y;\psi),
$$
:::

The full record is now $(Y,R)$, and its likelihood includes the probability of the recorded pattern as well as the data density:

::: math-block
$$
L(D|\theta,\psi)=\prod_{i=1}^np_Y(Y_i|\theta)\pi(R_i|Y_i;\psi).
$$
:::

A joint prior $q(\theta,\psi)$ completes the Bayesian specification. Unlike the ignorable case, the response-mechanism factor generally affects the distribution used to fill in missing values. Consequently, changing that model can change the posterior for the scientific parameter, even when the observed values themselves are held fixed.
:::::

:::::: math-passage
The resulting Gibbs sampler alternates among missing values, data-model parameters, and response-mechanism parameters. Their conditional kernels are

::: math-block
$$
[D_{\mathrm{mis}}|D_{\mathrm{obs}};\theta,\psi]\propto
L(D|\theta,\psi),
$$
:::

::: math-block
$$
[\theta|D;\psi]\propto
q(\theta,\psi)\prod_{i=1}^np_Y(Y_i|\theta),
$$
:::

::: math-block
$$
[\psi|D;\theta]\propto
q(\theta,\psi)\prod_{i=1}^n\pi(R_i|Y_i;\psi),
$$
:::

The first kernel is also the one used to simulate missing values in a nonignorable Monte Carlo E step. The other two hold the completed data fixed while updating their respective parameter blocks. At each step, only the variable being sampled is allowed to vary; the remaining arguments specify the conditioning information.
::::::

:::: math-passage
Nonignorability changes the missing-value update as well as the missingness-parameter update. A draw of $Z_{\mathrm{mis}}$ must be weighted by how compatible the completed record is with its observed response pattern. The complete-data kernel is

::: math-block
$$
q(\theta,\psi)\prod_i p(Y_i,Z_i;\theta)p(R_i\mid
Y_i,Z_i;\psi).
$$
:::

For the $\psi$ update, hold $\theta$ and completed data fixed but treat $\psi$ as the varying argument. A prior can regularize a poorly identified direction, but posterior concentration in such a direction may reflect prior information rather than information in the observations. Sensitivity to those assumptions is part of the analysis.
::::

## 5.8 Multiple imputation and pooling {#section-8}

Multiple imputation, introduced by Rubin in work beginning in the late 1970s and reviewed in Rubin (1996), turns predictive uncertainty into a collection of completed datasets. Missing entries are filled in $K$ times, each time with a new plausible draw. The same complete-data analysis is then applied to every dataset, producing $K$ estimates of the scientific target and $K$ estimates of their sampling variances. Pooling these results must account for the uncertainty within a completed-data analysis and the variation between plausible completions. That second component is what distinguishes the calculation from an ordinary analysis of a single filled-in dataset.

::::::: math-passage
Let $\widehat\theta^{(k)}$ be the estimate obtained from imputed dataset $k$, and let $\widehat V^{(k)}$ be its estimated sampling covariance, for example the inverse observed-information matrix. The pooled point estimate is the average of the completed-data estimates:

::: math-block
$$
\widehat\theta=\frac1K\sum_{k=1}^K\widehat\theta^{(k)}.
$$
:::

Two covariance components enter the uncertainty calculation. The average complete-data covariance measures within-imputation variation,

::: math-block
$$
\overline V=\frac1K\sum_{k=1}^K\widehat
V^{(k)},
$$
:::

while the sample covariance of the estimates measures between-imputation variation:

::: math-block
$$
\widehat
B=\frac1{K-1}\sum_{k=1}^K(\widehat\theta^{(k)}-\widehat\theta)(\widehat\theta^{(k)}-\widehat\theta)^{\mathrm
T}.
$$
:::

For $K\geq2$, Rubin's pooled covariance estimator combines them as

::: math-block
$$
\begin{equation}\tag{5.3}\label{eq:mi_var}
\widehat V^{MI}=\overline V+(1+K^{-1})\widehat B.\end{equation}
$$
:::

The factor $1+K^{-1}$ includes the Monte Carlo uncertainty from using a finite number of imputations. Each covariance is calculated on the same parameter scale; averaging standard errors or separate significance tests would give a different procedure.
:::::::

The predictive interpretation explains why repeated completion is needed. If $\theta$ were known, one could generate a complete dataset by first drawing $D_{\mathrm{obs}}$ from $[D_{\mathrm{obs}}\mid\theta]$ and then drawing $D_{\mathrm{mis}}$ from $[D_{\mathrm{mis}}\mid
D_{\mathrm{obs}};\theta]$. In an actual study, the first of these steps has already occurred. Multiple imputation supplies repeated draws for the second, preserving the observed values while representing uncertainty about the missing ones. The variation between completed-data estimates is measured by $\widehat
B$. Since $\theta$ is itself unknown, a valid imputation procedure must also address parameter uncertainty rather than simply choosing a fixed conditional distribution.

### Finite-imputation uncertainty and intervals

The within-imputation covariance estimates the uncertainty that would remain if a completed dataset had actually been observed. The between-imputation covariance measures the additional variation across plausible completions. Writing $T=\widehat
V^{MI}$ for their pooled value, the extra term $\widehat B/K$ accounts for Monte Carlo variation in a finite average. Increasing the number of imputations reduces this simulation component without removing the underlying missing-data uncertainty.

:::::: math-passage
For a scalar target with large complete-data degrees of freedom, define the relative increase in variance by

::: math-block
$$
r=\frac{(1+K^{-1})\widehat B}{\overline
V}.
$$
:::

An approximate reference degrees of freedom and the resulting confidence interval are

::: math-block
$$
\nu=(K-1)\left(1+\frac1r\right)^2,
$$
:::

::: math-block
$$
\widehat\theta\ \pm\ t_{\nu,1-\alpha/2}\sqrt
T.
$$
:::

When the complete-data sample is small, a further degrees-of-freedom adjustment is needed. For vector targets, the covariance matrices must be pooled on a common parameter scale. These distinctions explain why point estimates, standard errors, and confidence limits can require different numbers of imputations to become numerically stable.
::::::

## 5.9 Proper imputation and uncertainty {#section-9}

### The limitations of fixing the parameter

A simple plug-in procedure starts from an initial estimate $\widehat\theta^{init}$, perhaps a complete-case MLE, and draws each completion from $[D_{\mathrm{mis}}\mid
D_{\mathrm{obs}};\widehat\theta^{init}]$. This is an example of improper imputation. Its first difficulty is model accuracy: an inconsistent initial estimate gives the wrong conditional distribution even in large samples. A consistent initial estimate can support consistency of the resulting MI estimate under suitable assumptions, but it does not establish that the completed-data analysis is efficient or that the usual pooling variance is valid. The second difficulty, developed next, remains even when the initial estimate is well chosen.

:::: math-passage
Holding $\widehat\theta^{init}$ fixed across imputations omits its uncertainty. As a result, the usual pooling formula ($\ref{eq:mi_var}$) need not estimate the variance of the resulting MI estimator correctly. There are also settings in which an initially efficient estimator produces a less efficient MI estimator after this procedure (Tsiatis, 2006; Davidian, 2017). An approximate way to restore parameter variation is to draw a new value at the start of imputation $k$:

::: math-block
$$
N(\widehat\theta^{init},\widehat\Sigma^{init}),
$$
:::

Here $\widehat\Sigma^{init}$ estimates the covariance of $\widehat\theta^{init}$. Missing values are then drawn from $[D_{\mathrm{mis}}\mid
D_{\mathrm{obs}};\theta_k]$. This construction uses a large-sample normal approximation for parameter uncertainty, so its justification depends on the quality of both that approximation and the initial estimator.
::::

:::: math-passage
Drawing a fresh parameter value addresses the missing variability in the plug-in construction, but it cannot correct a distribution centered at an inconsistent estimate. A Bayesian predictive construction instead averages over the parameter posterior:

::: math-block
$$
p(D_{\mathrm{mis}}\mid D_{\mathrm{obs}})=\int
p(D_{\mathrm{mis}}\mid D_{\mathrm{obs}},\theta)p(\theta\mid
D_{\mathrm{obs}})\,d\theta.
$$
:::

This distribution propagates parameter uncertainty together with the conditional variability of the missing values. Under a correctly specified identifiable model, suitable prior support, and appropriate regularity, it provides a basis for proper imputation. Valid frequentist pooling also requires a suitable complete-data analysis and an appropriate relationship between the imputation and analysis models. Predictive simulation alone does not remove those requirements.
::::

Sampling from the predictive distribution can sometimes be done directly in two stages: draw $\theta$ from $[\theta\mid D_{\mathrm{obs}}]$, then draw $D_{\mathrm{mis}}$ from $[D_{\mathrm{mis}}\mid
D_{\mathrm{obs}};\theta]$. When the first distribution is difficult to sample, the data-augmentation construction provides an alternative. Starting from a prior $q(\theta)$, it alternates the missing-value and parameter conditionals, producing the predictive missing-value draws as one marginal of the joint chain. The algorithm in the next section makes explicit how those draws are retained for multiple imputation.

### What makes an imputation proper

:::: math-passage
The predictive distribution contains two distinct sources of uncertainty. For a scalar missing quantity $W$, the law of total variance gives

::: math-block
$$
\begin{aligned}
\operatorname{Var}(W\mid D_{\mathrm{obs}})
&=E_{\theta\mid D_{\mathrm{obs}}}\{\operatorname{Var}(W\mid
D_{\mathrm{obs}},\theta)\}\\
&\quad+\operatorname{Var}_{\theta\mid D_{\mathrm{obs}}}\{E(W\mid
D_{\mathrm{obs}},\theta)\}.
\end{aligned}
$$
:::

The first term averages the residual uncertainty at a fixed parameter; the second measures how the conditional prediction changes across the posterior. Fixing $\theta$ at a fitted value generally loses the second component, even when random residual errors are still added. Deterministic mean imputation can lose both components and alter associations.
::::

Bayesian predictive simulation is a coherent way to propagate uncertainty under a specified model. It does not guarantee frequentist consistency for every prior, every misspecified imputation model, or every downstream analysis. A prior excluding the truth, an incompatible imputation model, or omission of an interaction needed by the analysis can invalidate inference. Proper-imputation arguments require suitable models, regularity, and an appropriate relationship between imputation and analysis.

## 5.10 Posterior predictive imputation {#section-10}

:::::: math-passage
Initialize the parameter and alternate two conditional draws. At iteration $j+1$, first draw a new set of missing values,

::: math-block
$$
D_{\mathrm{mis}}^{(j+1)}\sim
p(D_{\mathrm{mis}}\mid D_{\mathrm{obs}},\theta^{(j)})\propto
L(D\mid\theta^{(j)}),
$$
:::

and then update the parameter using the newly completed data:

::: math-block
$$
\theta^{(j+1)}\sim
p(\theta\mid D_{\mathrm{obs}},D_{\mathrm{mis}}^{(j+1)})\propto
L(D\mid\theta)q(\theta).
$$
:::

These are the same data-augmentation updates used for a fully Bayesian analysis. Their interpretation depends on which output is retained. After a burn-in of $M$ iterations and suitable convergence checks, the parameter draws $\theta^{(M+1)},\theta^{(M+2)},\ldots$ describe the parameter posterior. For multiple imputation, retain completed datasets instead:

::: math-block
$$
D^{(j)}=(D_{\mathrm{obs}},D_{\mathrm{mis}}^{(j)}),\qquad
j=M+1,M+2,\ldots.
$$
:::

The observed entries remain fixed in every dataset. The completed-data analyses and their pooling then supply the inferential output, with Monte Carlo dependence between retained draws taken into account when choosing and assessing the imputation run.
::::::

## 5.11 Multivariate normal imputation {#section-11}

:::: math-passage
So far, the same probability model has supplied both the imputations and the completed-data analysis. To distinguish their roles, write $L(D\mid\theta)$ for the analyst's likelihood and $L^{imp}(D\mid\theta^*)$ for the imputer's likelihood. The latter determines both $[\theta^*\mid D]$ and $[D_{\mathrm{mis}}\mid
D_{\mathrm{obs}};\theta^*]$. Using the same model gives

::: math-block
$$
L^{imp}(D\mid\theta)=L(D\mid\theta).
$$
:::

Meng's (1994) notion of congeniality concerns whether imputation and analysis can be justified together under a compatible inferential construction. Identical likelihoods provide one natural way to arrange this compatibility, but they are not its general definition. What matters is whether the imputation preserves the distributional features and uncertainty relevant to the analysis.
::::

A complex analysis model may be inconvenient for generating imputations. This motivates using a different imputation model, with $L^{imp}(D\mid\theta)\ne
L(D\mid\theta)$, while examining whether the resulting analyses remain compatible. For a vector of continuous variables, a multivariate normal model is a convenient example because its conditional distributions are explicit. Schafer (1997) discusses simulation settings in which this approximation performs reasonably even for some categorical variables when missingness is limited. Such results motivate investigation in a particular problem; they do not justify ignoring a variable's support or the associations required by the intended analysis.

Congeniality concerns whether the imputation and analysis procedures can be justified together under a suitable model. Using precisely the same likelihood is a useful sufficient construction in many examples; different model descriptions are not automatically uncongenial. Include variables, transformations, interactions, and outcome associations needed for the intended analysis. A multivariate normal approximation is convenient for continuous variables, but treating binary, bounded, or strongly skewed quantities as normal requires justification rather than a blanket claim of robustness.

### A regression factorization for monotone patterns

:::: math-passage
Multivariate normal imputation is especially simple under a monotone missingness pattern. Let $Y=(Y_1,\ldots,Y_p)$ and factor its distribution into successive regressions on the preceding variables. For $j=1,\ldots,p$, take

::: math-block
$$
\begin{equation}\tag{5.4}\label{eq:mono_mvn}
Y_j\mid (Y_1,\cdots,
Y_{j-1})=N\left(\beta_{j0}+\beta_{j1}Y_1+\cdots+\beta_{j,j-1}Y_{j-1},
\tau_j^{-1}\right).\end{equation}
$$
:::

The coefficient $\beta_{j0}$ is an intercept, and $\tau_j$ is the conditional precision. This representation connects a joint normal distribution to a sequence of ordinary normal regressions. The original course discussed software examples including SAS PROC MI, NORM, AMELIA, and MICE; the statistical distinction is between a joint-model construction such as this one and separately specified conditional models, which are considered below.
::::

Write $\beta_j=(\beta_{j0},\ldots,\beta_{j,j-1})^{\mathrm
T}$. A monotone pattern means that once $Y_j$ is missing, all later components $Y_{j+1},\ldots,Y_p$ are missing as well. Consequently, every subject with an observed response in regression $j$ also has all of that regression's predictors observed. Subjects lost earlier contribute no information about $(\beta_j,\tau_j)$: integrating an unobserved normal response gives one, removing its regression parameters from the likelihood. The observed-data likelihood therefore factors into complete regression contributions at successive stages. With a suitable factorized prior, the resulting parameter posteriors have closed forms, and direct sequential predictive draws replace a Gibbs iteration.

::::: math-passage
If we choose the improper prior $q(\beta_1,\tau_1,\cdots, \beta_p,\tau_p)\propto
\prod_{j=1}^p\tau_j^{-1}$, then, one can show

::: math-block
$$
\tau_j\mid D_{\mathrm{obs}} \sim
\widehat\tau_j\chi^2_{n_j-j}/n_j,
$$
:::

::: math-block
$$
\beta_j\mid (D_{\mathrm{obs}}; \tau_j) \sim
N\left(\widehat\beta_j,
\tau_j^{-1}(S_j^{\mathrm{T}}S_j)^{-1}\right),
$$
:::

where $n_j$ is the number of complete cases for the $j$th regression model, $\widehat\beta_j$ and $\widehat\tau_j$ are the CC (thus observed-data) MLEs, and $S_j$ is the design matrix for the regression model ($\ref{eq:mono_mvn}$). Moreover, because of the factorization in the likelihood, the $(\beta_j, \tau_j)$ are independent given $D_{\mathrm{obs}}$. So, to draw $(\beta_j, \tau_j)$ from the posterior, one first draws $\tau_j$ from $[\tau_j\mid D_{\mathrm{obs}}]$ and then $\beta_j$ from $[\beta_j\mid D_{\mathrm{obs}};
\tau_j]$.
:::::

:::: math-passage
Now, to complete the imputation, it remains to draw $D_{\mathrm{mis}}$ from $[D_{\mathrm{mis}}\mid
D_{\mathrm{obs}};\theta^*]$, where $\theta^*=(\beta_1,\tau_1,\cdots,
\beta_p,\tau_p)$ are freshly drawn from their posterior. Because of the special structure, this can be done progressively in $j=1,\cdots, p$. For each subject, assume the missing values in $Y_{1},\cdots,
Y_{j-1}$ have been imputed. Then, because of the imputation model ($\ref{eq:mono_mvn}$), $Y_j$ can be imputed by

::: math-block
$$
Y_j=\beta_{j0}+\beta_{j1}Y_1+\cdots+\beta_{j,j-1}Y_{j-1}+\tau_j^{-1/2}\epsilon_j,
$$
:::

where $\epsilon_j$ is simulated from $N(0,1)$.
::::

### A complete monotone-regression draw

::::: math-passage
Under a monotone pattern, order variables so that subjects observed on variable $j$ also have its predecessors. Let $n_j$ be their number, let $X_j$ be the $n_j\times j$ design including an intercept, and let $y_j$ be their responses. For the factorized regression prior $q(\beta_j,\tau_j)\propto\tau_j^{-1}$, define

::: math-block
$$
\widehat\beta_j=(X_j^TX_j)^{-1}X_j^Ty_j,\qquad
\operatorname{RSS}_j=\|y_j-X_j\widehat\beta_j\|^2.
$$
:::

Integration over the $j$ regression coefficients gives

::: math-block
$$
\tau_j\mid
D_{\mathrm{obs}}\sim\operatorname{Gamma}\left(\frac{n_j-j}{2},\frac{\operatorname{RSS}_j}{2}\right),\qquad
\beta_j\mid\tau_j,D_{\mathrm{obs}}\sim
N_j\{\widehat\beta_j,\tau_j^{-1}(X_j^TX_j)^{-1}\}.
$$
:::

Equivalently, $\tau_j=\chi^2_{n_j-j}/\operatorname{RSS}_j$. If $\widehat\tau_j=n_j/\operatorname{RSS}_j$, the multiplier is $\widehat\tau_j/n_j$, not $\widehat\tau_j^2/n_j$. These draws require $n_j>j$, full column rank, and positive residual sum of squares. Use the same sampled coefficient and precision within one imputed dataset, but independent residual draws for different subjects. Progress from earlier to later variables, using previously imputed predecessors where necessary. Draw new parameter values for the next imputed dataset.
:::::

An arbitrary missingness pattern generally lacks this convenient factorization. Under a joint multivariate normal model, data augmentation can instead alternate draws of the missing values, the mean vector $\mu$, and the covariance matrix $\Sigma$. The conjugate construction supplies multivariate conditional distributions, allowing entire blocks to be updated rather than every scalar component separately. This is the same computational advantage seen in the earlier normal-regression example. Yuan (2011) describes implementations in SAS PROC MI for both monotone and nonmonotone patterns; the underlying distinction is whether the observed-data factorization permits direct successive regression draws or requires a joint iterative calculation.

## 5.12 Chained equations {#section-12}

Chained equations offer a different construction for mixed continuous and categorical data. Instead of approximating the entire vector by a joint normal distribution, one specifies a regression model for each incomplete variable conditional on the others. This is called fully conditional specification (FCS), or multivariate imputation by chained equations (MICE) (van Buuren, 2007; van Buuren and Groothuis-Oudshoorn, 2011). Its flexibility comes from matching each regression family to its response variable. The accompanying difficulty is that a collection of sensible regressions does not automatically define a compatible joint model.

Let the full observation be $Y=(Y_1,\ldots,Y_p)$. Begin by filling missing entries with initial values, for example draws from their observed marginal values, while recording which entries were originally missing. For the first incomplete variable, say $Y_1$, fit a regression on $(Y_2,\ldots,Y_p)$ using subjects whose $Y_1$ is observed. Then draw replacements for the missing $Y_1$ values from the fitted predictive construction. A continuous response may use a normal regression, a binary response a logistic regression, and a count response a Poisson regression. The other variables enter at their currently observed or imputed values, so later updates will change some of the conditioning information used in this first step.

Continue with $Y_2$ and then each remaining incomplete variable, always fitting the regression to its observed responses and replacing only the entries originally missing. One pass through the variables is a cycle. Subsequent cycles use the latest completed values as predictors, which explains why the equations are described as chained. The lecture illustration used 10--20 cycles to seek stabilization, but the necessary run length depends on the problem. A completed run supplies one imputed dataset; repeating the construction with appropriate new draws supplies $K$ datasets for pooling. Stability of the imputed summaries and adequacy of the conditional models must be assessed alongside the mechanics of cycling.

:::: math-passage
Now we look at what precisely is done in the imputation for the missing $Y_j$. If $Y_j$ is continuous, then the regression model can be chosen to be a linear regression, and the imputation is similar to what's been described in the imputation step for the monotone missing case with multivariate normal model. If $Y_j$ is categorical, then typically a GLM is used to model it, say, logistic regression. Denote the MLE of the regression parameter as $\widehat\beta_j$. Then, drawing from the "posterior predictive distribution" is similar to what we have mentioned as the "sort-of" proper imputation. Specifically, we first draw $\beta_j$ from

::: math-block
$$
N(\widehat\beta_j,\widehat\Sigma_j),
$$
:::

where $\widehat\Sigma_j$ is an estimate for the variance of $\widehat\beta_j$.
::::

:::: math-passage
Then, $Y_j$ is drawn from

::: math-block
$$
\mbox{Bin}\left(1,\frac{\exp(\beta_j^{\mathrm{T}}Z_j)}{1+\exp(\beta_j^{\mathrm{T}}Z_j)}\right),
$$
:::

where $Z_j=(Y_1,\cdots, Y_{j-1}, Y_{j+1},
Y_p)^{\mathrm{T}}$. The MICE algorithm is implemented in, for example, SAS PROC MI using the FCS option and the R package MICE. Some caveats: The MICE/FCS approach is predicated on specification of full conditional models; that is, in our example for each $j$, models for $Y_j$ as a function of all other variables.
::::

Compatibility has a precise meaning: there must exist a single joint distribution with all the proposed full conditionals. Independently chosen regressions for mixed variable types need not satisfy this condition. Thus a chained-equation procedure cannot automatically inherit the target-distribution argument of a Gibbs sampler whose conditionals are derived from a known joint model. This distinction does not by itself establish failure of the algorithm. It means that convergence, the distribution reached by cycling, and validity for the intended analysis are separate questions, each of which needs justification.

### Compatibility, cycling, and analysis

For a continuous variable one may draw regression parameters and then draw normal residuals; for a binary variable one may draw logistic-regression parameters and then Bernoulli responses. Every update conditions on the most recent values of the other variables and replaces only entries that were originally missing. Retain the original missingness indicators throughout the algorithm.

Some sets of full conditional models are compatible with a joint distribution, while others are not. Incompatibility does not prove that an iterative procedure fails to settle down, but a stationary distribution of the algorithm need not have all the stipulated full conditionals. Update order, model specification, and the analysis target can therefore matter. Iteration counts alone do not establish stability; inspect convergence of imputed summaries and fitted parameters rather than treating a fixed number of cycles as certification.

## 5.13 Nonignorable imputation and sensitivity analysis {#section-13}

:::: math-passage
For nonignorable missingness, the imputation model must incorporate additional identifying assumptions. A selection-model construction specifies $\pi(R\mid Y;\psi)$ and combines it with the imputer's full-data likelihood:

::: math-block
$$
L^{imp}(D|\theta^*)\prod_{i=1}^n\pi(R_i|Y_i;\psi),
$$
:::

A joint prior for $(\theta^*,\psi)$ then supplies the posterior predictive distribution used for completion. The analyst may apply the same complete-data calculation to datasets generated under several alternative assumptions, which makes MI useful for sensitivity analysis. The resulting scientific conclusions still depend on those assumptions, however; separating imputation from analysis does not make nonignorability irrelevant to the person interpreting the results.
::::

The number of imputations controls the precision of the simulation, not the validity of the identifying model. Early MI literature often considered $K=3$--$5$, whereas later guidance discussed $K=30$--$50$ or a rule based on the percentage of missing information. These historical numbers reflect different precision requirements and should not be treated as universal thresholds. In particular, a point estimate can stabilize before its standard error or confidence limits do. The calculation below relates the Monte Carlo error directly to the between-imputation variation and provides a clearer basis for deciding whether additional imputations are needed.

### Sensitivity parameters and Monte Carlo precision

A useful pattern-mixture sensitivity analysis starts with a MAR imputation distribution and shifts missing outcomes by a specified $\delta$: $Y_{\mathrm{mis}}^{(\delta)}=Y_{\mathrm{mis}}^{(0)}+\delta$ for a continuous response. The parameter $\delta$ states how the missing outcomes differ after conditioning on observed predictors. Repeat imputation and analysis across scientifically interpretable values. For binary responses, a shift on the log-odds scale respects their support. These are explicitly different identifying assumptions; the observed data generally cannot estimate the shift without additional information.

The number of imputations should be chosen with simulation precision in mind. With approximately independent proper imputations, the Monte Carlo standard error of $\overline\theta$ is approximately $\sqrt{\widehat B/K}$ for a scalar estimand. An often-used large-sample relative-efficiency approximation is $(1+\gamma/K)^{-1}$, where $\gamma$ is the fraction of missing information. Increasing $K$ reduces simulation error; it does not fix an incorrect imputation model. Choose enough imputations that Monte Carlo error is small relative to the inferential uncertainty and check the stability of interval endpoints and other reported quantities.

## 5.14 References {#section-14}

Davidian, m. (2017) Lecture notes: Statistical methods for analysis with missing data. [http://www4.stat.ncsu.edu/\~davidian/st790/notes.html](http://www4.stat.ncsu.edu/~davidian/st790/notes.html){.uri rel="noopener noreferrer" target="_blank"}.

Gilks, W. R. & Wild, P. (1992). Adaptive rejection sampling for Gibbs sampling. Applied Statistics, 337-348.

Meng, X. L. (1994). Multiple-imputation inferences with uncongenial sources of input. Statistical Science, 538-558.

Rubin, D. B. (1996). Multiple imputation after 18$+$ years. Journal of the American statistical Association, 91, 473-489.

Schafer, J. L. (1997). Analysis of incomplete multivariate data. CRC press.

Tsiatis, A. (2006). Semiparametric theory and missing data. New York: Springer.

van Buuren, S. (2007). Multiple imputation of discrete and continuous data by fully conditional specification. Statistical methods in medical research, 16, 219-242.

van Buuren, S. & Groothuis-Oudshoorn, K. (2011). MICE: Multivariate imputation by chained equations in R. Journal of statistical software, 45.

Yuan, Y. (2011). Multiple imputation using SAS software. Journal of Statistical Software, 45, 1-25.
