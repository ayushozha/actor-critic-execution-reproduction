# Frozen controlled execution experiment

The experiment runs in the present day. Scientific ingredients are restricted to public versions available before 2018-09-20. Python 3.6.6, NumPy 1.15.1, and PyTorch 0.4.1 are isolated under `runtime/`. The protocol and configuration are hash-bound before any training or held-out evaluation.

## Question and scope

Does training deterministic deep actor-critic execution policies across uncertain temporary-impact parameters improve held-out risk-adjusted simulated execution cost relative to nominal training, a TWAP schedule, and a nominal discrete Almgren-Chriss schedule? DDPG and TD3 are compared under the same small network and interaction budget. The study is a controlled simulation, not historical market calibration, a reproduction of another paper's numerical results, or evidence of live profitability. The experimental design adapts pre-cutoff execution and continuous-control methods. New numerical outputs have their actual execution dates.

## Market and objective

Sell initial inventory X=1 over N=20 decisions, dt=1/N. At decision k, q_k>=0 is the trade, x_k=x_{k-1}-q_k, and the temporary price concession is h(v_k)=eta*v_k^alpha/L_k, where v_k=q_k/dt. The impact cost is q_k*h(v_k). There is no permanent impact, spread, limit-order queue, or fill uncertainty; every allowed trade fills. This restricted monotone sell model cannot trade round trips. It tests specification sensitivity rather than establishing no-arbitrage properties of a general transient-impact model.

Liquidity is observable and independent of the agent: log L_k = log ell + u_k, with stationary Gaussian AR(1) u_0~N(0,s_epsilon^2/(1-rho^2)), u_{k+1}=rho*u_k+s_epsilon*epsilon_k, rho=.8 and s_epsilon=.1. The episode's true eta and alpha are hidden. A randomized regime samples eta log-uniformly over [.05,.2], alpha uniformly over [.8,1.2], and ell uniformly over [.75,1.25], independently at each episode. Nominal training fixes eta=.1, alpha=1, ell=1. These ranges are design choices, not empirically estimated values.

Training reward is -10*[eta*q_k*(q_k/dt)^alpha/L_k + lambda*sigma^2*dt*x_k^2], with lambda=25, sigma=.2, and discount=1. It optimizes an explicit risk proxy and omits zero-mean price shocks from the reward to reduce unnecessary simulation variance. Evaluation independently records impact cost, the holding-risk penalty, realized price slippage -sigma*sqrt(dt)*sum_k(x_k*z_k), implementation shortfall (impact plus slippage), and the deterministic risk-adjusted proxy (impact plus holding-risk penalty). The same price and liquidity innovation arrays are used for every method. Because trades depend on stochastic liquidity, the unconditional variance of total shortfall additionally includes impact variability; lambda*sigma^2*dt*sum x_k^2 is the chosen price-risk component, not an assertion that it equals total unconditional variance.

## State, action, and controls

The actor sees only [remaining_steps/N, remaining_inventory/X, log L_k]. It sees no true eta or alpha, mean-liquidity parameter, future liquidity or price innovation, critic target, or evaluation scenario identifier. A tanh output a in [-1,1] trades q=x*min[1, 3(a+1)/(2n)] at n remaining decisions. At n=1, the trade is x regardless of a, enforcing completion. TWAP trades x/n. The nominal discrete quadratic AC control trades a remaining-inventory fraction c_n/(b+c_n), where b=eta_0/dt, A_1=b, c_n=lambda*sigma^2*dt+A_{n-1}, and A_n=b*c_n/(b+c_n). The analytic control is exact only for deterministic L=1 and alpha=1 with the stated after-trade risk discretization. It retains the nominal fractions in all held-out scenarios. A preflight asserts every nominal AC fraction satisfies the common action bound min(1,3/n); no nonlinear or hindsight oracle is used.

## Learning and evaluation plan

DDPG uses one critic, replay, an actor, target networks, and Adam. TD3 adds twin critics with clipped-minimum targets, clipped Gaussian target-action noise, and actor/target updates every second critic update, following the June 7, 2018 version. Both use two 64-unit ReLU layers, 64-transition batches, learning rates .001, tau=.005, and Gaussian exploration sd=.15. Networks are smaller and the 20,000-transition budget shorter than the original physics benchmarks; this is stated as a controlled adaptation, not an exact replication. The final checkpoint is evaluated without selecting an intermediate checkpoint. Five fixed seeds cross two algorithms and two training regimes, for 20 training runs and 400,000 environment transitions. Each run has 1,000 uniform-action warm-up transitions and then one update per transition. Training trajectories and checkpoint intervals are saved. There is no early stopping, evaluation-based tuning, or hidden selection.

Eight prespecified scenarios include the nominal setting, doubled/halved eta, alpha=.75/1.25, halved liquidity, a joint eta=.25/alpha=1.3/ell=.6 shift outside the training ranges, and ell=1.5. Each has 1,024 common held-out episodes; deterministic policies are evaluated without exploration. Primary comparison: TD3 randomized versus TD3 nominal, equally weighted over the seven nonnominal scenarios. Secondary comparisons: DDPG randomized versus nominal, TD3 versus DDPG, and each group versus TWAP and fixed nominal AC. Confidence intervals use five paired training-seed means with Student t(4)=2.7764451052. Evaluation episodes estimate each seed's conditional mean and are not treated as 5,120 independently trained policies. Report all scenario outcomes and negative results. No multiple-comparison significance claim is made from the exploratory scenario table.

## Provenance and verification

Before running, save config and code hashes, true UTC freeze time, runtime identity, and named source versions. Save every raw episode result, common shock arrays, training episode log, checkpoints at 5,000-transition intervals, actual update counts, runtime/elapsed records, and exceptions. Preflight verifies finite outputs and gradient updates, inventory conservation, liquidation completion, positive impact cost, no hidden-state inputs, reproducible shock arrays, exact AC objective against its value recurrence and direct positive-definite quadratic solve, and the common action bounds. The present-day administrative download and manuscript rendering software do not participate in the scientific experiment. No market dataset or stock is used.

## Historical anchors inspected

- Lillicrap et al., `1509.02971v5`, 2016-02-29: Algorithm 1 and actor/critic replay and target-network sections.
- Fujimoto et al., `1802.09477v2`, 2018-06-07: Algorithm 1, target smoothing, delayed update, and hyperparameter section.
- Hendricks and Wilcox, `1403.2229v1`, 2014-03-10: discrete liquidation, inventory/time state, temporary impact, price-risk variance, and forced final completion.
- Curato, Gatheral and Lillo, `1412.4839v1`, 2014-12-16: nonlinear impact assumptions and their model-dependent limitations. This protocol uses temporary rather than their transient-impact model.
- Parameter randomization was public before the cutoff; the manuscript's source registry separately verifies the exact Tobin (2017) and Peng (2018) sources. The current parameter ranges are explicit experiment choices.
