From Applied Hydrology (1988) equation 9.3.14, we have

$c_{k} = \frac{dQ}{dA}$.

By assuming Manning's Equation, we get

$\frac{dQ}{dA} = \frac{d}{dA} \frac{1}{n} s^{1/2} A R_{h}^{2/3}$.

Pulling out terms not dependant on A yields

$\frac{dQ}{dA} = \frac{s^{1/2}}{n} * \frac{d}{dA} A R_{h}^{2/3}$.

We can then use the derivative product rule to arrive at

$\frac{dQ}{dA} = \frac{s^{1/2}}{n} * [A\frac{d R_{h}^{2/3}}{dA} + R_{h}^{2/3} ]$,

which can be rewritten as 

$\frac{dQ}{dA} = \frac{s^{1/2}}{n} * [\frac{A}{p^{2/3}} \frac{d A^{2/3}}{dA} + R_{h}^{2/3}]$

and reduced using the chain rule to

$\frac{dQ}{dA} = \frac{s^{1/2}}{n} * [\frac{A}{p^{2/3}} \frac{2 A^{-1/3}}{3} + R_{h}^{2/3}]$

$\frac{dQ}{dA} = \frac{s^{1/2}}{n} * [\frac{2}{3} R_{h}^{2/3} + R_{h}^{2/3}]$

$\frac{dQ}{dA} = \frac{5}{3} \frac{s^{1/2}R_{h}^{2/3}}{n}$.

This can be further simplified to

$\frac{dQ}{dA} = \frac{5}{3} \frac{Q}{A}$

or

$\frac{dQ}{dA} = \frac{5}{3} V$.


To summarize, under the assumption of Manning's flow, kinematic wave celerity equals $\frac{5}{3}$ of the flow velocity.


$c_{k} = \frac{5}{3} V$.
