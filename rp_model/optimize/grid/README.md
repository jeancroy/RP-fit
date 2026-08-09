# RP fixed-grid solver

The ingredient and skill rates are restricted to 0.1% ticks, so the complete search space contains 351 ingredient
rates and 170 skill rates: 59,670 pairs. This solver evaluates that entire grid while sharing the ingredient and skill
component calculations across pairs.

For each observation it builds one 351-by-170 predicted-RP matrix, applies the production truncation and rounding
rules, and accumulates integer squared error. It returns every rate pair at the global minimum, so one result means the
fit is unique and more than one result means the available observations are ambiguous. The selected result is replayed
through `compute_rp()` before it is returned.

This remains an exhaustive solution over the declared ticks while reusing component calculations across the complete
grid.
