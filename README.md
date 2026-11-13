# Polymarket "Next CEO of NVIDIA?" 4-hour-ahead quantile forecaster

Entry for make.fun's `polymarket-nvidia-ceo-forecast-01`.

`main.py` reads one case (`/input/<id>.in`: every market of the event with its Yes price for each
minute of the 72 hours up to a checkpoint) and writes `/output/<id>.out`: for every market and each
of the next 240 minutes, the 5/25/50/75/95 quantiles of that minute's price.

The model: a price is a bounded probability, so the state space is the logit of the price. For each
market the program measures how far the logit moves over a spread of horizons in its own 72 hours of
minutes, reads the empirical quantiles off that distribution, maps them back through the sigmoid, and
lets the band widen with the horizon the way the market's own history does. Quiet markets get narrow
bands, markets that jump get fat ones — unlike a fixed spread applied to every market.

Runs offline, standard library only, deterministic, well inside the case's 8-second budget.
