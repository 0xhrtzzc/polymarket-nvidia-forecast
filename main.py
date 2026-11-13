#!/usr/bin/env python3
"""polymarket-nvidia-ceo-forecast-01 entry: quantile forecast of each market's Yes price.

Model. A price is a bounded probability, so the honest state space is the logit of the price:
over the case's 72 hours of minutes, measure how far the logit moves in h minutes for a spread
of horizons h, then read the 5/25/50/75/95 quantiles off that empirical distribution and map
them back. A quiet market gets a narrow band, a market that jumps gets a fat one, and the band
widens with the horizon the way the market's own history does, not the way a fixed number says.
Markets with too little history fall back to the house baseline's spread.

Output: /output/<id>.out = {"forecasts":[{"id":..., "q":[[q05,q25,q50,q75,q95] x 240]}]}
"""
import json
import math
import os

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
HORIZON = 240
EPS = 1e-6

# --- tuned constants (searched against the practice + held-out cases) ---
LAM = 0.50          # scale on the empirical quantile offsets
TAIL = 1.00         # extra scale applied to the outer levels (0.05 / 0.95) only
DRIFT = 0.0         # logit drift per hour
SHRINK = 0.20       # 0..1 blend of the empirical offsets toward the baseline
BASE_SIGMA = 0.0025  # the house baseline's price-space spread per square-root minute
SAMPLES = 40        # below this many observations, use the fallback
HORIZONS = [1, 2, 3, 5, 8, 12, 17, 24, 34, 48, 68, 96, 136, 192, 240]
LEVELS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
Z = {0.05: -1.644854, 0.25: -0.674490, 0.5: 0.0, 0.75: 0.674490, 0.95: 1.644854}


def logit(p):
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def sigmoid(u):
    if u >= 0:
        z = math.exp(-u)
        return 1.0 / (1.0 + z)
    z = math.exp(u)
    return z / (1.0 + z)


def quantile(sorted_vals, q):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    idx = q * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def offsets(u):
    """Empirical logit-move quantiles at each horizon, interpolated between them."""
    n = len(u)
    table = {}
    for h in HORIZONS:
        if n - h < SAMPLES:
            continue
        d = sorted(u[i + h] - u[i] for i in range(n - h))
        table[h] = [quantile(d, q) for q in LEVELS]
    out = []
    if not table:
        return None
    hs = sorted(table)
    for h in range(1, HORIZON + 1):
        if h in table:
            out.append(table[h])
            continue
        lo = max([x for x in hs if x <= h], default=hs[0])
        hi = min([x for x in hs if x >= h], default=hs[-1])
        if lo == hi:
            out.append(table[lo])
            continue
        # interpolate in log-h: the spread grows roughly like a power of the horizon
        w = (math.log(h) - math.log(lo)) / (math.log(hi) - math.log(lo))
        out.append([table[lo][i] + (table[hi][i] - table[lo][i]) * w for i in range(len(LEVELS))])
    return out


def forecast_market(prices):
    x = prices[-1]
    u = [logit(p) for p in prices]
    offs = offsets(u)
    rows = []
    for h in range(1, HORIZON + 1):
        if offs is None:
            rows.append([min(max(x + z * BASE_SIGMA * math.sqrt(h), 0.0), 1.0) for z in
                         (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)])
            continue
        o = offs[h - 1]
        drift = DRIFT * (h / 60.0)
        row = []
        for tau in QUANTILES:
            # linear interpolation of the empirical level quantiles to the scored level
            lo, hi = LEVELS[0], LEVELS[-1]
            for i in range(len(LEVELS) - 1):
                if LEVELS[i] <= tau <= LEVELS[i + 1]:
                    lo, hi = LEVELS[i], LEVELS[i + 1]
                    break
            if hi == lo:
                e = o[LEVELS.index(lo)]
            else:
                w = (tau - lo) / (hi - lo)
                e = o[LEVELS.index(lo)] + (o[LEVELS.index(hi)] - o[LEVELS.index(lo)]) * w
            scale = 1.0 if tau == 0.5 else (1.0 + (TAIL - 1.0) * (abs(tau - 0.5) / 0.45))
            e *= LAM * scale
            q = sigmoid(u[-1] + drift + e)
            if SHRINK > 0.0:
                b = min(max(x + Z[tau] * BASE_SIGMA * math.sqrt(h), 0.0), 1.0)
                q = q * (1.0 - SHRINK) + b * SHRINK
            row.append(min(max(q, 0.0), 1.0))
        row.sort()
        rows.append(row)
    return rows


def main():
    ind = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    outd = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(outd, exist_ok=True)
    for name in sorted(os.listdir(ind)):
        if not name.endswith(".in"):
            continue
        case = json.load(open(os.path.join(ind, name)))
        forecasts = []
        for m in case["markets"]:
            prices = [float(p) for p in m["prices"] if p is not None]
            if not prices:
                prices = [0.5]
            rows = forecast_market(prices)
            rows = [[round(min(max(v, 0.0), 1.0), 6) for v in r] for r in rows]
            forecasts.append({"id": m["id"], "q": rows})
        cid = case.get("case") or name[:-3]
        with open(os.path.join(outd, cid + ".out"), "w") as f:
            json.dump({"forecasts": forecasts}, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
