# Bayesian evidence

```bash
python client.py evidence RUN_DIR                       # the latest network, 2048 live points
python client.py evidence RUN_DIR -i 7 --n-live 1024
python client.py evidence RUN_DIR --prior omega_b=0.019:0.025,omega_cdm=0.095:0.145
python client.py evidence RUN_DIR --prior-file ranges.yaml    # {omega_b: [0.019, 0.025], ...}
python client.py evidence RUN_DIR --region none               # the bare network, also where the run has a region
python client.py evidence RUN_DIR --region 1                  # the run's region with its edge at 1 reach
```

The evidence of a CLiENT run is computed from its trained network with the
[BEST](https://github.com/AndreasNygaard/best-inference) nested sampler. Install BEST,
set `BEST_PATH` to its checkout, or pass `--best-path`. Only the network is evaluated:
no Boltzmann code runs, and no MontePython or Cobaya model is built. A 27D Planck run
takes minutes, where nested sampling of the true likelihood costs CPU-years. It works
for runs made with the `cobaya` and the `montepython` wrapper.

## What is reported

BEST samples a uniform prior on a box B, the *sampling box*, and returns

    ln Z_box = ln [ (1/V_B) ∫_B exp(t) dx ],

where t is what the network learned. That number depends on the box: on the 27D Gaussian
below it changes by 10 between two boxes that hold the same posterior mass. The command
therefore also reports **ln Z in the run's own prior convention**, the number the
backend's own nested sampler would give:

| backend | what the network learned (t) | ln Z |
|---|---|---|
| Cobaya | Cobaya's log-posterior: ln L plus its normalised priors (uniform, Gaussian, ...) | ln Z_box + ln V_B |
| MontePython | ln L as `compute_lkl` returns it: the likelihoods plus MontePython's Gaussian nuisance priors, unnormalised as MontePython adds them | ln Z_box + ln V_B − Σᵢ ln(rangeᵢ) |

MontePython's priors are uniform over each varying parameter's `[min, max]` in the
`.param` file. Where a side is `None`, as is usual for cosmological parameters, **the
run's domain is used for it**, the region in `metadata.json` that the network was trained
and the posterior sampled in. The output lists every parameter's range and where it came
from (`.param`, `run domain`, `.param (upper: run domain)`, `user`).

This ln Z does not depend on the box, provided the box holds the posterior mass: on
the 27D Gaussian, the domain and a ±6σ box give the same value to 0.002. A face of the
box that lies inside the prior and that the posterior reaches is reported: more than
10⁻³ of the posterior within 2% of the box width of the face is flagged. The evidence
then misses the mass beyond it. A face on the prior's own edge is never a problem: a
Cobaya bound, the run domain standing in for an unbounded MontePython side, or a user
range.

## Your own prior ranges

`--prior a=lo:hi,b=lo:hi` or `--prior-file ranges.yaml` (a mapping `name: [lo, hi]`;
the command line wins where both name a parameter) replace the uniform prior of the
named parameters:
- **MontePython:** the range replaces the `.param` range or the domain.
- **Cobaya:** the range replaces a uniform prior, and the network's density is
  renormalised accordingly. A parameter with a non-uniform Cobaya prior (Gaussian, ...)
  cannot be given a range; change it in the Cobaya input instead.

A range narrower than the box shrinks the box to it. A wider one is fine as long as the
box holds the posterior mass; the face check above tells you if it does not. Parameter
names are those of `metadata.json`, e.g. `100*theta_s=1.03:1.05`.

## The sampling box

By default, the box is the run's domain. `--n-sigma K` uses each parameter's proposal
centre ± K proposal σ (from `metadata.json`) within the domain; `--keep-bounds a,b`
exempts parameters. `--box OLD.json` reuses the `lower`/`upper` recorded in an earlier
output of this command for the same parameters, inside the same domain. A smaller box saves
a few iterations. It must still hold the posterior, and ln Z tells you whether it does,
where ln Z_box cannot.

## The region of the boosts

A network trained on boost points with `acquisition.boost.region` is to be used inside
the region its boosts reach ([boost_background_region.md](boost_background_region.md)):
outside, it can rise far above the best true value. The evidence integrates such a
network inside its region, as profiles (`region: run`) and derived chains do. The region
is `RUN_DIR/boost/region_it_M.npz` with the largest M at or below the network's
iteration; outside it the nested sampler sees the region's wall, not the network.

| option | what is integrated |
|---|---|
| (not given) | the network inside its region when the run has one for it, over the whole box otherwise |
| `--region` or `--region run` | the network inside its region; the command stops when the run has none |
| `--region E` (a number) | the network inside the same region with its edge at E times the reach of the boost points, as the profile setting `region: E`; the output's name gets `_edgeE`, beside the default one |
| `--region none` | the bare network over the whole box |

The first line of the output says which it is, and the output file records the region's
file, hash and edge. On a two-parameter toy whose network rises outside the region (the
one of `tests/test_evidence.py`) the region gave ln Z = −0.06 ± 0.07, where −0.003 is
exact, and the bare network +35.8.

**A smaller edge.** The run's edge (2 in the measured runs) lets the network rise where the
boost points end. On a branch of a SINU run (13 October 2026) that was enough for its last
network to grow a maximum in a corner of the box, 19.6 above the best true value and 148 above the truth
there, at 1.87 reaches: its evidence put 99.97% of its weight there and came out 8 above every
other network's. `--region 1` keeps the integral to where the boost points are. Such an
evidence is another target than the run's region, so a true check made in the run's region
does not correct it (the summary says so).

## Single precision, and when BEST stops

BEST works in single precision and has a stopping rule of its own. The command does one
thing about each.

**A reference is subtracted.** BEST adds every batch of dead points to the running ln Z
in single precision. A real log-likelihood is near −15656, where single-precision
numbers are 10⁻³ apart, and near the bulk a batch raises ln Z by about that or less, so
part of every addition would be rounded away. BEST is therefore given the network minus a
reference: the best log-likelihood of the network's training set
(`training_data/data_it_N.csv`), as a single-precision number, so that the difference is
exact. The reference is added back in double precision to ln Z, to ln Z_box and to the
log-likelihood column of the chain; H and σ do not depend on it. BEST's own progress
display is relative to the reference, which the first line of the output names. When the
network's own training file is gone the nearest iteration's is used (any constant near the
peak serves). A run with no training set at all gets no reference: the first line says
so, and a warning follows the result when the log-likelihood is in the hundreds or beyond.

**`--tolerance` is a fraction.** BEST stops when the live points' estimate of the
evidence still to come falls below this fraction (default 10⁻³) of what the dead points
have added up. BEST's own tolerance is the logarithm of that ratio, and the command gives it
the logarithm. Given 10⁻³ itself, BEST stops every run when the live points hold half the
evidence: half of the reported ln Z is then the live points' mean, which σ does not describe.

Measured on a four-parameter Gaussian with −15656 added to its log-likelihood, 8000 live
points, four seeds (the case of `tests/test_evidence_precision.py`): taken to 10⁻³ of the
evidence without the reference, ln Z was 0.45 to 0.47 too low with σ = 0.030; with the
reference it was off by −0.008 to +0.016. Stopped at one half, three seeds
without the reference were within their error (−0.016 to +0.004, σ = 0.029): on this
case the rounding costs most in the second half of a run, so the reference and the
stopping fraction belong together.

BEST also adds ln X up in single precision, n_live_updates/n_live at every iteration,
and the reference does nothing about that. The sum is exact when `--n-live` is a power of
two and drifts otherwise: with 8000 live points BEST's ln X stood at −15.683549 after
125 460 dead points, where −15.682500 is exact, and with 8192 it was exact. That is far
below σ on a run this short. What it comes to on a real run, where ln X goes below −100,
has not been measured. The default is therefore 2048 live points and not 2000 (the error
√(H / n_live) is the same to one part in eighty), and the command
says so when it is given a number that is not a power of two. The files are named after
the number (`evidence_it_N_nlive2048.json`), so runs with different numbers stand side by
side.

One real run has been repeated with and without the reference and this stopping rule (network
30 of a 49-parameter run, 2000 live points both times, the same seed): ln Z = −15656.423 ± 0.198
with them and −15656.412 ± 0.198 without, with 197 550 dead points against
162 750. They move that value by 0.011, a twentieth of its error.

## Outputs

`RUN_DIR/evidence/evidence_it_N_nliveM[_TAG].json` holds:
- ln Z and ln Z_box, with BEST's error σ = √(H/n_live) and the information H;
- ln V_box, the prior correction, the box, and the prior table with each parameter's
  posterior mass at the box faces;
- the network file's hash (`model_sha256`) and the region used (`region`: file, hash and
  edge, or null for the bare network);
- the reference subtracted for BEST (`lnL_reference`, and the file it is the best value
  of), `tolerance` as the fraction above, and the share of the evidence the live points
  held when BEST stopped (`live_fraction`);
- the faces flagged, the number of dead points, whether the iteration cap stopped BEST
  (a warning, since ln Z would then be incomplete), and the timing.

Beside it, `evidence_it_N_nliveM[_TAG].txt` with `.paramnames` and `.ranges` is a weighted
GetDist chain of the nested samples: a posterior from the same run, independent of the
MCMC chain.

BEST's σ is defined as √(H/n_live): the scatter of ln Z from the random shrinkage of the
prior volume, for live points drawn as nested sampling assumes. It is not a convergence
test, and it does not contain
- the network's error against the true likelihood;
- live points that are not drawn as assumed (a mode that was lost, draws that are not
  independent);
- the live points' part of the sum, which is 10⁻³ of the evidence at the default
  `--tolerance` and is recorded as `live_fraction`;
- what single precision still loses (the section above);
- posterior mass outside the box, which the face check reports.

Two seeds that agree within their σ test the second of these. Two `--n-live` values that
do test the fourth as well, which depends on the number of live points. The evidence
integral is dominated by the typical set, about d/2 log-units below the peak, where
training data is plentiful. A network can therefore be accurate for the evidence and poor
at the peak, or the reverse: validate profiles and evidence separately (see
[PROFILES_AND_EVIDENCE.md](PROFILES_AND_EVIDENCE.md)).

## Live points replaced per iteration

Each iteration BEST replaces `--n-live-updates` live points, each by a slice-sampling chain of
5 steps per parameter, and each step brackets and shrinks its slice in 14 to 17 network calls,
one after another, for all the chains of the iteration together. A call costs about as much
for 10 points as for a hundred (0.5 ms against 1.2 ms for 128, on the laptop with the
production thread settings), so the time is the number of calls in sequence: 1127 an iteration
in 16 parameters and 4082 in 49, about 1200 and 4300 network evaluations per dead point. The
default is a sixteenth of the live points, 128 of 2048.

BEST draws each replacement above the likelihood of the point it replaces and sorts the old
and new points together before it removes any, so its count of the prior volume, 1/n per
removed point, holds for a batch. Measured against 10:

| target | 10 per iteration | 128 per iteration |
|---|---|---|
| correlated Gaussian, 16 parameters, H = 40 (exact 0) | -- | −0.10 ± 0.07 (5 seeds, σ = 0.14) |
| correlated Gaussian, 49 parameters, H = 79 (exact 0) | -- | −0.24 (1 seed, σ = 0.20) |
| SINU network 31 | −5341.81 ± 0.14 | −5341.41, −5341.66 ± 0.14 |
| 49-parameter network 22 (corrected) | −15656.19 ± 0.20, 613 min | −15656.58 ± 0.20, 141 min |
| exact 31-parameter ridge (ln Z − exact) | −0.57, −0.82 | −0.68, −0.37, −0.57 |

It was 2.5 times faster on the laptop and 4.3 on the cluster. The last row is a bias of BEST
itself on that target, the same with either setting: the ridge's hyperbolic arms run along
two faces of the box, and BEST comes out about 0.6 below its exact evidence, three times its
quoted error. Its error √(H / n_live) understates the error on such a target; repeat a run
with other seeds before quoting a difference of that size.

## Validation

All runs on the development laptop's CPU (Apple M3), two at a time. They were made without the reference and the
stopping fraction of the section above and have not been repeated with them: these runs
stopped when their live points held half the evidence.

| target | backend | box | BEST (live points) | reference |
|---|---|---|---|---|
| 27D Gaussian, it5 network | Cobaya | run domain | ln Z = −67.25 ± 0.29 (500), 40 min | exact −66.914 (1.1σ) |
| the same | Cobaya | ±6σ | ln Z = −67.25 ± 0.26 (500), 30 min | exact −66.915 (1.3σ) |
| 31D banana + ridge, it10 | Cobaya | an exact reference's ±4σ box | ln Z_box = −98.55 ± 0.22 (500), 28 min | exact −98.233 (1.4σ) |
| Planck 2018 ΛCDM (TTTEEE, lowE, lensing, BAO), it7 | MontePython | run domain | ln Z = −1474.31 ± 0.15 (1000), 44 min; −1474.06 ± 0.33 (200) | Laplace approximation from the run's chain: −1474.71 |

- **Box-independence.** On the 27D Gaussian, the two boxes give ln Z within 0.002 of
  each other, while their ln Z_box differ by 10.3. The exact references differ by 0.0016.
  Both sit 0.33 below the exact value: that is the network, not the box, and earlier
  comparisons of this network were also about 1.3σ low.
- **Only about half the 27D Gaussian lies inside its domain**, because several nuisance
  means sit near their hard lower bound of 0. That bound is Cobaya's own prior edge, so
  nothing is flagged, correctly.
- **The ridge's ±4σ box holds only 79.5% of the target's Gaussian mass.** The command
  flags 14 faces there, so its ln Z (−99.39; exact −99.08 on that box) is correctly not
  presented as the full evidence.
- **On the Planck run**, the run's ±3σ domain cuts into five nuisance posteriors: A_sz
  0.9% at its lower face, ksz_norm 0.5%, and three at 0.1-0.2%. This is flagged. Giving
  those parameters prior ranges equal to the domain (`--prior`) would make the domain
  the prior and remove it.

## Matching true checks to an evidence

The true check and the BEST output each record the target they were made on: model/metadata
hashes, ordered domain, integration box, resolved region/hash/edge and T=1 native objective.
These hashes describe a verified private snapshot loaded before sampling. The evidence,
analysis posterior/true-check, and derived-chain consumers load the model, metadata and
resolved region from that snapshot, rather than hashing live files after the calculation.
A change while copying stops the load; replacing the source later leaves the result bound
to the earlier snapshot. A later check of the replacement network therefore cannot correct
that result. This also works for saved runs without phase records.

Where both do, the summary (`hpc/analysis/evidence_summary.py`) applies the correction only when they are the same. A
check over the full box cannot correct an evidence over a smaller box, even at the same
iteration, and a bare-network check cannot correct a region-restricted evidence. A check or
an evidence made by a version of CLiENT that did not record targets has none: it is joined to
the other by the network's number, and the summary names the networks for
which that was done.

The summary compares the networks with each other, the difference to the last one and the
spread of the corrected evidences, whatever each was integrated over: every boost network
has its own region, and they are estimates of one evidence as far as each region holds the
posterior. A note says when they were not integrated over the same part of the box.
Evidences in different prior conventions are not compared. The convention is the name of
the convention and every parameter's prior as it was defined (its range and where that came
from); the `priors` table of an output also carries the integration box and the posterior
mass measured at its faces, which are not part of it (compared along with it, that sampled
mass would make every evidence "another convention" than every other). Several runs of one network
count as the sampler's scatter only on the same target and convention, with the same
stopping rule and tolerance, and none of them stopped at its iteration cap.

A file whose recorded target cannot be used (damaged, an unknown layout, entries missing)
gets no correction, and a note says which file. Only a file that records nothing is joined by
the network's number.

**Where the true check has no sample.** The true check's samples come from the network's
posterior chain; the evidence's from its own nested sampling. When the evidence's posterior
(its weighted chain beside the JSON) lies where the check has no sample, the correction says
nothing about that mass, and the summary gives none. It measures two shares of that posterior:
at network values above the highest of the check's samples, and farther from their mean, in
their covariance, than the farthest of them. Were the samples draws of the same posterior, each
would be about 1 / (n + 1), 0.2% for 500; above 5% there is no correction, and a note gives
both (`unsampled` in `evidence.json`). A sample's own distance is taken from the mean and
covariance of the other samples, as an evidence point's is from those of all of them: measured
from a mean and covariance it entered, it is smaller by a share that grows with the dimension,
and in 49 parameters the farthest of 500 left 1.9 to 5.1% of sound networks outside. On
end-to-end runs of 13 October 2026
(16, 31 and 49 parameters) twelve networks had at most 0.5% and 0.75%, and network 21 of
that SINU branch 99.9%: a maximum its chain never reached.

The true check uses the sampled target's values (including its region wall) in the importance
ratio, not bare-network values outside that region. Refused points weigh zero; unresolved
NaN/+inf evaluations withhold a correction. Overlap and mixing remain necessary.
