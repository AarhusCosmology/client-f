# A small Gaussian: know the answer first

Consider the objective on $[-5,5]^2$,

$$\ell(x,y)=-\frac12\left[x^2+\frac{(y-0.6x)^2}{0.64}\right]+C.$$

For a fixed $x$, the nuisance optimum is $y_*(x)=0.6x$. It remains inside the domain for every allowed $x$, so

$$2[\ell_{\max}-\ell_{\mathrm{prof}}(x)]=x^2.$$

This gives a check on the meaning of a profile without assuming that any trained network is exact. The Cobaya wrapper adds the normalised uniform-prior constant to the native objective; that constant cancels in the relative profile but matters for evidence.

## Execute the exact checks

```sh
python docs/site/examples/analytic_checks.py
```

The script checks the nuisance optimum, two ridge identities and the equivalence of the two temperature-weight formulas used in the [methods chapter](../methods.md). It uses only the Python standard library, runs during documentation CI and makes zero backend calls. It does not train or certify a network.

## Run the actual CLiENT smoke example

Use the [first-run command](../getting-started.md). The short budget is intended to exercise configuration, true evaluation, model saving and sampling. A poor final approximation with this tiny budget is not evidence against the method; increase and validate budgets for an accuracy study.

```{literalinclude} gaussian_client.yaml
:language: yaml
:caption: CLiENT smoke configuration
```

```{literalinclude} gaussian_likelihood.yaml
:language: yaml
:caption: Analytic Cobaya target; no CLASS++ or data installation
```

After a successful run, profile `x` using the saved network and compare the true-refined curve with $x^2$:

```sh
python client.py profiles RUN_DIR --params x --points 5 --workers 2 --output PROFILE_DIR
```

Keep the true unrestricted anchor, node values and finite-call ledger separate from surrogate predictions. This second command needs the scientific environment and is not part of the lightweight documentation build.
