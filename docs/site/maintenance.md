# Building and maintaining the manual

## Documentation-only environment

From the repository root, with Python 3.10 or newer:

```sh
python -m venv .venv-docs
. .venv-docs/bin/activate
python -m pip install -r docs/site/requirements.txt
python docs/site/examples/analytic_checks.py
python docs/site/check_docs.py
python -m sphinx -W --keep-going -b html docs/site docs/site/_build/html
python docs/site/check_html.py docs/site/_build/html
python -m http.server 8765 --directory docs/site/_build/html
```

Open `http://localhost:8765`. TensorFlow, CLASS++, MPI, Planck data and the scientific environment are not needed for this build. A browser connection is needed for CDN-served MathJax to render mathematical notation.

## Sources and generated files

- `docs/site/` holds the navigation, overview pages, build configuration and examples.
- The guides stay in `docs/`, `hpc/` and the repository root. The build stages copies, fixes relative links and copies referenced figures/PDFs. It does not edit the originals.
- `_generated/reference/configuration.md` comes from the configuration source AST. The JSON beside it makes the inventory machine-readable. Conditional parser defaults (currently the true-batch convergence threshold) are labelled and checked separately, rather than being mistaken for unconditional defaults. The checks exercise real parser defaults using only the lightweight configuration modules.
- `_generated/api_source/` contains the selected source files for AutoAPI. Runtime code is never imported by the API generator.
- Generated pages/assets and `_build/` are ignored by Git. `_generated/provenance.json` records the revision and guide hashes.

Add a guide to the curated navigation when it is appropriate for a task-oriented entry point. Linked Markdown reports are included in the searchable library automatically. Other repository files link to the source revision. A missing repository-relative link fails the checks.

When removing or renaming pages, clean the old build with `python -m sphinx -M clean docs/site docs/site/_build` before rebuilding; `-E` alone does not remove stale HTML files.

## Automated checks

The documentation workflow builds on pull requests and pushes to `main` and `features` (on the organisation's self-hosted runners), runs the small analytic checks, checks configuration claims and validates local HTML targets and anchors. It uploads the HTML as a review artifact. It **does not publish a website or run expensive scientific examples**.

The analytic script checks exact Gaussian/ridge and importance-weight identities. It is not an end-to-end training test. The short CLiENT Gaussian run is a separate reproducible smoke command, and real-likelihood examples retain their recorded validation provenance.
