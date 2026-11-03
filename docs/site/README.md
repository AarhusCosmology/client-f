# CLiENT documentation site

Start with the [GitHub-readable documentation index](../README.md).

## Build locally

Build from the repository root in a documentation-only environment:

```sh
python -m pip install -r docs/site/requirements.txt
python docs/site/examples/analytic_checks.py
python docs/site/check_docs.py
python -m sphinx -W --keep-going -b html docs/site docs/site/_build/html
python docs/site/check_html.py docs/site/_build/html
python -m http.server 8765 --directory docs/site/_build/html
```

See `maintenance.md` for source ownership, generated files and examples. The documentation workflow builds the same
HTML as an artifact.

If pages have been removed or renamed since your last build, run `python -m sphinx -M clean docs/site docs/site/_build` first to remove stale generated output.
