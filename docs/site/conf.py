"""Documentation only: source parsing never imports scientific backends."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / '_ext'))
from prepare import prepare

revision = prepare(ROOT, HERE)
project = 'CLiENT'
author = 'Aarhus Cosmology'
copyright = '2026, Aarhus Cosmology'
version = revision[:7]
release = revision
extensions = ['myst_parser', 'autoapi.extension', 'sphinx.ext.napoleon',
              'sphinx.ext.mathjax', 'sphinx_copybutton']
source_suffix = {'.md': 'markdown', '.rst': 'restructuredtext'}
exclude_patterns = ['_build', '_generated/api_source', 'README.md', 'requirements.txt']
myst_enable_extensions = ['colon_fence', 'dollarmath', 'amsmath', 'deflist']
myst_heading_anchors = 6
html_theme = 'pydata_sphinx_theme'
html_title = 'CLiENT documentation'
html_static_path = ['_static']
html_css_files = ['client.css']
html_theme_options = {
    'logo': {'text': 'CLiENT'},
    'github_url': 'https://github.com/AarhusCosmology/client',
    'navbar_align': 'left',
    'show_nav_level': 2,
    'navigation_depth': 3,
    'external_links': [{'name': 'CLASS++', 'url': 'https://aarhuscosmology.github.io/CLASSpp/'}],
    'announcement': 'Manual · source ' + version,
}
html_sidebars = {'**': ['sidebar-nav-bs']}
html_context = {'default_mode': 'light'}
html_show_sourcelink = True
html_last_updated_fmt = ''
autoapi_type = 'python'
autoapi_dirs = [str(HERE / '_generated/api_source')]
autoapi_python_use_implicit_namespaces = False
autoapi_root = 'api'
autoapi_options = ['members', 'undoc-members', 'show-module-summary']
autoapi_add_toctree_entry = True
autoapi_keep_files = False
# Several existing scientific docstrings use Markdown inline code. Translate this
# notation for the reStructuredText API renderer; never edit the runtime sources.
def api_docstring(app, what, name, obj, options, lines):
    import re
    if name.rsplit('.', 1)[-1] in ('find_maxima', 'audit_optima'):
        for i, line in enumerate(lines):
            indent = line[:len(line)-len(line.lstrip())]
            line = line.lstrip()
            if line == 'dicts with': lines[i] = indent + 'dicts with::'
            for prefix, param in [('project : ', 'project'), ('inputs, targets : ', 'inputs'), ('evaluate(points) -> (inputs, targets, derived): ', 'evaluate'), ('build_loss(targets) : ', 'build_loss'), ('config : ', 'config'), ('n_append : ', 'n_append')]:
                if line.startswith(prefix): lines[i] = indent + ':param ' + param + ': ' + line[len(prefix):]
    if name.rsplit('.', 1)[-1] == 'tempered_verdict':
        for i, line in enumerate(lines):
            if line.lstrip().startswith('Returns dict('):
                lines.insert(i, '')
                break
    for i, line in enumerate(lines):
        lines[i] = re.sub(r'(?<!`)`([^`\n]+)`(?!`)', r'``\1``', line)

def setup(app):
    app.connect('autodoc-process-docstring', api_docstring)

# This is a curated API, not a scan of every implementation dependency.
# Unselected internal imports are not re-exported; all document/link warnings stay fatal.
suppress_warnings = ['autoapi.python_import_resolution']
