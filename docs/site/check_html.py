"""Check all generated internal links, anchors, images and downloads."""
from pathlib import Path
import sys
from urllib.parse import urlsplit, unquote
from bs4 import BeautifulSoup


def main():
    root=Path(sys.argv[1]).resolve()
    files=[p for p in root.rglob('*.html') if '_static' not in p.relative_to(root).parts]
    ids={}; soups={}
    for p in files:
        soup=BeautifulSoup(p.read_text(),'html.parser'); soups[p]=soup
        ids[p]={t['id'] for t in soup.select('[id]')}
    errors=[]; checked=0
    for p,soup in soups.items():
        for tag in soup.select('[href], img[src], script[src]'):
            url=tag.get('href',tag.get('src','')); parsed=urlsplit(url)
            if parsed.scheme or parsed.netloc or url.startswith('//'): continue
            name=unquote(parsed.path)
            q=(p.parent/name).resolve() if name else p
            if q.is_dir(): q=q/'index.html'
            if not q.is_relative_to(root) or not q.exists():
                errors.append(f'{p.relative_to(root)} -> missing {url}');continue
            if parsed.fragment and q.suffix=='.html' and unquote(parsed.fragment) not in ids.get(q,set()):
                errors.append(f'{p.relative_to(root)} -> absent anchor {url}')
            checked+=1
    if errors:
        print('\n'.join(sorted(set(errors))));raise SystemExit(1)
    print(f'Passed: {len(files)} HTML pages; {checked} local links/assets/anchors checked.')

if __name__=='__main__': main()
