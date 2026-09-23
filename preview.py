"""Build and serve locally; rebuild when manuscript or design files change."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent

def snapshot():
    files = [ROOT / n for n in ('book.json', 'build.py', 'style.css', 'comments.js', 'math-config.js')]
    for folder in ('manuscript', 'templates', 'figures', 'slides', 'vendor'):
        files.extend(p for p in (ROOT / folder).rglob('*') if p.is_file())
    return {str(p): p.stat().st_mtime_ns for p in files}

def rebuild():
    return subprocess.run([sys.executable, str(ROOT / 'build.py')], cwd=ROOT).returncode

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8261)
    args = parser.parse_args()
    previous = snapshot()
    if rebuild():
        raise SystemExit('Build failed; fix the reported error before previewing.')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(SimpleHTTPRequestHandler, directory=str(ROOT / '_site')))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f'Preview: http://127.0.0.1:{args.port}/ — save a manuscript, then refresh the browser.', flush=True)
    try:
        while True:
            time.sleep(1)
            current = snapshot()
            if current != previous:
                previous = current
                print('Changes detected; rebuilding...', flush=True)
                if rebuild():
                    print('Build failed. Fix the error and save again.', flush=True)
                else:
                    print('Ready — refresh your browser.', flush=True)
    except KeyboardInterrupt:
        server.shutdown()

if __name__ == '__main__':
    main()
