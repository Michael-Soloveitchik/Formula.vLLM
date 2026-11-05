#!/usr/bin/env python3
"""Dashboard server: python http.server + gzip for text/json responses.

Serves from the benchmark ROOT (web/index.html fetches ../index.json,
../results/all.json, ../data/...). Gzips .json/.jsonl/.html/.js/.css/.md/.txt
when the client accepts it: index.json 115 KB -> 33 KB, data/index.jsonl
9.9 MB -> 1.8 MB, which matters over the Tailscale Funnel (~150 KB/s).

    python3 serve.py [port] [--bind 127.0.0.1]
"""
import gzip
import http.server
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GZ_EXT = {".json", ".jsonl", ".html", ".js", ".css", ".md", ".txt", ".diff", ".svg"}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_head(self):
        path = self.translate_path(self.path)
        ext = os.path.splitext(path)[1].lower()
        accepts = "gzip" in self.headers.get("Accept-Encoding", "")
        if not (accepts and ext in GZ_EXT and os.path.isfile(path)):
            return super().send_head()
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError:
            self.send_error(404, "File not found")
            return None
        body = gzip.compress(raw, 6)
        self.send_response(200)
        self.send_header("Content-type", self.guess_type(path))
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Vary", "Accept-Encoding")
        self.end_headers()
        return io.BytesIO(body)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    port = int(args[0]) if args else 8000
    bind = "127.0.0.1"
    if "--bind" in sys.argv:
        bind = sys.argv[sys.argv.index("--bind") + 1]
    print(f"dashboard: http://{bind}:{port}/web/  (root={HERE}, gzip on)")
    http.server.ThreadingHTTPServer((bind, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
