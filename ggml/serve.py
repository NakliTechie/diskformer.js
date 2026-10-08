# Serves the repo root for ggml/chat.html and ggml/gate.html with caching off, so a rebuilt wasm or edited worker is
# always the one that runs: python3 ggml/serve.py [port]
import functools, http.server, sys

class NoStore(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8797
http.server.ThreadingHTTPServer(('127.0.0.1', port), NoStore).serve_forever()
