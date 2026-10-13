# Serves the repo root for ggml/chat.html and ggml/gate.html with caching off, so a rebuilt wasm or edited worker is
# always the one that runs: python3 ggml/serve.py [port] [root]
# Range requests are answered (206), and any origin may read, so a page on another local port can copy a GGUF from
# disk into its OPFS with the worker's chunked downloader, e.g. python3 ggml/serve.py 8798 ~/models then
# download {url: 'http://127.0.0.1:8798/width3/<file>.gguf'} from the page on :8797.
import functools, http.server, os, re, sys

class NoStore(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Range')
        self.send_header('Access-Control-Expose-Headers', 'Content-Range, Content-Length')
        self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        m = re.fullmatch(r'bytes=(\d+)-(\d*)', self.headers.get('Range', ''))
        path = self.translate_path(self.path)
        if not m or not os.path.isfile(path):
            return super().do_GET()
        size = os.path.getsize(path)
        a = int(m.group(1))
        b = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
        if a > b:
            self.send_error(416)
            return
        self.send_response(206)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Range', f'bytes {a}-{b}/{size}')
        self.send_header('Content-Length', str(b - a + 1))
        self.end_headers()
        with open(path, 'rb') as f:
            f.seek(a)
            left = b - a + 1
            while left > 0:
                chunk = f.read(min(left, 1 << 20))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8797
root = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
handler = functools.partial(NoStore, directory=root)
http.server.ThreadingHTTPServer(('127.0.0.1', port), handler).serve_forever()
