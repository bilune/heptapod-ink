#!/usr/bin/env python3
"""Local static server for web/ with cross-origin isolation (enables multi-threaded WASM).

  python3 web/serve.py            -> http://localhost:8795   (works from any cwd)
  PORT=9000 python3 web/serve.py
"""
import functools
import http.server
import os

WEB = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", "8795"))


class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"serving {WEB} on http://localhost:{PORT}")
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), functools.partial(H, directory=WEB)).serve_forever()
