import http.server
import socketserver
import os
import sys

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Mute polling noise (200 OKs and 404 Not Found during startup)
        try:
            msg = format % args
            if "200" in msg or "404" in msg:
                return
        except Exception:
            pass
        super().log_message(format, *args)

    def do_GET(self):
        # Strip query parameters
        path = self.path.split('?')[0]
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        if path in ["/", "/index.html"]:
            self.send_file(os.path.join(project_root, "api", "static", "index.html"), "text/html")
        elif path == "/api/state":
            self.send_file(os.path.join(project_root, "data", "active_game_state.json"), "application/json")
        elif path == "/api/entities":
            self.send_file(os.path.join(project_root, "data", "active_entities.json"), "application/json")
        elif path == "/api/species":
            self.send_file(os.path.join(project_root, "data", "species_db.json"), "application/json")
        elif path == "/api/biome_map.png":
            self.send_file(os.path.join(project_root, "images", "biome_map.png"), "image/png")
        else:
            self.send_error(404, f"File not found: {path}")

    def send_file(self, filepath, content_type):
        if not os.path.exists(filepath):
            self.send_error(404, f"File not found: {os.path.basename(filepath)}")
            return
        
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Internal server error: {str(e)}")

def start_server(port=8000):
    handler = DashboardHandler
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"\n[Dashboard Server] Running live at http://localhost:{port}")
            httpd.serve_forever()
    except Exception as e:
        print(f"[Dashboard Server] Error starting server: {e}", file=sys.stderr)

if __name__ == "__main__":
    start_server()
