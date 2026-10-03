#!/usr/bin/env python3
"""
report_map_server.py — local tile server for the live map embedded in the
auto_classify_points (main.py) reports.

Each *_report.html has a real, pannable/zoomable Leaflet map with a band-combo
picker (presets like true_color/ndsi/cloud_qa, or pick any 3 bands yourself
for a custom RGB combo). The map asks THIS server, running on your machine,
to build the Earth Engine tile layer live — that's what lets you flip between
band combinations on demand instead of only seeing the fixed chips baked into
the report.

Run once, from the same folder as main.py (needs earthengine-api + ee auth,
same as main.py):
    python report_map_server.py --project YOUR_GCP_PROJECT

Then open the report THROUGH this server (it prints the links on startup),
e.g.:
    http://127.0.0.1:8765/20241130/training_candidates_2024-11-30_ls0_report.html

Opening the *_report.html file directly (double-click / file://) mostly works
too, but some browsers block a file:// page from calling a localhost API
(Private Network Access / CORS) — opening the http://127.0.0.1 link avoids
that entirely because then the page and the API are on the same origin.
Leave this running, Ctrl+C to stop.
"""

import argparse
import functools
import http.server
import json
import os
import socketserver
import urllib.parse

import ee

import report_main as pipeline  # reuses build_full_image / make_view / VIEWS from report_main.py

_image_cache = {}


def get_full_image(toa_id, l2_id):
    key = (toa_id, l2_id or None)
    if key not in _image_cache:
        _, full = pipeline.build_full_image(toa_id, l2_id or None)
        _image_cache[key] = full
    return _image_cache[key]


class Handler(http.server.SimpleHTTPRequestHandler):
    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        # lets the report work even when opened as a file:// page instead of
        # through this server (some browsers still allow it with these set)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/api/ping":
                return self._json(200, {"ok": True})
            if parsed.path == "/api/tile":
                return self._handle_tile(q)
            if parsed.path == "/api/inspect":
                return self._handle_inspect(q)
            if parsed.path == "/api/qa_overlay":
                return self._handle_qa_overlay(q)
            if parsed.path.startswith("/api/"):
                return self._json(404, {"error": "not found"})
            return super().do_GET()
        except Exception as e:
            return self._json(400, {"error": str(e)})

    def _handle_tile(self, q):
        toa_id = q["toa"][0]
        l2_id = q.get("l2", [""])[0] or None
        rgb_max = float(q.get("rgb_max", ["0.4"])[0])
        view_name = q["view"][0]
        full = get_full_image(toa_id, l2_id)

        if view_name == "__custom__":
            bands = [q["r"][0], q["g"][0], q["b"][0]]
            mins, maxs = [], []
            for b in bands:
                dmn, dmx = pipeline.DEFAULT_VIS.get(b, (0, 1))
                mins.append(float(q.get(f"{b}_min", [dmn])[0]))
                maxs.append(float(q.get(f"{b}_max", [dmx])[0]))
            vis_img = full.select(bands).visualize(min=mins, max=maxs)
        else:
            spec = pipeline.VIEWS.get(view_name)
            if not spec:
                return self._json(400, {"error": f"unknown view {view_name}"})
            vis_img = pipeline.make_view(full, spec, rgb_max)

        mapid = vis_img.getMapId()
        self._json(200, {"url": mapid["tile_fetcher"].url_format})

    def _handle_qa_overlay(self, q):
        toa_id = q["toa"][0]
        l2_id = q.get("l2", [""])[0] or None
        full = get_full_image(toa_id, l2_id)
        vis_img = pipeline.qa_overlay_image(full)
        mapid = vis_img.getMapId()
        self._json(200, {"url": mapid["tile_fetcher"].url_format})

    def _handle_inspect(self, q):
        toa_id = q["toa"][0]
        l2_id = q.get("l2", [""])[0] or None
        lat = float(q["lat"][0])
        lon = float(q["lon"][0])
        full = get_full_image(toa_id, l2_id)
        pt = ee.Geometry.Point([lon, lat])
        values = full.reduceRegion(ee.Reducer.first(), pt, scale=30).getInfo()
        classify = pipeline.classify(values)
        qa = pipeline.decode_qa(values.get("QA_PIXEL"))
        self._json(200, {"lat": lat, "lon": lon, "values": values, "classify": classify, "qa": qa})

    def log_message(self, fmt, *args):
        print("  " + (fmt % args))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    ap.add_argument("--port", type=int, default=pipeline.MAP_SERVER_PORT)
    ap.add_argument("--root", default=os.getcwd(), help="folder to serve reports from (default: current folder)")
    args = ap.parse_args()

    print(f"Initializing Earth Engine (project={args.project}) ...")
    ee.Initialize(project=args.project)

    handler = functools.partial(Handler, directory=args.root)
    with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"map_server running on http://127.0.0.1:{args.port}  (serving {args.root})")
        reports = [os.path.relpath(os.path.join(dp, f), args.root).replace(os.sep, "/")
                   for dp, _, files in os.walk(args.root) for f in files if f.endswith("_report.html")]
        if reports:
            print("Open a report through this server (not by double-clicking the file):")
            for r in reports:
                print(f"   http://127.0.0.1:{args.port}/{r}")
        print("Leave this running. Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
