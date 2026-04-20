Glui cargo workspace manual prompt
#act as a senior developer. Carefully study the documentation of my glui stack and generate a comprehensive instruction / build manual for all the combined .zip packages ~ use my included initial build summary to gain further insight——————##################rust-gpui-app.zip

Key features wired
   •   Tanner variables (TannerConfig { active, tau, nu, alpha, resonance_delta }) threaded through the engine.
   •   CF4-per-segment stub that simulates τ-modulated bands, invariants, edge weight, Berry-curvature texture.
   •   Concurrency: mpsc (controls) + watch (last-value snapshots) for stable real-time rendering.
   •   UI: default egui fallback (compiles anywhere) showing controls + live bands; switchable to GPUI via feature flag.
   •   Recording scaffold: CSV writer with the canonical long-form schema (extend to Parquet by enabling the parquet feature and filling in writer code).

Where to plug your solver
   •   Replace the simulated math in crates/floquet_core/src/lib.rs with your real Floquet engine:
      •   Implement Propagator for Cf4Segment using your CF4 per-segment integrator.
      •   Populate Snapshot { bands_k, quasienergies, berry_curv, invariants, edge_weight, ... }.
   •   Keep τ usage local inside the step logic:
      •   ε_eff(t) = ε0 * (1 + α * sin(2π t / τ))
      •   Kimi/baseline separation via TannerConfig.active.

Extending the UI (quick wins)
   •   Add Phase Diagram & BZ Heatmap tabs in host_ui:
      •   Phase map: maintain a texture buffer; stream tiles via the same watch pattern.
      •   Heatmap: upload Snapshot.berry_curv into a texture; draw iso-lines.
   •   Add CSV ingest to host_ui (Right pane):
      •   Use the csv crate to map (k, ε, band) into the bands buffer for replay.
   •   Wire recording toggle:
      •   Create a Recorder in the app main; on ControlAction::Record(true) call open_csv(...),
then write_snapshot(&snap) on each frame.

Canonical schema (already in the recorder)

metric, mode, cycle_index, time_s, value,
sweep_param, sweep_value, order, theta, loop_size_pi,
sweep2_param, sweep2_value

   •   For bands: metric="bands", mode="baseline|kimi", theta=k, value=ε(k).
   •   For future heatmaps/phase tiles, fill theta/loop_size_pi or the sweep columns as appropriate.

⸻

. gpui-app.zip

What I added (plug-and-play)

1) Phase Map tiling pipeline
   •   New phase_tile(...) on the Propagator trait (implemented in Cf4Segment):
      •   Produces a tile over (A, Ω) with values in [-1,1] representing an invariant proxy.
      •   Sends tiles to the UI via a shared DataBus (see below).

2) Unified data bus (frames + tiles)
   •   data_model now defines:
      •   enum DataBus { Frame(Snapshot), Phase(PhaseTile) }
      •   UI subscribes once and can render live bands and phase-map tiles.

3) CSV ingest overlay (Bands tab)
   •   In host_ui (egui fallback), Right pane includes a CSV loader:
      •   Paste a path, click Load CSV, and it overlays (k, ε) points on the Bands plot.
      •   Reads the canonical long-form schema you’ve been using (pulls theta and value columns).

4) GPUI host scaffold
   •   Feature-gated gpui_host::run(...) added (minimal stub so the crate structure is ready).
   •   Keep building with egui by default; enable GPUI later:

cargo run -p app --no-default-features --features gpui



5) App wiring
   •   app now publishes both:
      •   live frames at ~60 Hz (DataBus::Frame)
      •   on-demand phase tiles when you click “Phase Map” (DataBus::Phase)
   •   Controls also include Run / Pause / Reset / Rebuild / Phase Map.

⸻

Where to plug your real math
   •   Replace the toy math in:
      •   crates/floquet_core/src/lib.rs::Cf4Segment::step
      •   crates/floquet_core/src/lib.rs::Propagator::phase_tile
   •   Keep τ-threading via TannerConfig (already passed into both).

⸻

: rust-gpui-app.zip

What’s in this build

1) Progressive phase-map tiling (A×Ω)
   •   Engine streams 64×64 tiles over the (A, Ω) grid into the UI as DataBus::Phase.
   •   Click “Phase Map” in the UI to watch tiles fill in live.

2) Parquet writer (arrow2)
   •   recorders crate adds a feature-gated Parquet pipeline that mirrors the canonical long-form schema you’re using for sweeps.
   •   CSV always on; Parquet is enabled by --features parquet.

3) BZ heatmap + cursor ∑F dA probe
   •   Snapshot carries berry_curv + (w,h).
   •   UI renders a real heatmap texture and lets you probe a local average via a radius slider. The readout shows ∑F dA (mean over the picked patch).

4) Optional WebView panel (hybrid path)
   •   Feature webview adds a wry-based window. It’s a minimal canvas right now—meant to host your existing WebGL/WebGPU shaders.
   •   Use this to embed your legacy Bloch sphere/BZ shaders without browser friction.

5) CLI for batch maps → PNG + CSV (+ Parquet)
   •   cli crate subcommand:

cargo run -p cli -- phase \
  --width 512 --height 512 \
  --a_min 0.0 --a_max 2.0 --o_min 0.05 --o_max 5.0 \
  --out_png phase.png --out_csv phase.csv


   •   Add --features parquet to also emit phase.parquet.

⸻


⸻

Where to plug your real math
   •   crates/floquet_core/src/lib.rs
      •   Cf4Segment::step — replace the band/curvature toy model with your CF4-per-segment solver and Berry curvature computation.
      •   Propagator::phase_tile — compute your invariant or winding/Chern proxy per (A, Ω) and return normalized values in [-1,1].

The τ-threading is already present via TannerConfig { tau, alpha, nu, resonance_delta } and is passed through the runtime → UI → export stack.

⸻

Files touched (high level)
   •   data_model — extended Snapshot (adds Berry grid dims), new DataBus::Phase.
   •   floquet_core — CF4 segment stub + phase tiler; fills Berry texture.
   •   host_ui — egui UI:
      •   Tabs: Bands, Phase Map, Time Trace, BZ Heatmap.
      •   CSV paste overlay for bands.
      •   BZ probe with radius slider and live ∑F dA readout.
      •   (feature) webview_bridge stub for hybrid route.
   •   recorders — CSV baseline + (feature) Parquet via arrow2.
   •   cli — batch phase map → PNG + CSV (+ Parquet if enabled).
   •   app — progressive tiling streamer (64×64 tiles).

⸻

: rust-gpui-app.zip

What’s included (and where)
   •   data_model – extended Snapshot (with berry_curv, berry_w, berry_h) and new DataBus::Phase.
      •   crates/data_model/src/lib.rs
   •   floquet_core – CF4 segment stub + phase tiler; fills a Berry texture.
      •   crates/floquet_core/src/lib.rs
      •   Implements Propagator::step(...) and Propagator::phase_tile(...)
   •   host_ui (egui)
      •   Tabs: Bands, Phase Map, Time Trace, BZ Heatmap
      •   CSV paste overlay (canonical long-form) for Bands
      •   BZ probe with radius slider and live ∑F dA readout
      •   (feature) webview_bridge stub for hybrid WebView route
      •   crates/host_ui/src/lib.rs
   •   recorders – CSV baseline + (feature) Parquet via arrow2
      •   crates/recorders/src/lib.rs
      •   Feature flag: parquet
   •   cli – batch phase map → PNG + CSV (+ Parquet if enabled)
      •   crates/cli/src/main.rs
      •   Command: phase
   •   app – progressive tiling streamer (64×64 tiles)
      •   crates/app/src/main.rs

Build & run

unzip rust-gpui-app.zip && cd rust-gpui-app

# Desktop app (egui fallback)
cargo run -p app

# Stream phase tiles on demand: click “Phase Map”

# GPUI backend (stub; ready to flesh out)
cargo run -p app --no-default-features --features gpui

# Parquet (CSV always on; Parquet behind feature)
cargo run -p app --features parquet

# CLI batch → PNG + CSV (+ Parquet with feature)
cargo run -p cli -- phase --width 512 --height 512 --out_png phase.png --out_csv phase.csv
cargo run -p cli --features parquet -- phase --out_parquet phase.parquet

Notes for your team
   •   τ-threading is exposed as TannerConfig { tau, alpha, nu, resonance_delta } and flows through runtime → UI → exports.
   •   The canonical long-form schema is consistent across app streams, CLI, CSV, and Parquet.
   •   Swap the toy math for your true CF4-per-segment solver in:
      •   Cf4Segment::step (bands, Berry)
      •   Propagator::phase_tile (invariant map)

: rust-gpui-app_v3.zip

What you now have
   •   Off-thread tile scheduler (cancel/priority):
      •   app/src/main.rs → TileScheduler uses a BinaryHeap for priority and a CancellationToken for generation-based cancel.
      •   UI click on Phase Map sends a refined PhaseRequest{... refine: true, priority: 1.0 }.
   •   One-click PNG/Parquet recording from the desktop app:
      •   Left pane has a Record (PNG/Parquet) toggle, plus base path textbox.
      •   recorders::Recorder writes:
         •   CSV (always) in canonical long-form schema.
         •   PNG bands frames via simple rasterizer (write_bands_png).
         •   Parquet (feature parquet) with metadata (schema_version, generator, timestamp).
   •   WebView IPC for shader-heavy views:
      •   host_ui::webview_bridge stub using wry; pushes band frames via evaluate_script into window.__bridge__.frame(JSON).
      •   Ready to swap in TypedArray/binary path later.
   •   Real(ish) edge-mode locator pass (glow overlay):
      •   Snapshot.edge_mask: Option<Vec<f32>> (0..1).
      •   Bands tab draws a glow using per-point alpha; plug in your true edge-localization metric.
   •   Canonical schema validator + Parquet metadata:
      •   recorders::validate_schema() checks column presence.
      •   Parquet writer stamps key/value metadata including schema_version and timestamp.
   •   GPUI renderer scaffold:
      •   Feature gpui path prints stub; slot in native canvas/heatmap draws when you’re ready.
      •   Egui fallback remains the default and fully functional.

Integration notes (where to swap in your real math)
   •   crates/floquet_core/src/lib.rs
      •   Replace toy bands/Berry with your CF4-per-segment and curvature.
      •   Populate edge_mask with your proper edge-mode locator.
      •   phase_tile(...) → compute invariant value per (A, Ω) cell.
   •   crates/app/src/main.rs
      •   Tile scheduler already streams 64×64 tiles; refine path emits 32×32 tiles prioritized near click.
   •   crates/recorders/src/lib.rs
      •   Extend the band PNG writer to overlay invariants or error bars if needed.
      •   Add additional metrics (e.g., OTOC/echo) by calling write_record with the same canonical schema columns.

rust-gpui-app_v4.zip

What I added
1.	#gpui renderer (native canvas/heatmap hooks)

   •   New module: crates/host_ui/src/gpui_renderer.rs
   •   Feature-gated with --features gpui.
   •   Currently a clean scaffold that boots the GPUI path; slots are in place for:
      •   Bands polyline plot (wrapped ε(k))
      •   Phase-map heatmap tiles (subscribe to DataBus::Phase)
      •   BZ heatmap with cursor probe
      •   Record toggle plumbing (mirrors egui path)

2.	WebView binary IPC for shader-heavy views

   •   New module: crates/host_ui/src/webview_bridge.rs (feature webview).
   •   JS side exposes window.__bridge__.frame64({k64, e64}), which base64-decodes directly into Float32Array and draws to a canvas (OffscreenCanvas-ready).
   •   Rust side shows how to encode Vec<f32> to base64 and push once via evaluate_script. You can call this repeatedly from your runtime thread if you embed the bridge.

3.	Kept all previously wired systems intact

   •   Off-thread tile scheduler with cancel/priority.
   •   One-click PNG/Parquet recording from the app.
   •   Edge-mode glow overlay in Bands.
   •   Canonical schema validator + Parquet metadata (version + timestamp).
   •   CLI batch phase-map → PNG + CSV (+ Parquet behind --features parquet).

Where to plug in your logic
   •   GPUI renderer: gpui_renderer.rs
      •   Replace the placeholder draw calls with your GPUI canvas primitives.
      •   Subscribe to watch::Receiver<Arc<DataBus>> and render Snapshot / PhaseTile.
   •   WebView (binary): webview_bridge.rs
      •   If you want true SharedArrayBuffer + OffscreenCanvas, add cross-origin isolation headers. Until then, the base64 → Float32Array path is robust and already fast enough for typical frame sizes.
   •   Runtime streaming: crates/app/src/main.rs
      •   Keep using the same watch::Sender<Arc<DataBus>> to notify both egui/GPUI/WebView frontends.
      •   The tile scheduler already supports refine clicks; just send ControlAction::PhaseRequest{...} from the UI.

All set—both modules are built and ready to drop in.

: rust-gpui-app_v5_modules.zip

What’s inside (plug straight into your v4 workspace)

1) GPUI renderer (native path)
   •   File: crates/host_ui/src/gpui_renderer.rs
   •   Subscribes to DataBus (frames + phase tiles).
   •   Runs a render loop you can replace with your GPUI canvas primitives.
   •   Start it by compiling with --features gpui:

cargo run -p app --no-default-features --features gpui



Notes: The upstream gpui API moves fast; this scaffold isolates all GPUI-specific calls so you can swap in your immediate-mode lines/heatmaps without touching the rest of the app.

2) WebView bridge (shader-heavy route)
   •   File: crates/host_ui/src/webview_bridge.rs
   •   Implements OffscreenCanvas + Worker with a base64 → Float32Array path for large frames.
   •   Exposes two JS entry points:
      •   window.__bridge__.frame64({ k64, e64 }) for high-throughput pushed frames
      •   window.__bridge__.frame({ k: f32[], e: f32[] }) for small control/debug frames
   •   Bring it up:

cargo run -p app --features webview


   •   The HTML sets COOP/COEP headers for cross-origin isolation so you can later switch to true SharedArrayBuffer if you want binary ring buffers.

⸻

How to wire them into your v4 app
1.	Replace the host_ui crate with the one in the zip (or copy the two files into your existing crates/host_ui/src/ and keep your lib.rs re-exports).
2.	Keep your egui path as-is—this is additive. You can toggle which UI you want at compile time:
      •   egui: cargo run -p app
      •   gpui: cargo run -p app --no-default-features --features gpui
      •   webview: cargo run -p app --features webview

No changes are required to the runtime, tiler, recorders, or data model. They already stream DataBus::Frame/Phase and support one-click PNG/Parquet recording.

⸻

: rust-gpui-app_v6.zip

What’s wired now
   •   GPUI path (native renderer)
      •   host_ui::gpui_renderer subscribes to frames & tiles and runs a render loop. It’s logging by default so it compiles even if the upstream gpui API has shifted—swap the log lines for your gpui canvas primitives in one place.
      •   Compile switch:

cargo run -p app --no-default-features --features gpui


   •   WebView with SAB + OffscreenCanvas
      •   host_ui::webview_bridge creates a SharedArrayBuffer ring (k/e packed), drives a Worker on an OffscreenCanvas, and supports a base64→Float32Array fallback.
      •   Two push routes:
         •   window.__bridge__.push64({k64,e64}) (high-throughput)
         •   window.__bridge__.frame({k:[], e:[]}) (debug/small)
      •   Launch:

cargo run -p app --features webview


   •   Tile-level progressive refinement overlays
      •   In Phase tab (egui), click to request a refined tile; you’ll see:
         •   Pending tile box (yellow) → turns green when the tile arrives.
      •   Off-thread scheduler still handles priority/cancel.
   •   Keyboard “inspect” mode
      •   Hold I while hovering the BZ heatmap to probe; tweak Probe radius slider in the left pane.
      •   Live readout shows ∑F dA over the probe disk.
   •   One-click PNG/Parquet recording
      •   Still wired through Recorder with the canonical schema; Parquet via --features parquet.

Where to plug in your real math
   •   crates/floquet_core/src/lib.rs
      •   Replace the toy Cf4Segment::step and phase_tile bodies with your CF4-per-segment solver, Berry curvature, and invariant computation.
   •   crates/recorders/src/lib.rs
      •   Extend write_snapshot to emit OTOC/echo and any custom metrics—just keep the long-form canonical columns.
   •   crates/host_ui/src/egui_host.rs
      •   The bands, phase-map, and BZ heatmap draws are fully in place (including edge-mode glow and overlays). Add any extra readouts or buttons here
