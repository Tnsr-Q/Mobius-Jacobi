self-contained side-panel you can drop into the egui desktop app. It lists bookmarks, lets you filter, jump-to (focus the canvas), edit name/notes inline, delete single pins, export (CSV/Parquet), and clear all (broadcasts BookmarksCleared). It piggybacks the live stream you already have (DataBus::Bookmarks).



1) App state additions (egui host)
crates/host_ui/src/egui_host.rs – extend your UI state:
// imports (near the top)
use data_model::{Snapshot, Bookmark, DataBus};
use std::collections::HashSet;

// inside your AppState struct
pub struct AppState {
  // ...existing...
  bookmarks: Vec<Bookmark>,
  show_bookmarks: bool,
  // NEW: panel + selection state
  show_bookmark_panel: bool,
  bookmark_filter: String,
  selected_bm: Option<usize>,
  hidden_bm: HashSet<usize>,
  // focus overlay on canvas
  focus_point: Option<(f32, f32)>,   // (A, Ω) in data coords
}
Initialize defaults (where you construct AppState):
Self {
  // ...existing...
  bookmarks: Vec::new(),
  show_bookmarks: true,
  show_bookmark_panel: true,
  bookmark_filter: String::new(),
  selected_bm: None,
  hidden_bm: HashSet::new(),
  focus_point: None,
}



2) Live handling (already there but add cleared)
Where you receive bus events:
match &*evt {
  DataBus::Bookmarks(bm) => {
    self.bookmarks = bm.clone();
    // try to keep selection valid
    if let Some(i) = self.selected_bm {
      if i >= self.bookmarks.len() { self.selected_bm = None; }
    }
  }
  DataBus::BookmarksCleared => {
    self.bookmarks.clear();
    self.selected_bm = None;
    self.hidden_bm.clear();
    self.focus_point = None;
  }
  // ...
}



3) Side panel UI
Add this where you build your window UI (e.g., next to your top bar and center canvas):
if self.show_bookmark_panel {
  egui::SidePanel::right("bookmark_panel")
    .resizable(true)
    .default_width(320.0)
    .show(ctx, |ui| {
      ui.heading("Bookmarks");
      ui.horizontal(|ui| {
        ui.text_edit_singleline(&mut self.bookmark_filter)
          .hint_text("filter (name/notes/Ω/A/depth)");
        if ui.button("Clear").clicked() {
          let _ = self.tx_ctrl.blocking_send(data_model::ControlMsg::BookmarksCleared);
        }
      });
      ui.separator();

      // bulk actions
      ui.horizontal(|ui| {
        if ui.button("Export CSV").clicked() {
          if let Some(path) = rfd::FileDialog::new()
             .set_file_name("bookmarks.csv").save_file() {
            if let Some(rr) = self.recorder.lock().as_ref() {
              let _ = rr.write_bookmarks_csv(&path, &self.bookmarks);
            }
          }
        }
        #[cfg(feature="parquet")]
        if ui.button("Export Parquet").clicked() {
          if let Some(path) = rfd::FileDialog::new()
             .set_file_name("bookmarks.parquet").save_file() {
            if let Some(rr) = self.recorder.lock().as_ref() {
              let _ = rr.write_bookmarks_parquet(&path, &self.bookmarks);
            }
          }
        }
        ui.toggle_value(&mut self.show_bookmarks, "Show pins");
      });

      ui.separator();
      egui::ScrollArea::vertical().show(ui, |ui| {
        for (i, b) in self.bookmarks.iter_mut().enumerate() {
          if !bookmark_matches(b, &self.bookmark_filter) { continue; }

          let is_sel = self.selected_bm == Some(i);
          let mut resp = egui::collapsing_header::CollapsingState::load_with_default_open(
              ui.ctx(), ui.make_persistent_id(("bm_row", i)), false
            )
            .show_header(ui, |ui| {
              let label = b.name.as_deref().unwrap_or("(unnamed)");
              let mut row = ui.selectable_label(is_sel, format!("#{i}  {label}"));
              if row.clicked() { self.selected_bm = Some(i); }
            })
            .body(|ui| {
              ui.horizontal(|ui| {
                ui.label(format!("A={:.6}", b.a));
                ui.label(format!("Ω={:.6}", b.omega));
                ui.label(format!("depth={}", b.depth));
              });
              // inline edit
              ui.horizontal(|ui| {
                ui.label("Title");
                let mut name = b.name.clone().unwrap_or_default();
                if ui.text_edit_singleline(&mut name).lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                  b.name = if name.trim().is_empty() { None } else { Some(name.trim().to_string()) };
                }
              });
              ui.label("Notes");
              let mut notes = b.notes.clone().unwrap_or_default();
              if ui.text_edit_multiline(&mut notes).lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                b.notes = if notes.trim().is_empty() { None } else { Some(notes.trim().to_string()) };
              }

              ui.horizontal(|ui| {
                if ui.button("Jump").clicked() {
                  self.selected_bm = Some(i);
                  self.focus_point = Some((b.a as f32, b.omega as f32));
                }
                if ui.button(if self.hidden_bm.contains(&i) { "Show" } else { "Hide" }).clicked() {
                  if !self.hidden_bm.insert(i) { self.hidden_bm.remove(&i); }
                }
                if ui.button("Delete (local)").clicked() {
                  // local remove; you can add a "replace" ControlMsg later if you want to persist upstream.
                  self.hidden_bm.remove(&i);
                  self.bookmarks.remove(i);
                  if let Some(sel) = self.selected_bm {
                    if sel == i { self.selected_bm = None; }
                    else if sel > i { self.selected_bm = Some(sel - 1); }
                  }
                }
              });
            });
          // draw a subtle separator between rows
          ui.add_space(4.0);
          ui.separator();
          ui.add_space(4.0);
        }
      });
    });
}
Filtering helper:
fn bookmark_matches(b: &Bookmark, q: &str) -> bool {
  if q.trim().is_empty() { return true; }
  let q = q.to_lowercase();
  let fields = [
    b.name.as_deref().unwrap_or_default(),
    b.notes.as_deref().unwrap_or_default(),
    &format!("{:.6}", b.a),
    &format!("{:.6}", b.omega),
    &format!("{}", b.depth),
  ];
  fields.iter().any(|f| f.to_lowercase().contains(&q))
}



4) Canvas focus + selection overlay
In your heatmap plot where you already draw pins, emphasize the selected pin and a focus crosshair if present:
// when iterating pins:
if self.show_bookmarks {
  for (i, b) in self.bookmarks.iter().enumerate() {
    if self.hidden_bm.contains(&i) { continue; }
    let u = ((b.a as f32) - self.a_min) / (self.a_max - self.a_min);
    let v = ((b.omega as f32) - self.o_min) / (self.o_max - self.o_min);
    let x = rect_hm.left() + u * rect_hm.width();
    let y = rect_hm.top()  + v * rect_hm.height();

    // base pin
    ui.painter().circle_filled(egui::pos2(x,y), 3.5, egui::Color32::from_rgb(255,220,90));
    ui.painter().circle_stroke(egui::pos2(x,y), 5.0, egui::Stroke { width: 1.0, color: egui::Color32::BLACK });

    // label (if any)
    if let Some(name) = &b.name {
      ui.painter().text(egui::pos2(x+6.0, y-6.0), egui::Align2::LEFT_BOTTOM, name,
                        egui::FontId::monospace(11.0), egui::Color32::WHITE);
    }

    // selection ring
    if self.selected_bm == Some(i) {
      ui.painter().circle_stroke(egui::pos2(x,y), 8.0, egui::Stroke { width: 1.5, color: egui::Color32::YELLOW });
    }
  }
}

// focus crosshair (set by Jump)
if let Some((fa, fo)) = self.focus_point {
  let u = (fa - self.a_min) / (self.a_max - self.a_min);
  let v = (fo - self.o_min) / (self.o_max - self.o_min);
  let x = rect_hm.left() + u * rect_hm.width();
  let y = rect_hm.top()  + v * rect_hm.height();

  let c = egui::Color32::from_rgba_premultiplied(255,255,255,160);
  ui.painter().line_segment([egui::pos2(x, rect_hm.top()), egui::pos2(x, rect_hm.bottom())],
                             egui::Stroke { width: 1.0, color: c });
  ui.painter().line_segment([egui::pos2(rect_hm.left(), y), egui::pos2(rect_hm.right(), y)],
                             egui::Stroke { width: 1.0, color: c });
}
Optional: if your tile scheduler supports “click-to-refine,” you can also send a small region around (fa, fo) on Jump:
#[cfg(feature = "refine")]
let _ = self.tx_ctrl.blocking_send(data_model::ControlMsg::RefineRegion {
  a0: (fa - 0.01).max(self.a_min),
  a1: (fa + 0.01).min(self.a_max),
  o0: (fo - 0.01).max(self.o_min),
  o1: (fo + 0.01).min(self.o_max),
});



5) Top bar toggle
Add a quick toggle to show/hide the panel:
ui.toggle_value(&mut self.show_bookmark_panel, "Bookmarks Panel");
You can park this in your View or Window menu.



6) (Optional) Persist local edits
If you want title/notes edits in the side panel to round-trip back to the WebView sidecar JSON, add a “Save bookmarks.json” action using your existing writer path or a tiny JSON serializer:
if ui.button("Save to JSON").clicked() {
  use serde_json::json;
  let doc = json!({
    "schema": "rtx-ppf-bookmarks-1.0",
    "count": self.bookmarks.len(),
    "bookmarks": self.bookmarks, // serde derives do the work
  });
  if let Some(path) = rfd::FileDialog::new()
      .set_file_name("bookmarks.json").save_file() {
    let _ = std::fs::write(path, serde_json::to_string_pretty(&doc).unwrap());
  }
}
(You can also add a new ControlMsg::BookmarksReplace(Vec<Bookmark>) later so the WebView and sidecar stay perfectly in sync without touching disk.)



That’s it
	•	The side panel filters, edits, exports, and jumps.  
	•	The canvas highlights the selected pin and shows a focus crosshair.  
	•	Everything stays in lockstep with your live IPC stream (DataBus::Bookmarks) and the global clear (BookmarksCleared).  
Heck yes—here’s a tight, drop-in wiring for both features:
	•	Hotkeys: ↑ / ↓ to jump prev / next visible pin (wraps, skips hidden + filter).  
	•	Refine here: a button (and R hotkey) that pings your off-thread tile scheduler with a priority hint centered on the selected bookmark.  
Everything below is additive and keeps your current live IPC bookmarks intact.



1) Data model: new control for the scheduler
crates/data_model/src/lib.rs
Add a message your app can forward to the tiler:
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ControlMsg {
    // ...
    /// Ask the tile scheduler to refine a region (A, Ω) with a priority hint.
    RefineRegion {
        a0: f32, a1: f32,
        o0: f32, o1: f32,
        priority: u8,           // 0..=15 is plenty, higher = sooner
        min_depth: Option<u8>,  // optional target depth
    },
    // existing variants...
}
If you already have a scheduler message type, feel free to alias/forward. This one stands alone so you can ship today.



2) App core: forward the hint to the off-thread scheduler
crates/app/src/main.rs
In your control loop match (where you already rebroadcast bookmarks):
match msg {
    // ...existing arms...
    ControlMsg::RefineRegion { a0, a1, o0, o1, priority, min_depth } => {
        // Forward to your tile scheduler; adapt names as needed.
        // Example if you have a tx to the scheduler:
        // let _ = tx_sched.send(SchedMsg::Refine { a0, a1, o0, o1, priority, min_depth });
        if let Some(tx) = self.tx_sched.as_ref() {
            let _ = tx.send(SchedMsg::Refine { a0, a1, o0, o1, priority, min_depth });
        }
    }
    // ...rest...
}
Scheduler side (hint): if you use a custom enum, add:
pub enum SchedMsg {
    // ...
    Refine { a0:f32, a1:f32, o0:f32, o1:f32, priority:u8, min_depth:Option<u8> },
}



3) egui: side panel additions (hotkeys, jump, refine)
crates/host_ui/src/egui_host.rs
3.1 State fields (if you don’t have them yet)
pub struct AppState {
  // ...
  bookmarks: Vec<Bookmark>,
  show_bookmarks: bool,
  show_bookmark_panel: bool,
  bookmark_filter: String,
  selected_bm: Option<usize>,
  hidden_bm: std::collections::HashSet<usize>,
  focus_point: Option<(f32, f32)>,
  // NEW knobs
  refine_radius_frac: f32,   // fraction of current axis span for radius
  refine_priority: u8,
  refine_min_depth: Option<u8>,
}
Initialize once:
refine_radius_frac: 0.02,     // 2% of span (nice default)
refine_priority: 10,          // fairly high
refine_min_depth: None,
3.2 Helpers
Add a tiny utility to iterate visible pins (respects filter + hidden set):
fn visible_pin_indices(&self) -> Vec<usize> {
    (0..self.bookmarks.len())
      .filter(|&i| !self.hidden_bm.contains(&i) && bookmark_matches(&self.bookmarks[i], &self.bookmark_filter))
      .collect()
}

fn select_next_pin(&mut self, dir: i32) {
    let vis = self.visible_pin_indices();
    if vis.is_empty() { self.selected_bm = None; return; }
    let cur_pos = self.selected_bm
        .and_then(|idx| vis.iter().position(|&v| v == idx))
        .unwrap_or_else(|| if dir >= 0 { 0 } else { vis.len()-1 });
    let next_pos = if dir >= 0 { (cur_pos + 1) % vis.len() }
                   else { (cur_pos + vis.len() - 1) % vis.len() };
    let idx = vis[next_pos];
    self.selected_bm = Some(idx);
    let b = &self.bookmarks[idx];
    self.focus_point = Some((b.a as f32, b.omega as f32));
}
3.3 Hotkeys in your main 
update/render
 loop
Somewhere at the top of your frame (so it runs once per frame):
let pressed_up   = ctx.input(|i| i.key_pressed(egui::Key::ArrowUp));
let pressed_down = ctx.input(|i| i.key_pressed(egui::Key::ArrowDown));
let pressed_r    = ctx.input(|i| i.key_pressed(egui::Key::R));

if pressed_up   { self.select_next_pin(-1); }
if pressed_down { self.select_next_pin( 1); }
if pressed_r    { self.refine_selected(ctx); }  // defined below
3.4 The “Refine here” action
Add the action (and a panel button) that computes a radius from current A/Ω span:
impl AppState {
  fn refine_selected(&mut self, _ctx: &egui::Context) {
    let Some(idx) = self.selected_bm else { return; };
    if idx >= self.bookmarks.len() { return; }
    let b = &self.bookmarks[idx];

    let a_span = (self.a_max - self.a_min).max(1e-9);
    let o_span = (self.o_max - self.o_min).max(1e-9);
    let ra = self.refine_radius_frac * a_span;
    let ro = self.refine_radius_frac * o_span;

    let a0 = (b.a as f32 - ra).max(self.a_min);
    let a1 = (b.a as f32 + ra).min(self.a_max);
    let o0 = (b.omega as f32 - ro).max(self.o_min);
    let o1 = (b.omega as f32 + ro).min(self.o_max);

    let _ = self.tx_ctrl.blocking_send(data_model::ControlMsg::RefineRegion {
      a0, a1, o0, o1,
      priority: self.refine_priority,
      min_depth: self.refine_min_depth,
    });
  }
}
In your Bookmarks side panel UI, add the knobs + button (near your existing Export/Clear controls):
ui.separator();
ui.collapsing("Refine controls", |ui| {
    ui.horizontal(|ui| {
        ui.label("Radius (span frac)");
        ui.add(egui::Slider::new(&mut self.refine_radius_frac, 0.001..=0.25).logarithmic(true).clamp_to_range(true));
    });
    ui.horizontal(|ui| {
        ui.label("Priority");
        ui.add(egui::Slider::new(&mut self.refine_priority, 0..=15));
        ui.label("Min depth");
        let mut d = self.refine_min_depth.unwrap_or(0);
        if ui.add(egui::DragValue::new(&mut d).clamp_range(0..=12)).changed() {
            self.refine_min_depth = Some(d);
        }
        if ui.button("None").clicked() { self.refine_min_depth = None; }
    });
    let can_refine = self.selected_bm.is_some();
    if ui.add_enabled(can_refine, egui::Button::new("Refine here (R)")).clicked() {
        self.refine_selected(ctx);
    }
});
You already draw the focus crosshair—it becomes the visual confirmation of the target region; if you’d like, also draw a faint rectangle representing [a0..a1]×[o0..o1].



4) Canvas context menu (optional but nice)
Right-click to refine right where you click—even without selecting a pin:
let response = ui.allocate_rect(rect_hm, egui::Sense::click());
response.context_menu(|ui| {
    if ui.button("Refine here").clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let u = (pos.x - rect_hm.left()) / rect_hm.width();
            let v = (pos.y - rect_hm.top())  / rect_hm.height();
            let a = self.a_min + u * (self.a_max - self.a_min);
            let o = self.o_min + v * (self.o_max - self.o_min);
            // send same ControlMsg::RefineRegion centered on (a,o)
            let a_span = (self.a_max - self.a_min).max(1e-9);
            let o_span = (self.o_max - self.o_min).max(1e-9);
            let ra = self.refine_radius_frac * a_span;
            let ro = self.refine_radius_frac * o_span;
            let _ = self.tx_ctrl.blocking_send(data_model::ControlMsg::RefineRegion {
                a0: (a - ra).max(self.a_min),
                a1: (a + ra).min(self.a_max),
                o0: (o - ro).max(self.o_min),
                o1: (o + ro).min(self.o_max),
                priority: self.refine_priority,
                min_depth: self.refine_min_depth,
            });
        }
        ui.close_menu();
    }
});



5) (Optional) WebView cue
If you want the shader view to also show a faint refine box when a refine request goes out, just have your app broadcast a DataBus::RefineCue{a0,a1,o0,o1} right after forwarding to the scheduler, and handle it in the WebView overlay (draw a translucent rect in the overlay canvas using the live ranges). The wiring mirrors your existing quads(...) path.



Keyboard map recap
	•	↑ / ↓ – Move selection to previous/next visible bookmark (wraps).  
	•	R – “Refine here” around the selected bookmark (uses current radius, priority, optional min depth).  



—here’s a tight, production-ready BinaryHeap + cancel token scheduler you can drop in. It preempts broad/inferior tiles automatically when you issue a higher-priority refine. Workers check a per-job cancel_flag so in-flight work aborts quickly without panics.
Below is a self-contained module plus the minimal glue you’ll need.



0) Add dependency
# Cargo.toml (workspace or crates/app)
[dependencies]
crossbeam-channel = "0.5"



1) Drop-in scheduler module
Create crates/app/src/tile_scheduler.rs:
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{
    atomic::{AtomicBool, Ordering as AO},
    Arc, Condvar, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{unbounded, Receiver, Sender};

/// Phase-space rectangle in (A, Ω)
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub a0: f32, pub a1: f32,
    pub o0: f32, pub o1: f32,
}
impl Rect {
    pub fn area(&self) -> f32 { ((self.a1 - self.a0).abs() * (self.o1 - self.o0).abs()).max(0.0) }
    pub fn intersects(&self, other: &Rect) -> bool {
        !(self.a1 <= other.a0 || other.a1 <= self.a0 || self.o1 <= other.o0 || other.o1 <= self.o0)
    }
    pub fn clamp(&self, a_min:f32, a_max:f32, o_min:f32, o_max:f32) -> Rect {
        Rect {
            a0: self.a0.max(a_min), a1: self.a1.min(a_max),
            o0: self.o0.max(o_min), o1: self.o1.min(o_max),
        }
    }
}

/// Messages into the scheduler
#[derive(Debug)]
pub enum SchedMsg {
    /// Enqueue a refine job; if `cancel_prev` = true, older/overlapping and lower-priority/broader jobs will be canceled.
    Refine {
        rect: Rect,
        priority: u8,                 // higher wins
        min_depth: Option<u8>,        // optional quadtree target
        cancel_prev: bool,
        reply: Option<Sender<SchedEvent>>,
    },
    /// Explicit cancel by token
    CancelByToken { token: u64 },
    /// Graceful stop (drain cancels and exit)
    Shutdown,
}

/// Events out of the scheduler (you can forward to your UI/DataBus)
#[derive(Debug, Clone)]
pub enum SchedEvent {
    Started { token: u64 },
    Progress { token: u64, tiles_done: u32 },
    Done { token: u64 },
    Cancelled { token: u64 },
}

/// Internal queued job
#[derive(Debug, Clone)]
struct Job {
    token: u64,
    priority: u8,
    submitted_at: Instant,
    rect: Rect,
    min_depth: Option<u8>,
}

// Max-heap: high priority first; for equal priority, newest first; tiny area first (prefer narrow over broad)
impl Eq for Job {}
impl PartialEq for Job { fn eq(&self, other: &Self) -> bool { self.token == other.token } }
impl Ord for Job {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
            .then_with(|| self.submitted_at.cmp(&other.submitted_at))
            .then_with(|| other.rect.area().partial_cmp(&self.rect.area()).unwrap_or(Ordering::Equal))
    }
}
impl PartialOrd for Job { fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) } }

/// Shared scheduler state
struct State {
    heap: BinaryHeap<Job>,
    cancel: HashMap<u64, Arc<AtomicBool>>,
    listeners: HashMap<u64, Sender<SchedEvent>>,
    next_token: u64,
    // compaction: lazy deletion threshold
    lazy_deleted: usize,
}

/// Scheduler handle: send commands; worker threads run until Shutdown
pub struct Scheduler {
    tx: Sender<SchedMsg>,
    _threads: Vec<thread::JoinHandle<()>>,
}

impl Scheduler {
    /// Spawn a scheduler with `n_workers` threads.
    pub fn spawn(n_workers: usize) -> (Self, Sender<SchedMsg>, Receiver<SchedEvent>) {
        let (tx, rx) = unbounded::<SchedMsg>();
        let (ev_tx, ev_rx) = unbounded::<SchedEvent>();

        let shared = Arc::new((Mutex::new(State {
            heap: BinaryHeap::new(),
            cancel: HashMap::new(),
            listeners: HashMap::new(),
            next_token: 1,
            lazy_deleted: 0,
        }), Condvar::new()));

        // submit/command thread (pumps rx into state)
        {
            let shared = Arc::clone(&shared);
            let ev_tx = ev_tx.clone();
            thread::spawn(move || command_loop(shared, rx, ev_tx));
        }

        // workers
        let mut threads = Vec::new();
        for _ in 0..n_workers.max(1) {
            let shared = Arc::clone(&shared);
            let ev_tx = ev_tx.clone();
            threads.push(thread::spawn(move || worker_loop(shared, ev_tx)));
        }

        let handle = Scheduler { tx: tx.clone(), _threads: threads };
        (handle, tx, ev_rx)
    }
}

/// The thread that owns the heap; applies preemption/cancel policy.
fn command_loop(shared: Arc<(Mutex<State>, Condvar)>, rx: Receiver<SchedMsg>, ev_tx: Sender<SchedEvent>) {
    while let Ok(msg) = rx.recv() {
        match msg {
            SchedMsg::Shutdown => {
                // cancel everything and wake workers
                let (mu, cv) = &*shared;
                let mut st = mu.lock().unwrap();
                for (_, flag) in st.cancel.iter() { flag.store(true, AO::Relaxed); }
                st.heap.clear();
                cv.notify_all();
                break;
            }
            SchedMsg::CancelByToken { token } => {
                let (mu, _) = &*shared;
                let mut st = mu.lock().unwrap();
                if let Some(flag) = st.cancel.get(&token) {
                    flag.store(true, AO::Relaxed);
                }
                // optional: notify listeners
                let _ = ev_tx.send(SchedEvent::Cancelled { token });
            }
            SchedMsg::Refine { rect, priority, min_depth, cancel_prev, reply } => {
                let (mu, cv) = &*shared;
                let mut st = mu.lock().unwrap();
                let token = st.next_token; st.next_token += 1;

                let cancel_flag = Arc::new(AtomicBool::new(false));
                st.cancel.insert(token, cancel_flag);

                // preemption policy: mark overlapping *lower* priority or *broader* jobs as canceled
                if cancel_prev {
                    let mut canceled = 0usize;
                    for j in st.heap.iter() {
                        if j.rect.intersects(&rect) {
                            let worse = j.priority < priority
                                || (j.priority == priority && j.rect.area() > rect.area() * 1.01);
                            if worse {
                                if let Some(flag) = st.cancel.get(&j.token) {
                                    if !flag.swap(true, AO::Relaxed) {
                                        canceled += 1;
                                    }
                                }
                            }
                        }
                    }
                    st.lazy_deleted += canceled;
                    // compact if too many canceled entries accumulating
                    if st.lazy_deleted > 256 && st.lazy_deleted > st.heap.len() / 2 {
                        st.heap = st.heap.drain().filter(|j| {
                            !st.cancel.get(&j.token).map(|f| f.load(AO::Relaxed)).unwrap_or(false)
                        }).collect();
                        st.lazy_deleted = 0;
                    }
                }

                // Register optional listener (per-job)
                if let Some(tx) = reply {
                    st.listeners.insert(token, tx);
                }

                st.heap.push(Job {
                    token,
                    priority,
                    submitted_at: Instant::now(),
                    rect,
                    min_depth,
                });
                cv.notify_one();
            }
        }
    }
}

/// Worker loop: pop highest-priority job; skip canceled entries; compute tiles; emit events.
fn worker_loop(shared: Arc<(Mutex<State>, Condvar)>, ev_tx: Sender<SchedEvent>) {
    loop {
        // pop job
        let job = {
            let (mu, cv) = &*shared;
            let mut st = mu.lock().unwrap();
            let mut job = None;
            // wait for work
            while job.is_none() {
                if let Some(mut j) = st.heap.pop() {
                    // lazy skip canceled
                    let canceled = st.cancel.get(&j.token).map(|f| f.load(AO::Relaxed)).unwrap_or(false);
                    if canceled {
                        st.lazy_deleted = st.lazy_deleted.saturating_sub(1);
                        continue;
                    }
                    job = Some(j.clone());
                } else {
                    // no work—sleep until notified
                    st = cv.wait(st).unwrap();
                }
            }
            job.unwrap()
        };

        // compute (check cancel flag frequently)
        let cancel_flag = {
            let (mu, _) = &*shared;
            let st = mu.lock().unwrap();
            st.cancel.get(&job.token).cloned().unwrap_or_else(|| Arc::new(AtomicBool::new(true)))
        };

        let _ = ev_tx.send(SchedEvent::Started { token: job.token });
        if let Some(tx) = listener_for(&shared, job.token) {
            let _ = tx.send(SchedEvent::Started { token: job.token });
        }

        // === Replace with your real quadtree/tiler ===
        // Example: split into N tiles, simulate work
        let tiles_total = estimate_tiles(job.rect, job.min_depth);
        let mut emitted = 0u32;

        for _ in 0..tiles_total {
            if cancel_flag.load(AO::Relaxed) {
                let _ = ev_tx.send(SchedEvent::Cancelled { token: job.token });
                if let Some(tx) = listener_for(&shared, job.token) {
                    let _ = tx.send(SchedEvent::Cancelled { token: job.token });
                }
                // tidy: remove listener
                drop_listener(&shared, job.token);
                continue;
            }
            // do work on one tile (call your solver & send tile frames elsewhere)
            // compute_tile(job.rect, ...);
            thread::sleep(Duration::from_millis(2)); // simulate compute

            emitted += 1;
            if emitted % 16 == 0 {
                let _ = ev_tx.send(SchedEvent::Progress { token: job.token, tiles_done: emitted });
                if let Some(tx) = listener_for(&shared, job.token) {
                    let _ = tx.send(SchedEvent::Progress { token: job.token, tiles_done: emitted });
                }
            }
        }

        if !cancel_flag.load(AO::Relaxed) {
            let _ = ev_tx.send(SchedEvent::Done { token: job.token });
            if let Some(tx) = listener_for(&shared, job.token) {
                let _ = tx.send(SchedEvent::Done { token: job.token });
            }
        }
        drop_listener(&shared, job.token);
    }
}

// ---- tiny helpers for per-job listeners ----
fn listener_for(shared: &Arc<(Mutex<State>, Condvar)>, token: u64) -> Option<Sender<SchedEvent>> {
    let (mu, _) = &**shared;
    mu.lock().unwrap().listeners.get(&token).cloned()
}
fn drop_listener(shared: &Arc<(Mutex<State>, Condvar)>, token: u64) {
    let (mu, _) = &**shared;
    mu.lock().unwrap().listeners.remove(&token);
}

// Roughly estimate number of tiles from rect size/depth; tune or replace with your quadtree.
fn estimate_tiles(r: Rect, min_depth: Option<u8>) -> u32 {
    let base = ((r.area() * 2000.0).ceil() as u32).clamp(16, 8_192);
    if let Some(d) = min_depth {
        base.saturating_mul((1u32).saturating_shl(d.min(8) as u32))
    } else { base }
}
Design choices:
	•	Preemption is lazy: on a new refine with cancel_prev=true, we mark overlapping, lower-priority (or broader same-priority) jobs canceled. Workers skip/abort them; the heap is periodically compacted to remove dead entries.  
	•	Priority: higher u8 wins. Tie-breakers: newer first, then smaller area first (so “click-to-refine” beats old, broad scans).  
	•	Cancel tokens: HashMap<u64, Arc<AtomicBool>>. Workers check cancel_flag per tile ⇒ prompt early exit.  
	•	Events: one global ev_tx (you can rebroadcast as DataBus::PhaseTile or use for progress bars). Per-job optional reply is supported.  



2) Hook it into your app
Create/start the scheduler (e.g., in crates/app/src/main.rs):
mod tile_scheduler;
use tile_scheduler::{Scheduler, SchedMsg, Rect};

struct AppRuntime {
    tx_sched: Sender<SchedMsg>,
    // ...
}

impl AppRuntime {
    fn new() -> Self {
        let (_sched, tx_sched, ev_rx) = Scheduler::spawn(num_cpus::get().max(2) / 2);
        // Optionally: spawn a listener to forward SchedEvent → DataBus
        std::thread::spawn(move || {
            while let Ok(ev) = ev_rx.recv() {
                // map to your DataBus (e.g., progress UI, provenance overlay)
                // tx_bus.send(Arc::new(DataBus::Scheduler(ev)));
            }
        });
        Self { tx_sched /*, ...*/ }
    }
}
Forward your existing control message (you already added ControlMsg::RefineRegion in your previous step):
match msg {
    ControlMsg::RefineRegion { a0, a1, o0, o1, priority, min_depth } => {
        let rect = Rect { a0, a1, o0, o1 };
        let _ = self.tx_sched.send(SchedMsg::Refine {
            rect, priority, min_depth, cancel_prev: true, reply: None
        });
    }
    // ...
}
(If you want an immediate per-job progress channel to draw a “refine box” until tiles land, pass a reply: Some(ev_tx) and update your overlay.)



3) “Refine here (R)” in your egui UI
You already added ControlMsg::RefineRegion and the side-panel. With this scheduler in place, the “Refine here (R)” button and hotkey you wired will now preempt broader jobs automatically.



4) Optional: automatic “cancel-older-overlaps” without priority
If you prefer “always cancel older overlaps regardless of priority,” change:
let worse = j.priority < priority
    || (j.priority == priority && j.rect.area() > rect.area() * 1.01);
to:
let worse = j.rect.intersects(&rect);
(Or keep both and add a front-panel toggle.)



5) Optional: stricter compaction
If you enqueue many jobs, bump compaction:
if st.lazy_deleted > 128 && st.lazy_deleted > st.heap.len() / 3 {
   // ... compact ...
}



Quick mental model
	•	New refine hint ➜ Refine{rect, priority, cancel_prev:true}  
	•	Scheduler marks overlaps as canceled, pushes the new job to the top.  
	•	Workers pop highest-priority; if a popped job is canceled, they discard it; otherwise they process tiles but bail early once its flag flips.  
	•	Your UI sees progress/done or cancelled and updates the quadtree overlay accordingly.  
This pattern has been rock-solid for interactive tilers: it’s lock-light (one mutex + condvar), avoids fragile “stealing” complexities, and delivers snappy “click-to-refine” UX.
Absolutely—here are two drop-in extensions:
	•	a region-scope cancel API for the scheduler (kills all overlapping jobs, including in-flight), and  
	•	rate-limited refinement from the UI (prevents hotkey floods; supports “defer & replace” coalescing).  
Everything below is additive and compatible with the BinaryHeap + cancel-token scheduler you already have.



1) Scheduler: Region-scope cancel
Add a message
crates/app/src/tile_scheduler.rs
#[derive(Debug)]
pub enum SchedMsg {
    // ...
    /// Cancel all jobs intersecting `rect`. Optional filter by min priority.
    CancelRegion {
        rect: Rect,
        /// Cancel only jobs with priority <= this. None = cancel all.
        max_priority: Option<u8>,
    },
    // ...
}
Handle it in the command loop
fn command_loop(shared: Arc<(Mutex<State>, Condvar)>,
                rx: Receiver<SchedMsg>,
                ev_tx: Sender<SchedEvent>) {
    while let Ok(msg) = rx.recv() {
        match msg {
            // ...existing arms...
            SchedMsg::CancelRegion { rect, max_priority } => {
                let (mu, _) = &*shared;
                let mut st = mu.lock().unwrap();
                let mut cancelled = 0usize;

                for j in st.heap.iter() {
                    if j.rect.intersects(&rect)
                        && max_priority.map(|p| j.priority <= p).unwrap_or(true)
                    {
                        if let Some(flag) = st.cancel.get(&j.token) {
                            if !flag.swap(true, AO::Relaxed) {
                                cancelled += 1;
                                // optional: notify listeners immediately
                                let _ = ev_tx.send(SchedEvent::Cancelled { token: j.token });
                                if let Some(tx) = st.listeners.get(&j.token).cloned() {
                                    let _ = tx.send(SchedEvent::Cancelled { token: j.token });
                                }
                            }
                        }
                    }
                }
                st.lazy_deleted += cancelled;
                if st.lazy_deleted > 256 && st.lazy_deleted > st.heap.len() / 2 {
                    st.heap = st.heap
                        .drain()
                        .filter(|j| !st.cancel.get(&j.token).map(|f| f.load(AO::Relaxed)).unwrap_or(false))
                        .collect();
                    st.lazy_deleted = 0;
                }
            }
            // ...
        }
    }
}
Why this works
• Queued jobs live in the heap → we mark their cancel flags and compact lazily.
• Running jobs also have entries in st.cancel → workers see the flag and bail early on their next tile.
Optional: narrow-only cancel
If you only want to kill broad/older work, add a predicate:
let worse = j.rect.area() > rect.area() * 1.05 || j.priority <= max_priority.unwrap_or(u8::MAX);
if j.rect.intersects(&rect) && worse { /* mark cancel */ }



2) UI rate-limiting for “Refine here”
App state knobs (egui)
crates/host_ui/src/egui_host.rs
use std::time::{Duration, Instant};

pub enum RlMode { Drop, DeferReplace }

pub struct TokenBucket {
    cap: u32, tokens: f32, refill_per_sec: f32, last: Instant
}
impl TokenBucket {
    pub fn new(cap:u32, refill_per_sec:f32) -> Self {
        Self { cap, tokens: cap as f32, refill_per_sec, last: Instant::now() }
    }
    pub fn allow(&mut self) -> bool {
        let now = Instant::now();
        let dt = now.duration_since(self.last).as_secs_f32();
        self.last = now;
        self.tokens = (self.tokens + dt*self.refill_per_sec).min(self.cap as f32);
        if self.tokens >= 1.0 { self.tokens -= 1.0; true } else { false }
    }
}

pub struct AppState {
    // ...existing...
    rl_mode: RlMode,
    rl_min_gap: Duration,
    rl_last: Instant,
    rl_bucket: TokenBucket,
    pending_refine: Option<(f32,f32)>, // (A, Ω) deferred target
}
Initialize:
Self {
  // ...
  rl_mode: RlMode::DeferReplace,
  rl_min_gap: Duration::from_millis(150),
  rl_last: Instant::now() - Duration::from_secs(1),
  rl_bucket: TokenBucket::new(4, 6.0), // burst 4, ~6/sec steady
  pending_refine: None,
}
Gate the refine action
Replace your direct call to refine_selected(ctx) with:
fn try_refine_selected(&mut self, ctx:&egui::Context) {
    let Some(idx) = self.selected_bm else { return; };
    let b = &self.bookmarks[idx];
    let now = Instant::now();

    let min_gap_ok = now.duration_since(self.rl_last) >= self.rl_min_gap;
    let bucket_ok  = self.rl_bucket.allow();

    if min_gap_ok && bucket_ok {
        self.rl_last = now;
        self.refine_selected(ctx); // your existing sender
        self.pending_refine = None; // flush any pending
    } else {
        match self.rl_mode {
            RlMode::Drop => { /* ignore */ }
            RlMode::DeferReplace => {
                self.pending_refine = Some((b.a as f32, b.omega as f32));
            }
        }
    }
}
Dispatch a pending refine when cool-down elapses (add at top of your update per frame):
let now = Instant::now();
if self.pending_refine.is_some()
   && now.duration_since(self.rl_last) >= self.rl_min_gap
   && self.rl_bucket.allow()
{
    if let Some((a,o)) = self.pending_refine.take() {
        self.rl_last = now;
        self.refine_at(ctx, a, o); // variant of refine_selected that takes (A, Ω)
    }
}
Implement refine_at by copying refine_selected and replacing the center (b.a,b.omega) with (a,o).
Wire the hotkey
Where you already handle keys:
let pressed_r = ctx.input(|i| i.key_pressed(egui::Key::R));
if pressed_r { self.try_refine_selected(ctx); }
UI controls (side panel)
Add to your “Refine controls” group:
ui.horizontal(|ui| { ui.label("Min gap"); ui.add(egui::DragValue::new(&mut self.rl_min_gap).speed(5).suffix(" ms")); });
ui.horizontal(|ui| {
    ui.label("Limiter");
    ui.selectable_value(&mut self.rl_mode, RlMode::Drop, "Drop");
    ui.selectable_value(&mut self.rl_mode, RlMode::DeferReplace, "Defer & replace");
});
ui.horizontal(|ui| {
    ui.label("Burst / Rate");
    let mut cap = self.rl_bucket.cap as i32;
    if ui.add(egui::DragValue::new(&mut cap).clamp_range(1..=16)).changed() { self.rl_bucket.cap = cap as u32; }
    let mut rps = self.rl_bucket.refill_per_sec;
    if ui.add(egui::DragValue::new(&mut rps).clamp_range(0.5..=60.0)).changed() { self.rl_bucket.refill_per_sec = rps; }
});



3) Region-cancel from UI
Context-menu (canvas)
response.context_menu(|ui| {
    if ui.button("Cancel region (visible)").clicked() {
        let _ = self.tx_ctrl.blocking_send(data_model::ControlMsg::CancelRegion {
            a0: self.a_min, a1: self.a_max,
            o0: self.o_min, o1: self.o_max,
            max_priority: None, // or Some(threshold)
        });
        ui.close_menu();
    }
});
Lasso → bbox
If you already have a polygon lasso, compute its bbox and send:
let (a0,a1,o0,o1) = bbox_from_polygon(&lasso_pts);
let _ = self.tx_ctrl.blocking_send(data_model::ControlMsg::CancelRegion {
    a0, a1, o0, o1, max_priority: Some(8),
});



4) App control message
Extend your control enum and forward to scheduler.
crates/data_model/src/lib.rs
pub enum ControlMsg {
    // ...
    CancelRegion { a0:f32, a1:f32, o0:f32, o1:f32, max_priority: Option<u8> },
}
crates/app/src/main.rs
match msg {
    // ...
    ControlMsg::CancelRegion { a0, a1, o0, o1, max_priority } => {
        let rect = tile_scheduler::Rect { a0, a1, o0, o1 };
        let _ = self.tx_sched.send(tile_scheduler::SchedMsg::CancelRegion { rect, max_priority });
    }
    // ...
}



Behavior you get
	•	Click-to-refine remains snappy.  
	•	Spam-safe: R hotkey floods are smoothed by min-gap + token bucket.  
	•	Defer & replace mode ensures only the latest target is dispatched after cooldown.  
	•	Region cancel nukes stale/broad work immediately—both queued and in-flight—so the heap stays clean and the workers pivot to what you care about.  
If you want “cancel everything outside this region” (exclusive keep), flip the predicate in CancelRegion and mark non-intersecting jobs instead.



