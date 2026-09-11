"""Detect which side of the capture box the fish is on.

For now this only answers: is the fish to the LEFT or RIGHT of the capture box?

Usage:
    py fish_side.py                     # run over every img/*.png sample
    py fish_side.py path/to/frame.png   # run on specific image(s)
    py fish_side.py --screen            # grab the live region once and report
    py fish_side.py --live [interval]   # print the fish side in a loop (no input)
    py fish_side.py --play              # steer the box; logs to logs/play-*.csv
        options: --tick S --lead S --deadzone PX --smooth F --secs S --log PATH
    py fish_side.py --analyze logs/*.csv   # summarise play logs for calibration
"""

from __future__ import annotations

import csv
import glob
import os
import sys
import time
from datetime import datetime

import pydirectinput

import numpy as np
from PIL import Image

import config

pydirectinput.PAUSE = 0.0  # we manage our own timing in the control loop

# --- config (edit config.json, not here) -------------------------------
_C = config.load("fish_side")
REGION = tuple(_C["region"])    # minigame bar: left, top, width, height
DEADZONE = _C["deadzone"]       # hysteresis half-width around box centre (px)
LEAD = _C["lead"]               # seconds of momentum look-ahead
SMOOTH = _C["smooth"]           # EMA factor for the box position/velocity estimate
MAX_JUMP = _C["max_jump"]       # px jump from prediction that counts as a glitch
COAST_MAX = _C["coast_max"]     # frames to coast through glitch readings
V_CLAMP = _C["v_clamp"]         # box velocity hard cap (px/s)
BOX_W = tuple(_C["box_w"])      # plausible dark-box width range (px)
BAND = slice(*_C["band"])       # rows holding the fish + capture box
EDGE = _C["edge"]               # ignore dark runs within this many px of the border

# --- screen capture -------------------------------------------------------
# ImageGrab.grab is ~55 ms/frame on Windows, which caps the control loop at
# ~12 Hz. mss is ~3x faster; fall back to ImageGrab if it isn't installed.
try:
    import mss

    _SCT = mss.MSS()
    _MON = {"left": REGION[0], "top": REGION[1],
            "width": REGION[2], "height": REGION[3]}
    GRABBER = "mss"

    def _grab_rgb() -> np.ndarray:
        return np.asarray(_SCT.grab(_MON))[:, :, 2::-1]  # BGRA -> RGB
except Exception:  # pragma: no cover - env without mss
    from PIL import ImageGrab

    GRABBER = "imagegrab"

    def _grab_rgb() -> np.ndarray:
        left, top, width, height = REGION
        box = (left, top, left + width, top + height)
        return np.asarray(ImageGrab.grab(bbox=box).convert("RGB"))

class Controller:
    """PD steering of the capture box.

    Hold left mouse -> box accelerates right; release -> it falls left.
    The button is held as *state* across frames: mouseDown fires only on a
    left->right switch, mouseUp only on right->left, so a steady push stays
    steady instead of stuttering.

    Because the box has momentum, plain "hold until centred" overshoots every
    time. Instead we track the box's velocity and act on where it will be
    LEAD seconds from now, releasing early so it coasts onto the fish.
    """

    def __init__(
        self, deadzone: float = DEADZONE, lead: float = LEAD, smooth: float = SMOOTH
    ):
        self.deadzone = deadzone
        self.lead = lead
        self.smooth = smooth
        self.holding = False
        self._box_x: float | None = None
        self._box_v = 0.0
        self._t: float | None = None
        self._coast = 0
        # Last-tick values, exposed for logging / calibration.
        self.dt = 0.0
        self.err = 0.0
        self.signal = 0.0
        self.rejected = False

    def _set(self, hold: bool) -> None:
        if hold and not self.holding:
            pydirectinput.mouseDown(button="left")
            self.holding = True
        elif not hold and self.holding:
            pydirectinput.mouseUp(button="left")
            self.holding = False

    def update(self, res: dict) -> None:
        now = time.perf_counter()
        self.dt = 0.0 if self._t is None else now - self._t
        self.rejected = False
        if not res["ok"]:
            self._set(False)  # lost the fish -> stop pushing
            self._box_x = None
            self._t = None
            self._coast = 0
            return

        bx = res["box_x"]
        if self._box_x is None or self._t is None:
            self._box_x, self._box_v = bx, 0.0
            self._coast = 0
        else:
            pred = self._box_x + self._box_v * self.dt
            if abs(bx - pred) > MAX_JUMP and self._coast < COAST_MAX:
                # Phantom / glitch reading: coast on the model this frame.
                self.rejected = True
                self._coast += 1
                self._box_x = pred
                self._box_v *= 0.5
            else:
                self._coast = 0
                if self.dt > 1e-4:
                    raw_v = (bx - self._box_x) / self.dt
                    self._box_v += self.smooth * (raw_v - self._box_v)
                self._box_x += self.smooth * (bx - self._box_x)
        self._box_v = max(-V_CLAMP, min(V_CLAMP, self._box_v))
        self._t = now

        # Predicted error LEAD seconds ahead: current gap minus how far the
        # box will drift on its own in that time.
        self.err = res["fish_x"] - self._box_x
        self.signal = self.err - self.lead * self._box_v
        if self.signal > self.deadzone:
            self._set(True)
        elif self.signal < -self.deadzone:
            self._set(False)
        # within the deadzone: keep the current button state

    def stop(self) -> None:
        self._set(False)


def _runs(mask1d: np.ndarray):
    """Return [(start, end, length), ...] for contiguous True runs."""
    idx = np.where(mask1d)[0]
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    return [(g[0], g[-1], g.size) for g in np.split(idx, breaks + 1)]


def _fish(band: np.ndarray):
    """(center_x, state) for the fish blob, or (None, None)."""
    r, g, b = band[..., 0], band[..., 1], band[..., 2]
    green = (g > 110) & (r < g - 25) & (b < g - 15)
    red = (r > 110) & (g < r - 35) & (b < r - 35)

    best = None
    for state, mask in (("gain", green), ("loss", red)):
        cols = mask.sum(axis=0) > 2
        for start, end, length in _runs(cols):
            if length >= 25 and (best is None or length > best[0]):
                best = (length, (start + end) / 2.0, state)
    if best is None:
        return None, None
    return best[1], best[2]


def _box(band: np.ndarray, prev: float | None = None):
    """(center_x, lo, hi) of the dark capture box, or (None, None, None).

    Keeps only dark runs whose width is plausible for the box, merges the two
    halves when the fish splits it, then picks the candidate nearest ``prev``
    (or the widest one if there's no prior estimate) so a phantom blob at the
    widget edge can't hijack the reading.
    """
    r, g, b = band[..., 0], band[..., 1], band[..., 2]
    dark = (r < 55) & (g < 55) & (b < 70)
    cols = dark.sum(axis=0) > 15
    w = band.shape[1]

    raw = [
        (start, end)
        for start, end, length in _runs(cols)
        if not (end < EDGE or start > w - EDGE)
    ]
    if not raw:
        return None, None, None

    # Merge runs separated only by a narrow gap (fish sitting on the box).
    merged = [list(raw[0])]
    for start, end in raw[1:]:
        if start - merged[-1][1] <= 45:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    cands = [(lo, hi) for lo, hi in merged if BOX_W[0] <= hi - lo <= BOX_W[1]]
    if not cands:
        return None, None, None

    if prev is not None:
        lo, hi = min(cands, key=lambda s: abs((s[0] + s[1]) / 2.0 - prev))
    else:
        lo, hi = max(cands, key=lambda s: s[1] - s[0])
    return (lo + hi) / 2.0, lo, hi


def detect(img, prev_box: float | None = None) -> dict:
    """Return {'ok', 'side', 'state', 'fish_x', 'box_x', 'box_w'} for one frame.

    ``img`` may be a PIL image or an RGB ndarray. ``prev_box`` is an optional
    hint (last known box centre) used to reject phantom detections.
    """
    arr = img if isinstance(img, np.ndarray) else np.asarray(img.convert("RGB"))
    band = arr.astype(int)[BAND]
    fish_x, state = _fish(band)
    box_x, lo, hi = _box(band, prev_box)
    if fish_x is None or box_x is None:
        return {
            "ok": False,
            "side": None,
            "state": state,
            "fish_x": fish_x,
            "box_x": box_x,
            "box_w": None,
        }
    return {
        "ok": True,
        "side": "left" if fish_x < box_x else "right",
        "state": state,
        "fish_x": round(fish_x, 1),
        "box_x": round(box_x, 1),
        "box_w": round(hi - lo, 1),
    }


def from_screen(prev_box: float | None = None) -> dict:
    return detect(_grab_rgb(), prev_box)


def _status_line(res: dict, extra: str = "") -> str:
    if res["ok"]:
        return (
            f"fish {res['side']:5s}  "
            f"fish_x={res['fish_x']:<6} box_x={res['box_x']:<6} "
            f"{res['state']}  {extra}"
        )
    return f"no detection  (fish_x={res['fish_x']}, box_x={res['box_x']})  {extra}"


def live(interval: float = 0.2) -> None:
    """Continuously read the screen region and print the fish side. No input."""
    print(f"Reading region {REGION} every {interval:g}s. Ctrl+C to stop.")
    try:
        while True:
            res = from_screen()
            print(f"\r{_status_line(res):<74}", end="", flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped.")


LOG_COLUMNS = [
    "t", "dt", "ok", "state", "side",
    "fish_x", "box_raw", "box_w", "box_x", "box_v", "err", "signal", "hold", "rej",
]


def play(
    tick: float = 0.0,
    lead: float = LEAD,
    deadzone: float = DEADZONE,
    smooth: float = SMOOTH,
    secs: float | None = None,
    log: str | None = None,
) -> str | None:
    """Steer the capture box to keep it on the fish.

    Writes a per-tick CSV to ``log`` (default ``logs/play-<timestamp>.csv``) for
    calibration. Stops on Ctrl+C, or after ``secs`` seconds if given. Returns the
    log path.
    """
    if log is None:
        os.makedirs("logs", exist_ok=True)
        log = os.path.join(
            "logs", f"play-{datetime.now():%Y%m%d_%H%M%S}.csv"
        )
    ctrl = Controller(deadzone=deadzone, lead=lead, smooth=smooth)
    print(
        f"Playing region {REGION}  tick={tick:g} lead={lead:g} "
        f"deadzone={deadzone:g} smooth={smooth:g}\n"
        f"logging -> {log}   Ctrl+C to stop"
        + (f" (auto-stop {secs:g}s)" if secs else "")
    )
    start = time.perf_counter()
    n_ok = n_gain = rows = 0
    with open(log, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            f"# grabber={GRABBER} tick={tick} lead={lead} "
            f"deadzone={deadzone} smooth={smooth} max_jump={MAX_JUMP}"
        ])
        w.writerow(LOG_COLUMNS)
        try:
            while True:
                t = time.perf_counter() - start
                res = from_screen(ctrl._box_x)
                ctrl.update(res)
                w.writerow([
                    f"{t:.4f}", f"{ctrl.dt:.4f}", int(res["ok"]),
                    res["state"] or "", res["side"] or "",
                    "" if res["fish_x"] is None else res["fish_x"],
                    "" if res["box_x"] is None else res["box_x"],
                    "" if res["box_w"] is None else res["box_w"],
                    "" if ctrl._box_x is None else f"{ctrl._box_x:.1f}",
                    f"{ctrl._box_v:.1f}", f"{ctrl.err:.1f}",
                    f"{ctrl.signal:.1f}", int(ctrl.holding), int(ctrl.rejected),
                ])
                rows += 1
                n_ok += res["ok"]
                n_gain += res["state"] == "gain"
                bx = "--" if ctrl._box_x is None else f"{ctrl._box_x:.0f}"
                tag = (
                    f"box~{bx:<4} v={ctrl._box_v:+6.0f} err={ctrl.err:+6.0f} "
                    f"{'HOLD' if ctrl.holding else '    '}"
                    f"{' REJ' if ctrl.rejected else ''}"
                )
                fx = res["fish_x"]
                head = f"fish {res['side'] or '--':5s} @{fx if fx is not None else '--':<6}"
                print(f"\r{head}  {tag}   {res['state'] or '':4s}"[:90].ljust(90),
                      end="", flush=True)
                if secs and t >= secs:
                    break
                time.sleep(tick)
        except KeyboardInterrupt:
            pass
        finally:
            ctrl.stop()
    gain_pct = 100 * n_gain / rows if rows else 0
    ok_pct = 100 * n_ok / rows if rows else 0
    print(
        f"\nstopped. {rows} ticks, {ok_pct:.0f}% detected, "
        f"{gain_pct:.0f}% in GAIN.  log: {log}"
    )
    return log


def analyze(paths: list[str]) -> None:
    """Summarise one or more play logs for calibration."""
    for path in paths:
        rows = []
        with open(path, newline="") as fh:
            for row in csv.DictReader(
                (ln for ln in fh if not ln.startswith("#"))
            ):
                rows.append(row)
        if not rows:
            print(f"{path}: empty")
            continue
        hdr = ""
        with open(path) as fh:
            first = fh.readline().strip()
            if first.startswith("#"):
                hdr = first.lstrip("# ")
        n = len(rows)
        det = [r for r in rows if r["ok"] == "1"]
        gain = sum(r["state"] == "gain" for r in rows)
        hold = sum(r["hold"] == "1" for r in det)
        rej = sum(r.get("rej") == "1" for r in rows)
        dts = [float(r["dt"]) for r in rows[1:] if float(r["dt"]) > 0]
        errs = [abs(float(r["err"])) for r in det if r["err"] != ""]
        # Overshoot: signed err right after each HOLD->release edge.
        overs = []
        for a, b in zip(det, det[1:]):
            if a["hold"] == "1" and b["hold"] == "0":
                e = float(b["err"])
                overs.append(-e if e < 0 else 0.0)  # box past fish => err<0
        mean_err = sum(errs) / len(errs) if errs else 0
        p90 = sorted(errs)[int(0.9 * len(errs))] if errs else 0
        mean_dt = sum(dts) / len(dts) if dts else 0
        print(f"\n{path}")
        if hdr:
            print(f"  cfg: {hdr}")
        print(f"  ticks={n}  detected={100*len(det)/n:.0f}%  "
              f"GAIN={100*gain/n:.0f}%  hold-duty={100*hold/max(len(det),1):.0f}%")
        print(f"  loop: mean dt={1000*mean_dt:.0f}ms ({1/mean_dt:.0f} Hz)  "
              f"rejected frames={rej} ({100*rej/n:.0f}%)")
        print(f"  |err|: mean={mean_err:.1f}px  p90={p90:.1f}px  "
              f"max={max(errs) if errs else 0:.1f}px")
        if overs:
            print(f"  overshoot on release: mean={sum(overs)/len(overs):.1f}px  "
                  f"max={max(overs):.1f}px  (n={len(overs)})")


def _opt(argv: list[str], name: str, cast, default):
    """Read `--name value` from argv, else return default."""
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return cast(argv[i + 1])
    return default


def _main(argv: list[str]) -> None:
    if "--live" in argv:
        rest = [a for a in argv if a != "--live" and not a.startswith("-")]
        live(float(rest[0]) if rest else 0.2)
        return
    if "--analyze" in argv:
        i = argv.index("--analyze")
        paths = [a for a in argv[i + 1:] if not a.startswith("-")]
        analyze(paths or sorted(glob.glob("logs/*.csv")))
        return
    if "--play" in argv:
        play(
            tick=_opt(argv, "--tick", float, 0.0),
            lead=_opt(argv, "--lead", float, LEAD),
            deadzone=_opt(argv, "--deadzone", float, DEADZONE),
            smooth=_opt(argv, "--smooth", float, SMOOTH),
            secs=_opt(argv, "--secs", float, None),
            log=_opt(argv, "--log", str, None),
        )
        return
    if "--screen" in argv:
        print(from_screen())
        return
    paths = [a for a in argv if not a.startswith("-")] or sorted(glob.glob("img/*.png"))
    for path in paths:
        res = detect(Image.open(path))
        if res["ok"]:
            print(
                f"{path}: fish is {res['side']:5s}  "
                f"(fish_x={res['fish_x']}, box_x={res['box_x']}, {res['state']})"
            )
        else:
            print(f"{path}: no detection  ({res})")


if __name__ == "__main__":
    _main(sys.argv[1:])