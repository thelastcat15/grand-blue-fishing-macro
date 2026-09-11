"""
ตรรกะบอทตกปลา 3 ขั้นตอน
แยกจาก GUI เพื่อรันใน background thread และหยุดได้ทันที
"""

import time
import threading
import numpy as np
import cv2
import mss
from random import randint

import input_backend as mouse
from config import save_config


# ============================================================
# SCREEN CAPTURE
# ============================================================

sct = mss.mss()


def grab(region):
    """
    region = (x, y, width, height)
    """
    x, y, w, h = region

    img = np.array(
        sct.grab({
            "left": x,
            "top": y,
            "width": w,
            "height": h
        })
    )

    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


# ============================================================
# TEMPLATE
# ============================================================

def find_template(region, template_path, threshold):
    frame = grab(region)

    template = cv2.imread(
        template_path,
        cv2.IMREAD_COLOR
    )

    if template is None:
        return None

    res = cv2.matchTemplate(
        frame,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val >= threshold:

        th, tw = template.shape[:2]

        cx = (
            region[0]
            + max_loc[0]
            + tw // 2
        )

        cy = (
            region[1]
            + max_loc[1]
            + th // 2
        )

        return (
            cx,
            cy,
            max_val
        )

    return None


# ============================================================
# CAST BAR
# ============================================================

def cast_fill_ratio(region):

    frame = grab(region)

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    mask = cv2.inRange(
        hsv,
        np.array([90, 60, 60]),
        np.array([130, 255, 255])
    )

    filled_rows = np.any(
        mask,
        axis=1
    )

    return (
        filled_rows.sum()
        / mask.shape[0]
    )


# ============================================================
# FISH COLOR MASK
# ============================================================

def create_fish_mask(frame):
    """
    สร้าง mask สำหรับปลา
    รองรับ:
        - สีเขียว
        - สีแดง
    """

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # GREEN
    # --------------------------------------------------------

    green_mask = cv2.inRange(
        hsv,
        np.array([45, 80, 80]),
        np.array([75, 255, 255])
    )

    # --------------------------------------------------------
    # RED
    # --------------------------------------------------------

    red_mask1 = cv2.inRange(
        hsv,
        np.array([0, 80, 80]),
        np.array([10, 255, 255])
    )

    red_mask2 = cv2.inRange(
        hsv,
        np.array([170, 80, 80]),
        np.array([179, 255, 255])
    )

    red_mask = cv2.bitwise_or(
        red_mask1,
        red_mask2
    )

    # --------------------------------------------------------
    # GREEN OR RED
    # --------------------------------------------------------

    fish_mask = cv2.bitwise_or(
        green_mask,
        red_mask
    )

    # --------------------------------------------------------
    # REMOVE SMALL NOISE
    # --------------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    fish_mask = cv2.morphologyEx(
        fish_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    fish_mask = cv2.morphologyEx(
        fish_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return fish_mask


# ============================================================
# FIND FISH
# ============================================================

def _runs(mask):
    """คืนช่วง True ที่ต่อเนื่องกันใน mask หนึ่งมิติ"""
    indexes = np.where(mask)[0]
    if indexes.size == 0:
        return []

    breaks = np.where(np.diff(indexes) > 1)[0]
    groups = np.split(indexes, breaks + 1)
    return [
        (group[0], group[-1], len(group))
        for group in groups
    ]


def find_fish_x(frame, prev_fish_x=None):
    """หา center X ของปลาโดยนับช่วงคอลัมน์สีเขียว/แดง"""
    blue, green, red = cv2.split(frame)
    masks = (
        (green > 110) & (blue < green - 25) & (red < green - 15),
        (red > 110) & (green < red - 35) & (blue < red - 35),
    )
    candidates = []

    for mask in masks:
        columns = mask.sum(axis=0) > 2
        for start, end, length in _runs(columns):
            if length >= 25:
                candidates.append((start, end, length))

    if not candidates:
        return None

    if prev_fish_x is not None:
        nearest = min(
            candidates,
            key=lambda item: abs((item[0] + item[1]) / 2 - prev_fish_x)
        )
        if abs((nearest[0] + nearest[1]) / 2 - prev_fish_x) <= 120:
            return int(round((nearest[0] + nearest[1]) / 2))

    best = max(candidates, key=lambda item: item[2])
    return int(round((best[0] + best[1]) / 2))

# ============================================================
# FIND BLACK REEL BAR
# ============================================================

def _find_bar_candidates(frame, min_width=30, max_width=None, edge=3):
    """หาช่วงกรอบดำจาก dark runs ในแถบแนวนอน"""
    _, green, red = cv2.split(frame)
    blue = frame[:, :, 0]
    height, width = frame.shape[:2]
    band = slice(int(height * 0.10), int(height * 0.90))
    dark = (red[band] < 55) & (green[band] < 55) & (blue[band] < 70)
    columns = dark.sum(axis=0) > 15
    if max_width is None:
        max_width = width * 0.85

    raw = [
        (start, end)
        for start, end, _ in _runs(columns)
        if not (end < edge or start > width - edge)
    ]
    if not raw:
        return []

    merged = [list(raw[0])]
    for start, end in raw[1:]:
        if start - merged[-1][1] <= 45:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return [
        {
            "start": start,
            "end": end,
            "width": end - start + 1,
        }
        for start, end in merged
        if min_width <= end - start + 1 <= max_width
    ]


def _select_bar_candidate(
    frame,
    expected_width=None,
    min_width=30,
    max_width=None,
    edge=3,
):
    candidates = _find_bar_candidates(frame, min_width, max_width, edge)
    if not candidates:
        return None

    if expected_width is not None:
        matching = [
            candidate for candidate in candidates
            if abs(candidate["width"] - expected_width)
            <= max(expected_width * 0.45, 12)
        ]
        if matching:
            candidates = matching

    return max(
        candidates,
        key=lambda candidate: candidate["width"]
    )


def find_bar_x(
    frame,
    expected_width=None,
    min_width=30,
    max_width=None,
    edge=3,
):
    """คืนค่า center X ของก้อนดำที่มีรูปทรงเป็นแถบแนวนอน"""
    candidate = _select_bar_candidate(
        frame,
        expected_width,
        min_width,
        max_width,
        edge,
    )
    if candidate is None:
        return None
    return int(round((candidate["start"] + candidate["end"]) / 2))


def find_bar_width(
    frame,
    expected_width=None,
    min_width=30,
    max_width=None,
    edge=3,
):
    """คืนค่าความกว้างแถบดำ เพื่อใช้ปรับตัวข้ามรอบการทำงาน"""
    candidate = _select_bar_candidate(
        frame,
        expected_width,
        min_width,
        max_width,
        edge,
    )
    if candidate is None:
        return None
    return candidate["width"]

# ============================================================
# FIND FISH + BAR
# ============================================================

def find_fish_and_bar_x(
    region,
    prev_fish_x=None,
    expected_bar_width=None,
    bar_width_min=30,
    bar_width_max=None,
    bar_edge=3,
):

    frame = grab(region)

    fish_x = find_fish_x(
        frame,
        prev_fish_x
    )

    bar_x = find_bar_x(
        frame,
        expected_bar_width,
        bar_width_min,
        bar_width_max,
        bar_edge,
    )

    bar_width = find_bar_width(
        frame,
        expected_bar_width,
        bar_width_min,
        bar_width_max,
        bar_edge,
    )

    return fish_x, bar_x, bar_width

# ============================================================
# REEL BAR VISIBLE
# ============================================================

def reel_bar_visible(
    region,
    expected_width=None,
    width_min=70,
    width_max=220,
    edge=3,
):
    """
    ใช้ตอน Phase 2 เท่านั้น

    ไม่ใช้ใน Phase 3
    เพราะ Phase 3 จะ detect พร้อม fish/bar
    """

    fish_x, bar_x, _ = find_fish_and_bar_x(
        region,
        expected_bar_width=expected_width,
        bar_width_min=width_min,
        bar_width_max=width_max,
        bar_edge=edge,
    )
    return fish_x is not None and bar_x is not None


# ============================================================
# FISH BOT
# ============================================================

class FishBot:
    """
    Bot ตกปลา 3 ขั้นตอน

    1. Cast
    2. Shake
    3. Reel

    ใช้ thread แยกจาก GUI
    """

    def __init__(
        self,
        config,
        log_fn
    ):

        self.cfg = config
        self.log = log_fn

        self.stop_event = (
            threading.Event()
        )

        self.thread = None
        self.learned_bar_width = self.cfg.get("reel_bar_width")
        self.bar_width_samples = 0

    # ========================================================
    # START
    # ========================================================

    def start(self):

        if (
            self.thread
            and self.thread.is_alive()
        ):
            return

        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._run,
            daemon=True
        )

        self.thread.start()

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        self.stop_event.set()

        # ป้องกันเมาส์ค้าง
        try:
            mouse.mouse_up()
        except Exception:
            pass

        self._save_learned_bar_width()

    def _save_learned_bar_width(self):
        if self.learned_bar_width is None or self.bar_width_samples < 5:
            return
        self.cfg["reel_bar_width"] = round(self.learned_bar_width, 1)
        try:
            save_config(self.cfg)
            self.log(
                f"    จำความกว้างแถบดำไว้ {self.learned_bar_width:.1f}px"
            )
        except OSError as exc:
            self.log(f"    บันทึกค่าที่เรียนรู้ไม่ได้: {exc}")

    # ========================================================
    # IS RUNNING
    # ========================================================

    def is_running(self):

        return (
            self.thread is not None
            and self.thread.is_alive()
        )

    # ========================================================
    # MAIN LOOP
    # ========================================================

    def _run(self):

        self.log(
            "เริ่มบอทใน 3 วินาที... "
            f"(มือควบคุมเมาส์: "
            f"{mouse.BACKEND_NAME})"
        )

        # ----------------------------------------------------
        # Countdown
        # ----------------------------------------------------

        for _ in range(30):

            if self.stop_event.is_set():
                return

            time.sleep(0.1)

        try:

            while not self.stop_event.is_set():

                # ==================================================
                # PHASE 1
                # ==================================================

                self._phase_1_cast()

                if self.stop_event.is_set():
                    break

                # ==================================================
                # PHASE 2
                # ==================================================

                found = self._phase_2_shake()

                if self.stop_event.is_set():
                    break

                # ==================================================
                # PHASE 3
                # ==================================================

                if found:

                    self._phase_3_reel()

                if self.stop_event.is_set():
                    break

                # ==================================================
                # COOLDOWN
                # ==================================================

                self._cooldown()

        finally:

            # ----------------------------------------------------
            # สำคัญมาก
            # ----------------------------------------------------

            try:
                mouse.mouse_up()
            except Exception:
                pass

            self.log(
                "บอทหยุดทำงานแล้ว"
            )

            self._save_learned_bar_width()

    # ========================================================
    # COOLDOWN
    # ========================================================

    def _cooldown(self):

        cooldown = self.cfg.get(
            "post_catch_cooldown",
            2.0
        )

        if cooldown <= 0:
            return

        self.log(
            f"    พักคูลดาวน์ "
            f"{cooldown:.1f} วิ "
            f"ก่อนโยนเบ็ดใหม่..."
        )

        t_end = (
            time.time()
            + cooldown
        )

        while (
            time.time() < t_end
            and not self.stop_event.is_set()
        ):

            time.sleep(0.05)

    # ========================================================
    # PHASE 1
    # CAST
    # ========================================================

    def _phase_1_cast(self):

        region = self.cfg[
            "cast_bar_region"
        ]

        self.log(
            "[1] โยนเบ็ด: คลิกค้าง..."
        )

        mouse.mouse_down()

        best = 0.0
        stable = 0

        try:

            while not self.stop_event.is_set():

                ratio = cast_fill_ratio(
                    region
                )

                # ------------------------------------------------
                # Update best
                # ------------------------------------------------

                if ratio > best:

                    best = ratio
                    stable = 0

                else:

                    stable += 1

                # ------------------------------------------------
                # Full
                # ------------------------------------------------

                if ratio >= 0.97:
                    break

                # ------------------------------------------------
                # Bar หยุด
                # ------------------------------------------------

                if stable >= 8:
                    break

                time.sleep(0.02)

        finally:

            mouse.mouse_up()

        self.log(
            f"    ปล่อยที่ "
            f"{best * 100:.1f}%"
        )

    # ========================================================
    # PHASE 2
    # SHAKE
    # ========================================================

    def _phase_2_shake(
        self,
        max_seconds=5
    ):

        region = self.cfg[
            "shake_search_region"
        ]

        reel_region = self.cfg[
            "reel_bar_region"
        ]

        template = self.cfg[
            "shake_template_path"
        ]

        threshold = self.cfg[
            "match_threshold"
        ]

        self.log(
            "[2] กำลังหาไม้ปุ่ม SHAKE..."
        )

        t0 = time.time()
        visible_frames = 0
        confirm_frames = self.cfg.get(
            "reel_phase2_confirm_frames",
            3
        )
        width_min = self.cfg.get("reel_box_width_min", 70)
        width_max = self.cfg.get("reel_box_width_max", 220)
        edge = self.cfg.get("reel_edge", 3)

        while (
            time.time() - t0 < max_seconds
            and not self.stop_event.is_set()
        ):

            # ------------------------------------------------
            # หา SHAKE
            # ------------------------------------------------

            hit = find_template(
                region,
                template,
                threshold
            )

            if hit:

                # ------------------------------------------------
                # reset timer เมื่อเจอ SHAKE
                # ------------------------------------------------

                t0 = time.time()

                x, y, score = hit

                # ------------------------------------------------
                # random offset เล็กน้อย
                # ------------------------------------------------

                click_x = (
                    x
                    + randint(-3, 3)
                )

                click_y = y

                # ------------------------------------------------
                # click
                # ------------------------------------------------

                mouse.click_at(
                    click_x,
                    click_y
                )

                self.log(
                    f"    คลิก SHAKE "
                    f"({x},{y}) "
                    f"score={score:.2f}"
                )

                time.sleep(0.15)

            # ------------------------------------------------
            # ตรวจว่า reel bar มาแล้ว
            # ------------------------------------------------

            visible = reel_bar_visible(
                reel_region,
                self.learned_bar_width,
                width_min,
                width_max,
                edge,
            )

            if visible:
                visible_frames += 1
            else:
                visible_frames = 0

            if visible_frames >= confirm_frames:

                self.log(
                    "    แถบตกปลาโผล่มาแล้ว!"
                )

                return True

            time.sleep(0.05)

        self.log(
            "    หมดเวลาค้นหา SHAKE"
        )

        return False

    # ========================================================
    # PHASE 3
    # REEL
    # ========================================================

    def _phase_3_reel(self, max_seconds=25):
        region = self.cfg[
            "reel_bar_region"
        ]

        deadzone = self.cfg.get(
            "reel_deadzone",
            5
        )

        lead_time = self.cfg.get(
            "reel_lead_time",
            0.06
        )

        finish_missing = self.cfg.get(
            "reel_finish_missing",
            0.65
        )

        fish_missing_timeout = self.cfg.get(
            "reel_fish_missing",
            finish_missing
        )

        bar_missing_timeout = self.cfg.get(
            "reel_bar_missing",
            finish_missing
        )

        smoothing = self.cfg.get(
            "tracking_smoothing",
            0.40
        )

        loop_delay = self.cfg.get(
            "reel_loop_delay",
            0.008
        )

        t_end = (
            time.time()
            + max_seconds
        )

        max_jump = self.cfg.get("reel_max_jump", 80)
        coast_max = self.cfg.get("reel_coast_max", 3)
        velocity_clamp = self.cfg.get("reel_velocity_clamp", 1000)
        box_width_min = self.cfg.get("reel_box_width_min", 70)
        box_width_max = self.cfg.get("reel_box_width_max", 220)
        box_edge = self.cfg.get("reel_edge", 3)

        # State ของกล่อง: เก็บตำแหน่งและความเร็วไว้ข้ามเฟรม
        box_x = None
        box_velocity = 0.0
        previous_time = None
        coast_frames = 0
        holding = False
        missing_since = None
        fish_missing_since = None
        bar_missing_since = None

        self.log(
            "[3] เริ่ม tracking ปลา..."
        )

        try:

            while (
                time.time() < t_end
                and not self.stop_event.is_set()
            ):

                # ==================================================
                # Capture + detect
                # ==================================================

                fish_x, bar_x, bar_width = (
                    find_fish_and_bar_x(
                        region,
                        None,
                        self.learned_bar_width,
                        box_width_min,
                        box_width_max,
                        box_edge,
                    )
                )

                now = time.time()

                # ถ้าตรวจไม่ครบ ให้หยุดผลักทันทีและรอเฟรมถัดไป
                if fish_x is None or bar_x is None:
                    mouse.mouse_up()
                    holding = False

                    if fish_x is None:
                        if fish_missing_since is None:
                            fish_missing_since = now
                        elif now - fish_missing_since >= fish_missing_timeout:
                            self.log("    ปลาไม่พบต่อเนื่อง จบเฟส 3")
                            return True
                    else:
                        fish_missing_since = None

                    if bar_x is None:
                        if bar_missing_since is None:
                            bar_missing_since = now
                        elif now - bar_missing_since >= bar_missing_timeout:
                            self.log("    กรอบดำไม่พบต่อเนื่อง จบเฟส 3")
                            return True
                    else:
                        bar_missing_since = None

                    if fish_x is None and bar_x is None:
                        if missing_since is None:
                            missing_since = now
                        elif now - missing_since >= finish_missing:
                            self.log("    ปลาและแถบหายแล้ว จบเฟส 3")
                            return True
                    else:
                        missing_since = None

                    time.sleep(loop_delay)
                    continue

                missing_since = None
                fish_missing_since = None
                bar_missing_since = None

                if bar_width is not None:
                    if self.learned_bar_width is None:
                        self.learned_bar_width = float(bar_width)
                    else:
                        self.learned_bar_width += (
                            bar_width
                            - self.learned_bar_width
                        ) * 0.08
                    self.bar_width_samples += 1

                dt = 0.0 if previous_time is None else now - previous_time
                dt = max(dt, 0.001)

                if box_x is None or previous_time is None:
                    box_x = float(bar_x)
                    box_velocity = 0.0
                    coast_frames = 0
                    rejected = False
                else:
                    predicted_box = box_x + box_velocity * dt
                    rejected = (
                        abs(bar_x - predicted_box) > max_jump
                        and coast_frames < coast_max
                    )

                    if rejected:
                        coast_frames += 1
                        box_x = predicted_box
                        box_velocity *= 0.5
                    else:
                        coast_frames = 0
                        raw_velocity = (bar_x - box_x) / dt
                        box_velocity += smoothing * (
                            raw_velocity - box_velocity
                        )
                        box_x += smoothing * (bar_x - box_x)

                box_velocity = max(
                    -velocity_clamp,
                    min(velocity_clamp, box_velocity)
                )
                previous_time = now

                error = fish_x - box_x
                signal = error - lead_time * box_velocity

                # ==================================================
                # DEBUG
                # ==================================================

                self.log(
                    f"fish={fish_x} "
                    f"box={box_x:.1f} "
                    f"bar={bar_x} "
                    f"error={error:.1f} "
                    f"signal={signal:.1f} "
                    f"vel={box_velocity:.1f}"
                    f"{' REJECT' if rejected else ''}"
                )

                # ==================================================
                # PD CONTROL: กดค้างเป็น state ไม่ยิง pulse ทุกเฟรม
                # ==================================================

                if signal > deadzone:
                    if not holding:
                        mouse.mouse_down()
                        holding = True
                elif signal < -deadzone:
                    mouse.mouse_up()
                    holding = False

                # ==================================================
                # Loop
                # ==================================================

                time.sleep(
                    loop_delay
                )

        finally:

            mouse.mouse_up()

        self.log(
            "    จบขั้นตอนตกปลา (หมดเวลาสูงสุด)"
        )