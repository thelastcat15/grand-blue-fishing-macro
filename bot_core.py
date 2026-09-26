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


def _fish_masks(frame):
    """mask ปลาสีเขียว / สีแดง (คืนเป็น tuple ของ mask)"""
    blue, green, red = cv2.split(frame.astype(np.int16))
    return (
        (green > 110) & (blue < green - 25) & (red < green - 15),
        (red > 110) & (green < red - 35) & (blue < red - 35),
    )


def _fish_columns(frame):
    """คอลัมน์ที่มีพิกเซลปลา (ใช้เช็คว่าปลาบังกรอบดำอยู่ข้างไหน)"""
    green_mask, red_mask = _fish_masks(frame)
    return (green_mask | red_mask).sum(axis=0) > 2


def find_fish_x(frame, prev_fish_x=None):
    """หา center X ของปลาโดยนับช่วงคอลัมน์สีเขียว/แดง"""
    candidates = []

    for mask in _fish_masks(frame):
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
# FIND BLACK REEL BOX
# ============================================================
#
# กรอบดำ = สี่เหลี่ยมสีเทาเข้ม (B≈G≈R) มีลูกศรสีขาว "<" อยู่ข้างใน
#   - พื้นน้ำเป็นสีน้ำเงิน, ปลาเป็นเขียว/แดง -> มีสี (chroma สูง)
#   - กรอบดำ + ลูกศรขาว -> ไม่มีสี (chroma ต่ำ)
# จึงนับเฉพาะคอลัมน์ที่ "เกือบทั้งแถว" เป็นพิกเซลไม่มีสี
# เส้นเอ็นตกปลาบาง ๆ จะไม่ผ่านเกณฑ์นี้
#
# เมื่อปลาว่ายทับกรอบ กรอบจะถูกตัดเป็นท่อน -> เชื่อมท่อนที่คั่นด้วยปลา
# ถ้ายังเห็นแค่บางส่วน -> ใช้ขอบด้านที่ไม่ถูกบัง + ความกว้างที่เรียนรู้ไว้

BOX_COLUMN_FILL = 0.6       # สัดส่วนแถวที่ต้องเป็นสีเทา/ขาว จึงนับเป็นคอลัมน์ของกรอบ
BOX_MIN_VISIBLE = 18        # ส่วนของกรอบที่ต้องเห็นอย่างน้อย (px)
BOX_PARTIAL_RATIO = 0.85    # เห็นแคบกว่า expected * ค่านี้ = ถูกบัง/ถูกตัดขอบ
BOX_DEFAULT_WIDTH = 80.0    # ใช้ถ้ายังไม่เคยเรียนรู้ความกว้าง


def _neutral_mask(frame):
    """พิกเซลที่ไม่มีสี: เทาเข้มของกรอบ + ลูกศรขาว"""
    f = frame.astype(np.int16)
    chroma = f.max(axis=2) - f.min(axis=2)
    return chroma < 24


def _white_mask(frame):
    """ลูกศรสีขาวในกรอบ"""
    f = frame.astype(np.int16)
    return (f.min(axis=2) > 170) & (f.max(axis=2) - f.min(axis=2) < 40)


def detect_reel_box(frame, expected_width=None, edge=3):
    """
    หากรอบดำในแถบตกปลา

    คืนค่า dict:
        center   : center X ของกรอบ (ประมาณแม้ถูกปลาบัง)
        width    : ความกว้างที่มองเห็นจริง
        full     : True ถ้าเห็นกรอบครบ (ใช้เรียนรู้ความกว้างได้)
    หรือ None ถ้าไม่เจอ
    """
    height, width = frame.shape[:2]
    band = slice(int(height * 0.15), max(int(height * 0.85), 1))

    neutral = _neutral_mask(frame)[band]
    band_h = neutral.shape[0]
    if band_h == 0:
        return None

    strong = neutral.mean(axis=0) >= BOX_COLUMN_FILL
    chevron = _white_mask(frame)[band].sum(axis=0) >= band_h * 0.25
    fish_cols = _fish_columns(frame)

    runs = [[start, end] for start, end, _ in _runs(strong)]
    if not runs:
        return None

    exp_w = float(expected_width) if expected_width else None
    merge_limit = (exp_w or BOX_DEFAULT_WIDTH) * 1.3

    # เชื่อมท่อนที่คั่นด้วยช่องเล็ก ๆ หรือคั่นด้วยตัวปลา
    merged = [runs[0]]
    for start, end in runs[1:]:
        prev = merged[-1]
        gap = start - prev[1] - 1
        gap_is_fish = gap > 0 and fish_cols[prev[1] + 1:start].mean() >= 0.7
        if (gap <= 3 or gap_is_fish) and end - prev[0] + 1 <= merge_limit:
            prev[1] = end
        else:
            merged.append([start, end])

    candidates = []
    for start, end in merged:
        w = end - start + 1
        if w < BOX_MIN_VISIBLE:
            continue

        has_chevron = chevron[start:end + 1].sum() >= 3
        touch_left = start <= edge
        touch_right = end >= width - 1 - edge

        # ขอบกรอบ UI สีเทาที่ปลายแถบ (แคบ ติดขอบ ไม่มีลูกศร) -> ไม่ใช่กรอบดำ
        ref_w = exp_w or BOX_DEFAULT_WIDTH
        if (
            not has_chevron
            and (touch_left or touch_right)
            and w < ref_w * 0.6
        ):
            continue

        candidates.append({
            "start": start,
            "end": end,
            "width": w,
            "chevron": has_chevron,
            "touch_left": touch_left,
            "touch_right": touch_right,
            "fish_left": bool(fish_cols[max(0, start - 3):start].any()),
            "fish_right": bool(fish_cols[end + 1:end + 4].any()),
        })

    if not candidates:
        return None

    def score(c):
        s = c["width"]
        if c["chevron"]:
            s += 1000
        if exp_w is not None:
            s -= abs(c["width"] - exp_w) * 0.5
        return s

    box = max(candidates, key=score)
    start, end, w = box["start"], box["end"], box["width"]
    center = (start + end) / 2

    # กรอบชิดซ้ายสุด (ตำแหน่งพัก) ยังนับว่าเห็นครบ ถ้าลูกศรไม่ถูกตัด
    left_clipped = box["touch_left"]
    if left_clipped and box["chevron"]:
        chevron_start = start + int(np.argmax(chevron[start:end + 1]))
        left_clipped = chevron_start - start < 4

    # ชนขอบขวา = อาจรวมกับขอบ UI ปลายแถบ -> ไม่ใช้เรียนรู้ความกว้าง
    full = not (left_clipped or box["touch_right"]
                or box["fish_left"] or box["fish_right"])

    ref_w = exp_w or BOX_DEFAULT_WIDTH
    if not full:
        half = ref_w / 2
        if w < ref_w * BOX_PARTIAL_RATIO:
            # เห็นไม่ครบ -> ยึดขอบด้านที่ไม่ถูกบัง
            full = False
            if left_clipped and not box["touch_right"]:
                center = end - half + 0.5
            elif box["touch_right"] and not box["touch_left"]:
                center = start + half - 0.5
            elif box["fish_right"] and not box["fish_left"]:
                center = start + half - 0.5
            elif box["fish_left"] and not box["fish_right"]:
                center = end - half + 0.5
        elif w > ref_w * 1.15:
            # กว้างเกิน (ติดกับขอบ UI ที่ปลายแถบ) -> ยึดขอบด้านที่ไม่ติด
            full = False
            if box["touch_right"] and not box["touch_left"]:
                center = start + half - 0.5
            elif box["touch_left"] and not box["touch_right"]:
                center = end - half + 0.5

    return {
        "center": int(round(center)),
        "width": int(w),
        "full": full,
        "start": int(start),
        "end": int(end),
    }


def find_bar_x(
    frame,
    expected_width=None,
    min_width=None,
    max_width=None,
    edge=3,
):
    """คืนค่า center X ของกรอบดำ (min/max_width เก็บไว้เพื่อ compatibility)"""
    box = detect_reel_box(frame, expected_width, edge)
    return None if box is None else box["center"]


def find_bar_width(
    frame,
    expected_width=None,
    min_width=None,
    max_width=None,
    edge=3,
):
    """คืนค่าความกว้างกรอบดำ เฉพาะตอนเห็นครบ (ใช้เรียนรู้)"""
    box = detect_reel_box(frame, expected_width, edge)
    if box is None or not box["full"]:
        return None
    return box["width"]

# ============================================================
# FIND FISH + BAR
# ============================================================

def find_fish_and_bar_x(
    region,
    prev_fish_x=None,
    expected_bar_width=None,
    bar_width_min=None,
    bar_width_max=None,
    bar_edge=3,
    frame=None,
):

    if frame is None:
        frame = grab(region)

    fish_x = find_fish_x(
        frame,
        prev_fish_x
    )

    box = detect_reel_box(
        frame,
        expected_bar_width,
        bar_edge,
    )

    bar_x = None if box is None else box["center"]
    bar_width = (
        box["width"]
        if box is not None and box["full"]
        else None
    )

    # กันค่าเพี้ยน: เรียนรู้เฉพาะความกว้างที่อยู่ในช่วงที่ตั้งไว้
    if bar_width is not None:
        if bar_width_min and bar_width < bar_width_min * 0.7:
            bar_width = None
        elif bar_width_max and bar_width > bar_width_max:
            bar_width = None

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

                time.sleep(0.05)

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