"""ตรรกะบอทตกปลา 3 ขั้นตอน แยกจาก GUI เพื่อรันใน background thread และหยุดได้ทันที"""

import time
import threading
import numpy as np
import cv2
import mss

import input_backend as mouse

sct = mss.mss()


def grab(region):
    x, y, w, h = region
    img = np.array(sct.grab({"left": x, "top": y, "width": w, "height": h}))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def find_template(region, template_path, threshold):
    frame = grab(region)
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        return None
    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val >= threshold:
        th, tw = template.shape[:2]
        cx = region[0] + max_loc[0] + tw // 2
        cy = region[1] + max_loc[1] + th // 2
        return (cx, cy, max_val)
    return None


def cast_fill_ratio(region):
    frame = grab(region)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([90, 60, 60]), np.array([130, 255, 255]))
    filled_rows = np.any(mask, axis=1)
    return filled_rows.sum() / mask.shape[0]


def reel_bar_visible(region):
    frame = grab(region)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([45, 80, 80]), np.array([75, 255, 255]))
    return mask.sum() > 500


def find_fish_and_bar_x(region, black_thresh=40):
    """คืน (fish_x, bar_x, stats) โดย fish_x/bar_x เป็นพิกเซล x ภายในเฟรมของ region

    bar_x หาโดยดู 'ก้อน' (contour) สีดำที่สูงเกือบเต็มความสูงของแถบแต่แคบกว่าความกว้าง
    ทั้งหมดมาก (ลักษณะของตัวชี้/แถบควบคุมที่เลื่อนซ้ายขวา) แทนที่จะเฉลี่ยพิกเซลดำ
    ทั้งหมดในเฟรมตรงๆ ซึ่งจะโดนองค์ประกอบดำที่อยู่นิ่ง (กรอบ/ไอคอน/พื้นหลัง) ถ่วงค่าเฉลี่ย
    ให้เพี้ยนไปทางตำแหน่งขององค์ประกอบนิ่งนั้นแทน
    """
    frame = grab(region)
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, np.array([45, 80, 80]), np.array([75, 255, 255]))
    fish_x = None
    ys, xs = np.where(green_mask > 0)
    if len(xs) > 0:
        fish_x = int(xs.mean())

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    black_mask = (gray < black_thresh).astype(np.uint8) * 255

    bar_x = None
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        # ตัวชี้ควรสูงเกือบเต็มความสูงของแถบ แต่แคบกว่าความกว้างทั้งแถบมาก
        if ch >= h * 0.5 and cw <= w * 0.5:
            area = cw * ch
            if best is None or area > best[0]:
                best = (area, cx + cw // 2)
    if best:
        bar_x = best[1]
    else:
        # หาก้อนที่เข้าเงื่อนไขไม่เจอ -> fallback ใช้ค่าเฉลี่ยทั้งหมด (จะไม่แม่นเท่า)
        ys2, xs2 = np.where(black_mask > 0)
        if len(xs2) > 0:
            bar_x = int(xs2.mean())

    stats = {
        "fish_pixels": int((green_mask > 0).sum()),
        "black_pixels": int((black_mask > 0).sum()),
        "used_contour": best is not None,
    }
    return fish_x, bar_x, stats


def debug_snapshot(region, save_path="debug_reel.png"):
    """แคปเฟรมปัจจุบันของ region, รันตรวจจับเหมือนตอนบอททำงานจริง, วาดเส้นทับตำแหน่ง
    fish_x (เขียว) กับ bar_x (แดง) ลงในภาพ แล้วเซฟไว้ให้ดูว่าตรวจจับถูกจุดไหม"""
    fish_x, bar_x, stats = find_fish_and_bar_x(region)
    frame = grab(region)
    h = frame.shape[0]
    if fish_x is not None:
        cv2.line(frame, (fish_x, 0), (fish_x, h), (0, 255, 0), 2)
    if bar_x is not None:
        cv2.line(frame, (bar_x, 0), (bar_x, h), (0, 0, 255), 2)
    cv2.imwrite(save_path, frame)
    return fish_x, bar_x, stats


class FishBot:
    """รันลูป 3 ขั้นตอนใน thread แยก เรียก .start() / .stop() จาก GUI ได้ตลอดเวลา"""

    def __init__(self, config, log_fn):
        self.cfg = config
        self.log = log_fn
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def _run(self):
        self.log(f"เริ่มบอทใน 3 วินาที... (มือควบคุมเมาส์: {mouse.BACKEND_NAME})")
        for _ in range(30):
            if self.stop_event.is_set():
                return
            time.sleep(0.1)
        try:
            while not self.stop_event.is_set():
                self._phase_1_cast()
                if self.stop_event.is_set():
                    break
                found = self._phase_2_shake()
                if self.stop_event.is_set():
                    break
                if found:
                    self._phase_3_reel()
                if self.stop_event.is_set():
                    break
                self._cooldown()
        finally:
            self.log("บอทหยุดทำงานแล้ว")

    def _cooldown(self):
        """พักหลังตกจบ/หลุด ก่อนเริ่มโยนเบ็ดใหม่ — กันปัญหาโยนเบ็ดใหม่เร็วเกินไปจน
        เกมยังไม่พร้อม (ยังไม่จบ animation/ยังไม่ reset สถานะ) ทำให้บอทไปวนหา SHAKE
        ทั้งที่ยังไม่ได้เริ่มโยนเบ็ดจริง"""
        cooldown = self.cfg.get("post_catch_cooldown", 2.0)
        if cooldown <= 0:
            return
        self.log(f"    พักคูลดาวน์ {cooldown:.1f} วิ ก่อนโยนเบ็ดใหม่...")
        t_end = time.time() + cooldown
        while time.time() < t_end and not self.stop_event.is_set():
            time.sleep(0.05)

    def _phase_1_cast(self):
        region = self.cfg["cast_bar_region"]
        self.log("[1] โยนเบ็ด: คลิกค้าง...")
        mouse.mouse_down()
        best, stable = 0.0, 0
        while not self.stop_event.is_set():
            ratio = cast_fill_ratio(region)
            if ratio > best:
                best, stable = ratio, 0
            else:
                stable += 1
            if ratio >= 0.97 or stable >= 8:
                break
            time.sleep(0.02)
        mouse.mouse_up()
        self.log(f"    ปล่อยที่ {best * 100:.1f}%")

    def _phase_2_shake(self, max_seconds=30):
        region = self.cfg["shake_search_region"]
        reel_region = self.cfg["reel_bar_region"]
        template = self.cfg["shake_template_path"]
        self.log("[2] กำลังหาไม้ปุ่ม SHAKE...")
        t0 = time.time()
        while time.time() - t0 < max_seconds and not self.stop_event.is_set():
            hit = find_template(region, template, self.cfg["match_threshold"])
            if hit:
                x, y, score = hit
                # ขยับเมาส์ไปจุดใหม่ก่อนคลิกเสมอ เพื่อบังคับ mousemove event จริงๆ
                # ให้เกมเห็น ไม่ให้ 'คลิกลอย' เพราะเมาส์ถูกวาปไปตรงนั้นเฉยๆ
                mouse.click_at(x, y)
                self.log(f"    คลิก SHAKE ({x},{y}) score={score:.2f}")
                time.sleep(0.15)
            if reel_bar_visible(reel_region):
                self.log("    แถบตกปลาโผล่มาแล้ว!")
                return True
            time.sleep(0.05)
        self.log("    หมดเวลาค้นหา SHAKE")
        return False

    def _phase_3_reel(self, max_seconds=25):
        region = self.cfg["reel_bar_region"]
        deadzone = self.cfg.get("reel_deadzone", 6)
        lead_time = self.cfg.get("reel_lead_time", 0.05)
        self.log(f"[3] ประคองแถบดำให้ตรงปลา (deadzone={deadzone}px)...")
        t_end = time.time() + max_seconds
        holding = False
        prev_error = None
        prev_time = time.time()
        while time.time() < t_end and not self.stop_event.is_set():
            if not reel_bar_visible(region):
                self.log("    แถบตกปลาหายไป (จบ/หลุด)")
                break
            fish_x, bar_x, _stats = find_fish_and_bar_x(region)
            if fish_x is None or bar_x is None:
                time.sleep(0.005)
                continue

            error = fish_x - bar_x  # + = ปลาอยู่ขวาของแถบ ต้องคลิกไปขวา
            now = time.time()
            dt = max(now - prev_time, 1e-3)
            velocity = 0.0
            if prev_error is not None:
                velocity = (error - prev_error) / dt
            prev_error, prev_time = error, now

            # พยากรณ์ error ล่วงหน้าเล็กน้อย เพื่อชดเชย latency ของลูปตรวจจับ
            predicted_error = error + velocity * lead_time

            # hysteresis: สลับสถานะเฉพาะตอนเกิน deadzone เท่านั้น ไม่สลับถี่ตรงกลาง
            if predicted_error > deadzone:
                if not holding:
                    mouse.mouse_down()
                    holding = True
            elif predicted_error < -deadzone:
                if holding:
                    mouse.mouse_up()
                    holding = False
            # อยู่ในโซนกลาง -> คงสถานะเดิมไว้ (กันแถบสั่นถี่)

            time.sleep(0.01)
        if holding:
            mouse.mouse_up()
        self.log("    จบขั้นตอนตกปลา")
