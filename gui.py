"""
Auto Fishing Bot — GUI (Tkinter)
=================================
รันไฟล์นี้แล้วใช้ปุ่มในโปรแกรมลาก-เลือกพื้นที่บนจอแทนการพิมพ์พิกัดเอง

  python gui.py
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox

import mss
from PIL import Image, ImageTk

from config import load_config, save_config
from region_selector import RegionSelector
from bot_core import FishBot

ASSET_DIR = "assets"
os.makedirs(ASSET_DIR, exist_ok=True)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Fishing Bot")
        self.root.geometry("500x560")
        self.root.resizable(False, False)

        self.cfg = load_config()
        self.bot = FishBot(self.cfg, self.log)

        self._build_ui()

    # ---------------------------------------------------------- UI ----
    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        frame_regions = ttk.LabelFrame(self.root, text="1) ตั้งค่าพื้นที่บนจอ (ลากเลือกได้เลย)")
        frame_regions.pack(fill="x", **pad)

        self._region_row(frame_regions, "หลอดพลัง (โยนเบ็ด)", "cast_bar_region", self.select_cast_bar)
        self._region_row(frame_regions, "พื้นที่หาไม้ปุ่ม SHAKE", "shake_search_region", self.select_shake_search)
        self._region_row(frame_regions, "แถบตกปลาแนวนอน", "reel_bar_region", self.select_reel_bar)

        frame_template = ttk.LabelFrame(self.root, text="2) รูปปุ่ม SHAKE (ใช้จับคู่บนจอ)")
        frame_template.pack(fill="x", **pad)
        self.template_preview = ttk.Label(frame_template, text="(ยังไม่มีรูป)")
        self.template_preview.pack(side="left", padx=8, pady=8)
        ttk.Button(
            frame_template, text="ลากเลือกปุ่ม SHAKE บนจอ", command=self.select_shake_template
        ).pack(side="left", padx=8)

        frame_adv = ttk.LabelFrame(self.root, text="ตั้งค่าขั้นสูง (ปรับความแม่นยำ)")
        frame_adv.pack(fill="x", **pad)

        row_th = ttk.Frame(frame_adv)
        row_th.pack(fill="x", padx=6, pady=3)
        ttk.Label(row_th, text="Match threshold ปุ่ม SHAKE", width=26).pack(side="left")
        self.var_threshold = tk.DoubleVar(value=self.cfg.get("match_threshold", 0.75))
        ttk.Spinbox(
            row_th, from_=0.3, to=0.99, increment=0.05, width=6,
            textvariable=self.var_threshold, command=self._on_adv_change,
        ).pack(side="left")
        ttk.Label(row_th, text="(จับไม่ติด → ลดค่านี้ลง)", foreground="#666").pack(side="left", padx=6)

        row_dz = ttk.Frame(frame_adv)
        row_dz.pack(fill="x", padx=6, pady=3)
        ttk.Label(row_dz, text="Deadzone เฟส 3 (px)", width=26).pack(side="left")
        self.var_deadzone = tk.IntVar(value=self.cfg.get("reel_deadzone", 6))
        ttk.Spinbox(
            row_dz, from_=1, to=40, increment=1, width=6,
            textvariable=self.var_deadzone, command=self._on_adv_change,
        ).pack(side="left")
        ttk.Label(row_dz, text="(แถบสั่นถี่ → เพิ่มค่านี้ขึ้น)", foreground="#666").pack(side="left", padx=6)

        row_cd = ttk.Frame(frame_adv)
        row_cd.pack(fill="x", padx=6, pady=3)
        ttk.Label(row_cd, text="Cooldown ก่อนโยนเบ็ดใหม่ (วิ)", width=26).pack(side="left")
        self.var_cooldown = tk.DoubleVar(value=self.cfg.get("post_catch_cooldown", 2.0))
        ttk.Spinbox(
            row_cd, from_=0, to=15, increment=0.5, width=6,
            textvariable=self.var_cooldown, command=self._on_adv_change,
        ).pack(side="left")
        ttk.Label(row_cd, text="(โยนเร็วไปจนไม่ติด → เพิ่มค่านี้ขึ้น)", foreground="#666").pack(
            side="left", padx=6
        )

        frame_ctrl = ttk.LabelFrame(self.root, text="3) ควบคุมบอท")
        frame_ctrl.pack(fill="x", **pad)
        self.start_btn = ttk.Button(frame_ctrl, text="▶ เริ่มบอท", command=self.start_bot)
        self.start_btn.pack(side="left", padx=8, pady=8)
        self.stop_btn = ttk.Button(frame_ctrl, text="■ หยุดบอท", command=self.stop_bot, state="disabled")
        self.stop_btn.pack(side="left", padx=8, pady=8)
        ttk.Button(frame_ctrl, text="บันทึกค่าตั้งค่า", command=self.save_cfg).pack(side="left", padx=8, pady=8)

        frame_log = ttk.LabelFrame(self.root, text="สถานะ")
        frame_log.pack(fill="both", expand=True, **pad)
        self.log_text = tk.Text(frame_log, height=14, state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)

        self._refresh_region_labels()
        self._refresh_template_preview()
        self.log("โหลดค่าตั้งค่าเดิม (ถ้ามี) จาก fishbot_config.json แล้ว")

    def _region_row(self, parent, label, key, command):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=6, pady=4)
        ttk.Label(row, text=label, width=22).pack(side="left")
        value_label = ttk.Label(row, text="ยังไม่ได้เลือก", foreground="#aa3333")
        value_label.pack(side="left", padx=8)
        setattr(self, f"label_{key}", value_label)
        ttk.Button(row, text="ลากเลือกพื้นที่บนจอ", command=command).pack(side="right")

    def _refresh_region_labels(self):
        for key in ("cast_bar_region", "shake_search_region", "reel_bar_region"):
            label = getattr(self, f"label_{key}")
            region = self.cfg.get(key)
            if region:
                label.config(text=str(tuple(region)), foreground="#008800")
            else:
                label.config(text="ยังไม่ได้เลือก", foreground="#aa3333")

    def _refresh_template_preview(self):
        path = self.cfg.get("shake_template_path")
        if path and os.path.exists(path):
            img = Image.open(path)
            img.thumbnail((80, 80))
            self._tpl_img = ImageTk.PhotoImage(img)
            self.template_preview.config(image=self._tpl_img, text="")
        else:
            self.template_preview.config(text="(ยังไม่มีรูป)", image="")

    def _on_adv_change(self):
        try:
            self.cfg["match_threshold"] = float(self.var_threshold.get())
            self.cfg["reel_deadzone"] = int(self.var_deadzone.get())
            self.cfg["post_catch_cooldown"] = float(self.var_cooldown.get())
            self.log(
                f"ปรับค่า: match_threshold={self.cfg['match_threshold']:.2f}, "
                f"reel_deadzone={self.cfg['reel_deadzone']}px, "
                f"cooldown={self.cfg['post_catch_cooldown']:.1f}วิ"
            )
        except (tk.TclError, ValueError):
            pass

    def log(self, msg):
        def _append():
            self.log_text.config(state="normal")
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.root.after(0, _append)

    # ------------------------------------------------ region selection ----
    def _select_region(self, key):
        # ซ่อนหน้าต่างหลักชั่วคราวไม่ให้บังจอตอนลากเลือก
        self.root.withdraw()

        def done(region):
            self.root.deiconify()
            if region:
                self.cfg[key] = list(region)
                self._refresh_region_labels()
                self.log(f"ตั้งค่า {key} = {tuple(region)}")

        self.root.after(200, lambda: RegionSelector(self.root, done))

    def select_cast_bar(self):
        self._select_region("cast_bar_region")

    def select_shake_search(self):
        self._select_region("shake_search_region")

    def select_reel_bar(self):
        self._select_region("reel_bar_region")

    def select_shake_template(self):
        self.root.withdraw()

        def done(region):
            self.root.deiconify()
            if not region:
                return
            x, y, w, h = region
            with mss.mss() as sct:
                shot = sct.grab({"left": x, "top": y, "width": w, "height": h})
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            path = os.path.join(ASSET_DIR, "shake_button.png")
            img.save(path)
            self.cfg["shake_template_path"] = path
            self._refresh_template_preview()
            self.log(f"บันทึกรูปปุ่ม SHAKE ที่ {path}")

        self.root.after(200, lambda: RegionSelector(self.root, done))

    # -------------------------------------------------------- controls ----
    def save_cfg(self):
        save_config(self.cfg)
        self.log("บันทึกค่าตั้งค่าลงไฟล์ fishbot_config.json แล้ว")

    def start_bot(self):
        missing = [
            k for k in ("cast_bar_region", "shake_search_region", "reel_bar_region")
            if not self.cfg.get(k)
        ]
        if missing or not os.path.exists(self.cfg.get("shake_template_path", "")):
            messagebox.showwarning(
                "ยังตั้งค่าไม่ครบ", "กรุณาลากเลือกพื้นที่ทั้งหมด และรูปปุ่ม SHAKE ให้ครบก่อนเริ่มบอท"
            )
            return
        self.bot = FishBot(self.cfg, self.log)
        self.bot.start()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._watch_bot()

    def stop_bot(self):
        self.bot.stop()
        self.stop_btn.config(state="disabled")
        self.start_btn.config(state="normal")

    def _watch_bot(self):
        if self.bot.is_running():
            self.root.after(500, self._watch_bot)
        else:
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
