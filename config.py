"""เก็บ/โหลดค่าตั้งค่า (พิกัดกรอบต่างๆ) ลงไฟล์ JSON เพื่อไม่ต้องเลือกใหม่ทุกครั้ง"""

import json
import os

CONFIG_PATH = "fishbot_config.json"

DEFAULT_CONFIG = {
    "cast_bar_region": None,        # [x, y, w, h] หลอดพลังแนวตั้งตอนโยนเบ็ด
    "shake_search_region": None,    # [x, y, w, h] พื้นที่กว้างๆ ที่ปุ่ม SHAKE อาจโผล่
    "reel_bar_region": None,        # [x, y, w, h] แถบตกปลาแนวนอนทั้งแท่ง
    "shake_template_path": "assets/shake_button.png",
    "match_threshold": 0.75,
    "reel_deadzone": 6,      # px — โซนกลางที่ไม่สลับสถานะคลิก/ปล่อย (กันสั่นถี่ในเฟส 3)
    "reel_lead_time": 0.05,  # วินาที — พยากรณ์ error ล่วงหน้าเพื่อชดเชย latency ของลูป
    "reel_bar_width": None,  # ความกว้างแถบดำที่เรียนรู้จากภาพจริง
    "reel_control_range": 55,  # ระยะ error ที่ถือว่าแรงควบคุมเต็ม
    "reel_min_hold": 0.015,  # เวลากดขั้นต่ำต่อพัลส์
    "reel_max_hold": 0.14,  # เวลากดสูงสุดต่อพัลส์
    "reel_finish_missing": 0.65,  # ปลาและแถบหายต่อเนื่องกี่วินาทีจึงถือว่าจบ
    "reel_fish_missing": 0.65,  # ปลาหายต่อเนื่องกี่วินาทีจึงถือว่าจบเฟส
    "reel_bar_missing": 0.65,  # กรอบดำหายต่อเนื่องกี่วินาทีจึงถือว่าจบเฟส
    "reel_phase2_confirm_frames": 3,
    "reel_box_width_min": 70,
    "reel_box_width_max": 220,
    "reel_edge": 3,
    "reel_max_jump": 80,
    "reel_coast_max": 3,
    "reel_velocity_clamp": 1000,
    "post_catch_cooldown": 2.0,  # วินาที — พักหลังตกจบ/หลุด ก่อนเริ่มโยนเบ็ดใหม่
}


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(data)
        return cfg
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
