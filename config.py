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
