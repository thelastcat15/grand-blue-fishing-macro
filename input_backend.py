"""
เลเยอร์ควบคุมเมาส์ ทำ 2 อย่างที่ช่วยให้เกม 'เห็น' การคลิกจริงๆ:

1. ใช้ pydirectinput แทน pyautogui บน Windows ถ้ามีติดตั้งไว้
   (เกมจำนวนมาก โดยเฉพาะเกมที่ใช้ DirectInput/RawInput จะไม่รับตำแหน่งเมาส์ที่ถูก
   'วาป' ไปด้วย SetCursorPos ตรงๆ — pydirectinput ส่ง event ในแบบที่เกมพวกนี้เข้าใจ)
2. ก่อนคลิกทุกครั้ง จะขยับเมาส์ไปที่จุดหมายแบบมี duration สั้นๆ แล้ว "สั่น" 1px
   เพื่อบังคับให้เกิด mousemove event จริงก่อนที่จะกดปุ่ม — แก้ปัญหาคลิกไม่ติด
   ทั้งที่พิกัดตรงปุ่มพอดี (ต้องขยับเมาส์นิดนึงถึงจะคลิกได้)

ถ้าไม่มี pydirectinput (หรือไม่ใช่ Windows) จะ fallback ไปใช้ pyautogui อัตโนมัติ
"""

import time
import platform

_backend = None
BACKEND_NAME = "pyautogui"

if platform.system() == "Windows":
    try:
        import pydirectinput as _backend
        BACKEND_NAME = "pydirectinput"
    except ImportError:
        _backend = None

if _backend is None:
    import pyautogui as _backend
    BACKEND_NAME = "pyautogui"

_backend.FAILSAFE = True
_backend.PAUSE = 0


def move_to(x, y, duration=0.05):
    """ขยับเมาส์ไปตำแหน่ง (x, y) แบบมี duration + สั่น 1px เพื่อบังคับ mousemove event จริง"""
    _backend.moveTo(x, y, duration=duration)
    _backend.moveRel(1, 0, duration=0.01)
    _backend.moveRel(-1, 0, duration=0.01)


def click_at(x, y, hold=0.05, pre_delay=0.05):
    """ขยับไปที่ (x, y) แล้วค่อยคลิก (แยก mouseDown/mouseUp เอง แทนการเรียก .click() เฉยๆ)"""
    move_to(x, y)
    time.sleep(pre_delay)
    _backend.mouseDown()
    time.sleep(hold)
    _backend.mouseUp()


def mouse_down():
    _backend.mouseDown()


def mouse_up():
    _backend.mouseUp()
