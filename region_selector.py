"""หน้าต่างเต็มจอโปร่งใส ให้ลากเมาส์คลุมพื้นที่ที่ต้องการ แล้วคืนพิกัด (x, y, w, h)"""

import tkinter as tk


class RegionSelector:
    def __init__(self, root, on_done):
        self.on_done = on_done
        self.start_x = self.start_y = 0
        self.rect = None

        self.top = tk.Toplevel(root)
        self.top.attributes("-fullscreen", True)
        self.top.attributes("-alpha", 0.25)
        self.top.attributes("-topmost", True)
        self.top.configure(bg="black")
        self.top.config(cursor="cross")

        self.canvas = tk.Canvas(self.top, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.create_text(
            self.top.winfo_screenwidth() // 2, 40,
            text="ลากเมาส์คลุมพื้นที่ที่ต้องการ แล้วปล่อย   (Esc = ยกเลิก)",
            fill="white", font=("Arial", 16, "bold"),
        )

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.top.bind("<Escape>", lambda e: self._cancel())
        self.top.focus_force()

    def _on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="#00ff88", width=2,
        )

    def _on_drag(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def _on_release(self, event):
        x0, y0 = self.start_x, self.start_y
        x1, y1 = event.x, event.y
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        self.top.destroy()
        if w > 2 and h > 2:
            self.on_done((x, y, w, h))
        else:
            self.on_done(None)

    def _cancel(self):
        self.top.destroy()
        self.on_done(None)
