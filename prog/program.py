# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import time

class ImageCorrector:
    def __init__(self, root):
        self.root = root
        self.root.title("Аффинные преобразования (Max Density: 7)")
        self.root.geometry("1400x900")
        
        self.original_img = None
        self.processed_img = None
        self.left_photo = None
        self.right_photo = None
        
        self.points = [] 
        self.base_points = [] 
        self.triangles = []   
        self.dragging_idx = None
        self.last_update_time = 0
        
        self.image_offset_x = 0
        self.image_offset_y = 0
        
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        img_frame = ttk.Frame(main_frame)
        img_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = ttk.LabelFrame(img_frame, text="Исходное (Сетка)")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))
        self.left_canvas = tk.Canvas(left_frame, bg='#2b2b2b', highlightthickness=0)
        self.left_canvas.pack(fill=tk.BOTH, expand=True)
        
        right_frame = ttk.LabelFrame(img_frame, text="Результат (Real-time)")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5,0))
        self.right_canvas = tk.Canvas(right_frame, bg='#2b2b2b', highlightthickness=0)
        self.right_canvas.pack(fill=tk.BOTH, expand=True)
        
        tool_frame = ttk.Frame(main_frame)
        tool_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(tool_frame, text="📁 Открыть фото", command=self.load_image).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(tool_frame, text="Плотность (max 7):").pack(side=tk.LEFT, padx=(10, 2))
        self.grid_size_var = tk.IntVar(value=4)
        # Ограничили до 7
        self.grid_spin = ttk.Spinbox(tool_frame, from_=1, to=7, width=5, textvariable=self.grid_size_var)
        self.grid_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(tool_frame, text="📐 Разместить сетку", command=self.init_grid).pack(side=tk.LEFT, padx=5)
        ttk.Button(tool_frame, text="🔄 Сброс", command=self.reset_grid).pack(side=tk.LEFT, padx=5)

        self.left_canvas.bind("<Button-1>", self.on_click)
        self.left_canvas.bind("<B1-Motion>", self.on_drag)
        self.left_canvas.bind("<ButtonRelease-1>", self.on_release)
        self.left_canvas.bind("<Configure>", lambda e: self.update_displays())

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.processed_img = self.original_img.copy()
                self.points, self.base_points, self.triangles = [], [], []
                self.update_displays()

    def init_grid(self):
        if self.original_img is None: return
        
        # Защита от ввода руками значения больше 7
        n = self.grid_size_var.get()
        if n > 7:
            n = 7
            self.grid_size_var.set(7)
            
        h, w = self.original_img.shape[:2]
        self.points = [[float(x), float(y)] for y in np.linspace(0, h, n + 1) for x in np.linspace(0, w, n + 1)]
        self.base_points = [p[:] for p in self.points]
        
        self.triangles = []
        cols = n + 1
        for j in range(n):
            for i in range(n):
                p1, p2, p3, p4 = j*cols+i, j*cols+i+1, (j+1)*cols+i, (j+1)*cols+i+1
                self.triangles.append((p1, p2, p3))
                self.triangles.append((p2, p4, p3))
        
        self.process_warp()

    def process_warp(self, quality=cv2.INTER_LINEAR):
        if self.original_img is None or not self.points: return
        h, w = self.original_img.shape[:2]
        out_img = np.zeros_like(self.original_img)
        
        # Копируем локально для ускорения доступа
        pts = self.points
        b_pts = self.base_points
        
        for tri in self.triangles:
            src = np.float32([pts[i] for i in tri])
            dst = np.float32([b_pts[i] for i in tri])
            
            M = cv2.getAffineTransform(src, dst)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, dst.astype(np.int32), 255)
            
            warped = cv2.warpAffine(self.original_img, M, (w, h), flags=quality)
            out_img[mask > 0] = warped[mask > 0]
            
        self.processed_img = out_img
        self.update_displays()

    def on_click(self, event):
        if not self.points: return
        scale = self.get_scale()
        for i, p in enumerate(self.points):
            sx, sy = p[0]*scale + self.image_offset_x, p[1]*scale + self.image_offset_y
            if abs(sx - event.x) < 12 and abs(sy - event.y) < 12:
                self.dragging_idx = i
                break

    def on_drag(self, event):
        if self.dragging_idx is None: return
        
        curr_time = time.time()
        # Лимит обновлений для плавности (50 FPS)
        if curr_time - self.last_update_time < 0.02: return 
        self.last_update_time = curr_time

        scale = self.get_scale()
        h, w = self.original_img.shape[:2]
        nx = (event.x - self.image_offset_x) / scale
        ny = (event.y - self.image_offset_y) / scale
        
        self.points[self.dragging_idx] = [max(0, min(w, nx)), max(0, min(h, ny))]
        # Используем NEAREST при движении
        self.process_warp(quality=cv2.INTER_NEAREST)

    def on_release(self, event):
        if self.dragging_idx is not None:
            self.dragging_idx = None
            # Финальный качественный рендер
            self.process_warp(quality=cv2.INTER_LINEAR)

    def get_scale(self):
        if self.original_img is None: return 1.0
        h, w = self.original_img.shape[:2]
        cw, ch = self.left_canvas.winfo_width(), self.left_canvas.winfo_height()
        if cw < 10 or ch < 10: return 1.0
        return min(cw/w, ch/h) * 0.95

    def update_displays(self):
        if self.original_img is None: return
        scale = self.get_scale()
        h, w = self.original_img.shape[:2]
        
        self.image_offset_x = (self.left_canvas.winfo_width() - w*scale)//2
        self.image_offset_y = (self.left_canvas.winfo_height() - h*scale)//2

        # Обновление левой части (Сетка + Точки)
        img_l = cv2.resize(self.original_img, (int(w*scale), int(h*scale)))
        self.left_photo = ImageTk.PhotoImage(Image.fromarray(img_l))
        self.left_canvas.delete("all")
        self.left_canvas.create_image(self.left_canvas.winfo_width()//2, self.left_canvas.winfo_height()//2, image=self.left_photo)
        
        if self.points:
            for tri in self.triangles:
                pts = []
                for idx in tri:
                    pts.extend([self.points[idx][0]*scale + self.image_offset_x, 
                               self.points[idx][1]*scale + self.image_offset_y])
                self.left_canvas.create_polygon(pts, fill="", outline="#00ffcc", width=1)
            
            for p in self.points:
                sx, sy = p[0]*scale + self.image_offset_x, p[1]*scale + self.image_offset_y
                self.left_canvas.create_oval(sx-3, sy-3, sx+3, sy+3, fill="#ff3366", outline="white")

        # Обновление правой части (Результат)
        if self.processed_img is not None:
            img_r = cv2.resize(self.processed_img, (int(w*scale), int(h*scale)))
            self.right_photo = ImageTk.PhotoImage(Image.fromarray(img_r))
            self.right_canvas.delete("all")
            self.right_canvas.create_image(self.right_canvas.winfo_width()//2, self.right_canvas.winfo_height()//2, image=self.right_photo)

    def reset_grid(self):
        if self.base_points:
            self.points = [p[:] for p in self.base_points]
            self.process_warp()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCorrector(root)
    root.mainloop()
