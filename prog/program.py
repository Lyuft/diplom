# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import sys

class Params:
    CANVAS_SCALE = 0.95
    POINT_SIZE = 12

class ImageCorrector:
    def __init__(self, root):
        self.root = root
        self.root.title("Программа коррекции документов (Дипломная работа)")
        self.root.geometry("1400x900")
        
        self.original_img = None
        self.processed_img = None
        self.left_photo = None
        self.right_photo = None
        
        self.persp_points = []
        self.manual_mode = False
        self.dragging_idx = None
        self.show_grid = False  # Флаг сетки
        
        # Смещение для центрирования
        self.image_offset_x = 0
        self.image_offset_y = 0
        
        self.setup_ui()
        self.update_status("✅ Загрузите изображение для начала работы")

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Панель просмотра
        img_frame = ttk.Frame(main_frame)
        img_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = ttk.LabelFrame(img_frame, text="Исходное (Разметка точек)")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))
        self.left_canvas = tk.Canvas(left_frame, bg='white')
        self.left_canvas.pack(fill=tk.BOTH, expand=True)
        
        right_frame = ttk.LabelFrame(img_frame, text="Результат (Предпросмотр)")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5,0))
        self.right_canvas = tk.Canvas(right_frame, bg='white')
        self.right_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Кнопки управления
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0,10))
        ttk.Button(btn_frame, text="📁 Открыть", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🔄 Сбросить всё", command=self.reset_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="💾 Сохранить", command=self.save_image).pack(side=tk.RIGHT, padx=5)
        
        # Вкладки инструментов
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.X, pady=(0,10))
        
        # 1. Перспектива
        f1 = ttk.Frame(self.notebook)
        self.notebook.add(f1, text="🔲 Перспектива")
        ttk.Button(f1, text="✋ ВКЛ/ВЫКЛ Режим разметки", command=self.toggle_manual_mode).pack(pady=10)
        ttk.Label(f1, text="Поставьте 4 точки по углам: ЛевВерх -> ПравВерх -> ПравНиз -> ЛевНиз").pack()

        # 2. Дисторсия
        f2 = ttk.Frame(self.notebook)
        self.notebook.add(f2, text="🔄 Дисторсия")
        ttk.Button(f2, text="🌐 Показать/Скрыть сетку", command=self.toggle_grid).pack(pady=5)
        c2 = ttk.Frame(f2)
        c2.pack(fill=tk.X, padx=20)
        
        # Увеличиваем диапазон k1 до -1.5 (для сверхширокоугольных линз)
        # И k2 до 0.5 для компенсации сложных искажений
        self.dist_k1_scale = self._create_scale(c2, "Кривизна k1:", -1.5, 0.5, 0.0, 0, self.update_distortion)
        self.dist_k2_scale = self._create_scale(c2, "Кривизна k2:", -0.5, 0.5, 0.0, 1, self.update_distortion)
        
        # Увеличиваем лимит масштаба до 2.0, так как при сильной дисторсии 
        # края сильно "уходят" внутрь и нужно больше места
        self.dist_scale_scale = self._create_scale(c2, "Масштаб:", 0.5, 2.0, 1.0, 2, self.update_distortion)

        # 3. Качество
        f3 = ttk.Frame(self.notebook)
        self.notebook.add(f3, text="🖼️ Качество")
        c3 = ttk.Frame(f3)
        c3.pack(fill=tk.X, padx=20, pady=10)
        self.quality_denoise_scale = self._create_scale(c3, "Шум:", 0.0, 3.0, 0.0, 0, self.update_quality_enhance)
        self.quality_contrast_scale = self._create_scale(c3, "Контраст:", 1.0, 3.0, 1.0, 1, self.update_quality_enhance)
        self.quality_sharpen_scale = self._create_scale(c3, "Резкость:", 0.0, 6.0, 0.0, 2, self.update_quality_enhance)

        # События
        self.left_canvas.bind("<Button-1>", self.on_left_click)
        self.left_canvas.bind("<B1-Motion>", self.on_left_drag)
        self.left_canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.left_canvas.bind("<Configure>", lambda e: self.update_displays())
        
        self.status_var = tk.StringVar(value="Готово")
        ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN).pack(side=tk.BOTTOM, fill=tk.X)

    def _create_scale(self, parent, label, f, t, d, row, cmd):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        s = ttk.Scale(parent, from_=f, to=t, command=cmd)
        s.set(d)
        s.grid(row=row, column=1, sticky="we", padx=(10,20), pady=2)
        parent.columnconfigure(1, weight=1)
        return s

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.reset_image()

    def reset_image(self):
        if self.original_img is not None:
            self.processed_img = self.original_img.copy()
            self.persp_points.clear()
            self.dist_k1_scale.set(0.0)
            self.dist_k2_scale.set(0.0)
            self.dist_scale_scale.set(1.0)
            self.quality_denoise_scale.set(0.0)
            self.quality_contrast_scale.set(1.0)
            self.quality_sharpen_scale.set(0.0)
            self.update_displays()
            self.update_status("🔄 Изображение сброшено")

    def toggle_manual_mode(self):
        self.manual_mode = not self.manual_mode
        self.left_canvas.config(cursor="crosshair" if self.manual_mode else "arrow")
        self.update_displays()

    def toggle_grid(self):
        self.show_grid = not self.show_grid
        self.update_displays()

    def on_left_click(self, event):
        if not self.manual_mode or self.original_img is None: return
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        for i, p in enumerate(self.persp_points):
            if np.sqrt((img_x-p[0])**2 + (img_y-p[1])**2) < 30:
                self.dragging_idx = i
                return
        if len(self.persp_points) < 4:
            self.persp_points.append([float(img_x), float(img_y)])
            self.update_manual_perspective()

    def on_left_drag(self, event):
        if self.dragging_idx is not None:
            self.persp_points[self.dragging_idx] = list(self.canvas_to_image_coords(event.x, event.y))
            self.update_manual_perspective()

    def on_left_release(self, event):
        self.dragging_idx = None

    def update_manual_perspective(self):
        if len(self.persp_points) == 4 and self.original_img is not None:
            h, w = self.original_img.shape[:2]
            src = np.float32(self.persp_points)
            dst = np.float32([[0,0], [w,0], [w,h], [0,h]])
            M = cv2.getPerspectiveTransform(src, dst)
            self.processed_img = cv2.warpPerspective(self.original_img, M, (w, h))
        self.update_displays()

    def update_distortion(self, val=None):
        if self.original_img is None: return
        k1, k2, scale = self.dist_k1_scale.get(), self.dist_k2_scale.get(), self.dist_scale_scale.get()
        h, w = self.original_img.shape[:2]
        K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
        D = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (int(w*scale), int(h*scale)), cv2.CV_32FC1)
        self.processed_img = cv2.remap(self.original_img, map1, map2, cv2.INTER_CUBIC)
        self.update_displays()

    def update_quality_enhance(self, val=None):
        if self.original_img is None: return
        den, con, shp = self.quality_denoise_scale.get(), self.quality_contrast_scale.get(), self.quality_sharpen_scale.get()
        img = self.original_img.copy()
        if den > 0.1:
            img = cv2.fastNlMeansDenoisingColored(img, None, int(den*10), 10, 7, 21)
        if con > 1.0:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=con, tileGridSize=(8,8)).apply(l)
            img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        if shp > 0:
            gauss = cv2.GaussianBlur(img, (0, 0), shp)
            img = cv2.addWeighted(img, 1.0 + shp, gauss, -shp, 0)
        self.processed_img = img
        self.update_displays()

    def canvas_to_image_coords(self, cx, cy):
        scale = self.get_canvas_scale()
        return (cx - self.image_offset_x) / scale, (cy - self.image_offset_y) / scale

    def get_canvas_scale(self):
        if self.original_img is None: return 1.0
        h, w = self.original_img.shape[:2]
        return min(self.left_canvas.winfo_width()/w, self.left_canvas.winfo_height()/h) * Params.CANVAS_SCALE

    def update_displays(self):
        if self.original_img is None: return
        # Центрирование
        scale = self.get_canvas_scale()
        self.image_offset_x = (self.left_canvas.winfo_width() - self.original_img.shape[1]*scale)//2
        self.image_offset_y = (self.left_canvas.winfo_height() - self.original_img.shape[0]*scale)//2
        
        # Левый экран
        img_l = cv2.resize(self.original_img, None, fx=scale, fy=scale)
        self.left_photo = ImageTk.PhotoImage(Image.fromarray(img_l))
        self.left_canvas.delete("all")
        self.left_canvas.create_image(self.left_canvas.winfo_width()//2, self.left_canvas.winfo_height()//2, image=self.left_photo)
        
        if self.manual_mode:
            for i, p in enumerate(self.persp_points):
                cx, cy = p[0]*scale + self.image_offset_x, p[1]*scale + self.image_offset_y
                self.left_canvas.create_oval(cx-6, cy-6, cx+6, cy+6, fill="red", outline="white")
                self.left_canvas.create_text(cx, cy-15, text=str(i+1), fill="red", font=("Arial", 10, "bold"))

        # Правый экран
        if self.processed_img is not None:
            ph, pw = self.processed_img.shape[:2]
            ps = min(self.right_canvas.winfo_width()/pw, self.right_canvas.winfo_height()/ph) * 0.95
            img_r = cv2.resize(self.processed_img, None, fx=ps, fy=ps)
            self.right_photo = ImageTk.PhotoImage(Image.fromarray(img_r))
            self.right_canvas.delete("all")
            self.right_canvas.create_image(self.right_canvas.winfo_width()//2, self.right_canvas.winfo_height()//2, image=self.right_photo)
            
            if self.show_grid:
                w_r, h_r = self.right_canvas.winfo_width(), self.right_canvas.winfo_height()
                for i in range(0, w_r, 40): self.right_canvas.create_line(i, 0, i, h_r, fill="cyan", dash=(2,2))
                for i in range(0, h_r, 40): self.right_canvas.create_line(0, i, w_r, i, fill="cyan", dash=(2,2))

    def save_image(self):
        if self.processed_img is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path: cv2.imwrite(path, cv2.cvtColor(self.processed_img, cv2.COLOR_RGB2BGR))

    def update_status(self, msg): self.status_var.set(msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCorrector(root)
    root.mainloop()
