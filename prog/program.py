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
        self.root.title("Аффинные преобразования (Optimized)")
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
        self.current_scale = 1.0

        self.dragging_idx = None
        self.dragging_group = [] # Список индексов точек для группового перемещения
        
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
        
        ttk.Label(tool_frame, text="Плотность:").pack(side=tk.LEFT, padx=(10, 2))
        self.grid_size_var = tk.IntVar(value=4)
        self.grid_spin = ttk.Spinbox(tool_frame, from_=1, to=7, width=5, textvariable=self.grid_size_var)
        self.grid_spin.pack(side=tk.LEFT, padx=5)

        # ВЫБОР РЕЖИМА СЕТКИ
        ttk.Label(tool_frame, text="Тип:").pack(side=tk.LEFT, padx=(10, 2))
        self.grid_mode = tk.StringVar(value="Треугольники")
        self.mode_combo = ttk.Combobox(tool_frame, textvariable=self.grid_mode, 
                                       values=["Треугольники", "Квадраты"], width=12, state="readonly")
        self.mode_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(tool_frame, text="📐 Разместить сетку", command=self.init_grid).pack(side=tk.LEFT, padx=5)
        ttk.Button(tool_frame, text="🔄 Сброс", command=self.reset_grid).pack(side=tk.LEFT, padx=5)
        ttk.Button(tool_frame, text="💾 Сохранить", command=self.save_image).pack(side=tk.LEFT, padx=5)

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
            else:
                messagebox.showerror("Ошибка", "не удалось загрузить изображение")

    def save_image(self):
        if self.processed_img is None:
            messagebox.showwarning("Внимание", "Нет обработанного изображения для сохранения")
            return
            
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All files", "*.*")],
            title="Сохранить результат как..."
        )
        
        if path:
            try:
                # Конвертируем обратно из RGB (в котором работаем) в BGR (для OpenCV)
                final_bgr = cv2.cvtColor(self.processed_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(path, final_bgr)
                messagebox.showinfo("Успех", f"Изображение успешно сохранено:\n{path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить изображение: {e}")

    def init_grid(self):
        if self.original_img is None: 
            messagebox.showwarning("Внимание", "Сначала загрузите изображение")
            return
        
        try:
            n = int(self.grid_size_var.get())
        except:
            n = 4
            
        if n > 7: n = 7
        self.grid_size_var.set(n)
            
        h, w = self.original_img.shape[:2]
        self.points = [[float(x), float(y)] for y in np.linspace(0, h, n + 1) for x in np.linspace(0, w, n + 1)]
        self.base_points = [p[:] for p in self.points]
        
        self.triangles = [] # Будем использовать это имя переменной как универсальное для ячеек
        cols = n + 1
        mode = self.grid_mode.get()

        for j in range(n):
            for i in range(n):
                p1, p2, p3, p4 = j*cols+i, j*cols+i+1, (j+1)*cols+i, (j+1)*cols+i+1
                if mode == "Треугольники":
                    self.triangles.append((p1, p2, p3))
                    self.triangles.append((p2, p4, p3))
                else: # Квадраты
                    self.triangles.append((p1, p2, p4, p3))
        
        self.process_warp()

    def process_warp(self, quality=cv2.INTER_LINEAR):
        if self.original_img is None or not self.points: return
        h, w = self.original_img.shape[:2]
        out_img = np.zeros_like(self.original_img)
        
        for cell in self.triangles:
            src_pts = np.float32([self.points[i] for i in cell])
            dst_pts = np.float32([self.base_points[i] for i in cell])
            
            x, y, w_b, h_b = cv2.boundingRect(dst_pts)
            dst_rect = dst_pts - (x, y)
            
            mask = np.zeros((h_b, w_b, 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(dst_rect), (1.0, 1.0, 1.0), 16, 0)
            
            sx, sy, sw, sh = cv2.boundingRect(src_pts)
            sx_end, sy_end = min(w, sx + sw), min(h, sy + sh)
            img_patch = self.original_img[sy:sy_end, sx:sx_end]
            if img_patch.size == 0: continue

            src_in_patch = src_pts - (sx, sy)
            
            # ВЫБОР ТИПА ТРАНСФОРМАЦИИ
            if len(cell) == 3:
                matrix = cv2.getAffineTransform(np.float32(src_in_patch), np.float32(dst_rect))
                warped_patch = cv2.warpAffine(img_patch, matrix, (w_b, h_b), flags=quality, borderMode=cv2.BORDER_REFLECT_101)
            else:
                matrix = cv2.getPerspectiveTransform(np.float32(src_in_patch), np.float32(dst_rect))
                warped_patch = cv2.warpPerspective(img_patch, matrix, (w_b, h_b), flags=quality, borderMode=cv2.BORDER_REFLECT_101)
            
            slice_y, slice_x = slice(y, min(y + h_b, h)), slice(x, min(x + w_b, w))
            ph, pw = out_img[slice_y, slice_x].shape[:2]
            
            out_img[slice_y, slice_x] = (out_img[slice_y, slice_x] * (1 - mask[:ph, :pw]) + 
                                         warped_patch[:ph, :pw] * mask[:ph, :pw]).astype(np.uint8)
            
        self.processed_img = out_img
        self.update_displays()

    def get_scale(self):
        if self.original_img is None: return 1.0
        h, w = self.original_img.shape[:2]
        cw, ch = self.left_canvas.winfo_width(), self.left_canvas.winfo_height()
        if cw < 10 or ch < 10: return 1.0
        # Убрали магический 0.95 здесь, перенесли логику в update_displays
        return min(cw/w, ch/h)

    def update_displays(self):
        if self.original_img is None: return
        raw_scale = self.get_scale()
        self.current_scale = raw_scale * 0.95 
        h, w = self.original_img.shape[:2]
        sw, sh = int(w * self.current_scale), int(h * self.current_scale)
        self.image_offset_x = (self.left_canvas.winfo_width() - sw) // 2
        self.image_offset_y = (self.left_canvas.winfo_height() - sh) // 2

        self.left_photo = ImageTk.PhotoImage(Image.fromarray(cv2.resize(self.original_img, (sw, sh))))
        self.left_canvas.delete("all")
        self.left_canvas.create_image(self.left_canvas.winfo_width()//2, self.left_canvas.winfo_height()//2, image=self.left_photo)
        
        if self.points:
            for cell in self.triangles: # Рисуем хоть 3, хоть 4 угла
                pts_coords = []
                for idx in cell:
                    p = np.array(self.points[idx])
                    pts_coords.extend([p[0] * self.current_scale + self.image_offset_x, 
                                       p[1] * self.current_scale + self.image_offset_y])
                self.left_canvas.create_polygon(pts_coords, fill="", outline="#00ffcc", width=1)
            
            for p in self.points:
                sx, sy = p[0] * self.current_scale + self.image_offset_x, p[1] * self.current_scale + self.image_offset_y
                self.left_canvas.create_oval(sx-4, sy-4, sx+4, sy+4, fill="#ff3366", outline="white")

        if self.processed_img is not None:
            self.right_photo = ImageTk.PhotoImage(Image.fromarray(cv2.resize(self.processed_img, (sw, sh))))
            self.right_canvas.delete("all")
            self.right_canvas.create_image(self.right_canvas.winfo_width()//2, self.right_canvas.winfo_height()//2, image=self.right_photo)

    def on_click(self, event):
        if not self.points: return
        self.dragging_idx = None
        self.dragging_group = []
        
        # Константы чувствительности (в пикселях экрана)
        POINT_TOLERANCE = 15 
        LINE_TOLERANCE = 8

        # 1. Проверка вершин (Приоритет 1)
        for i, p in enumerate(self.points):
            sx = p[0] * self.current_scale + self.image_offset_x
            sy = p[1] * self.current_scale + self.image_offset_y
            if abs(sx - event.x) < POINT_TOLERANCE and abs(sy - event.y) < POINT_TOLERANCE:
                self.dragging_idx = i
                return

        # 2. Проверка граней (Приоритет 2)
        for cell in self.triangles:
            n = len(cell)
            for i in range(n):
                idx1, idx2 = cell[i], cell[(i + 1) % n]
                p1 = np.array([self.points[idx1][0] * self.current_scale + self.image_offset_x,
                               self.points[idx1][1] * self.current_scale + self.image_offset_y])
                p2 = np.array([self.points[idx2][0] * self.current_scale + self.image_offset_x,
                               self.points[idx2][1] * self.current_scale + self.image_offset_y])
                
                # Вычисляем расстояние от точки (клика) до отрезка p1-p2
                mouse = np.array([event.x, event.y])
                line_vec = p2 - p1
                mouse_vec = mouse - p1
                line_len = np.linalg.norm(line_vec)
                if line_len == 0: continue
                
                t = max(0, min(1, np.dot(mouse_vec, line_vec) / (line_len**2)))
                projection = p1 + t * line_vec
                dist = np.linalg.norm(mouse - projection)

                if dist < LINE_TOLERANCE:
                    self.dragging_group = [idx1, idx2]
                    self.last_mouse_x, self.last_mouse_y = event.x, event.y
                    return

        # 3. Проверка центра ячейки (Приоритет 3)
        for cell in self.triangles:
            poly = []
            for idx in cell:
                poly.append([self.points[idx][0] * self.current_scale + self.image_offset_x,
                             self.points[idx][1] * self.current_scale + self.image_offset_y])
            
            if cv2.pointPolygonTest(np.array(poly, dtype=np.int32), (event.x, event.y), False) >= 0:
                self.dragging_group = list(cell)
                self.last_mouse_x, self.last_mouse_y = event.x, event.y
                return

    def on_drag(self, event):
        if self.dragging_idx is None and not self.dragging_group: return
        
        curr_time = time.time()
        if curr_time - self.last_update_time < 0.016: return 
        self.last_update_time = curr_time

        h, w = self.original_img.shape[:2]

        if self.dragging_idx is not None:
            # Тянем одну вершину
            nx = (event.x - self.image_offset_x) / self.current_scale
            ny = (event.y - self.image_offset_y) / self.current_scale
            self.points[self.dragging_idx] = [max(0, min(w, nx)), max(0, min(h, ny))]
        
        elif self.dragging_group:
            # Тянем группу (грань или всю фигуру)
            dx = (event.x - self.last_mouse_x) / self.current_scale
            dy = (event.y - self.last_mouse_y) / self.current_scale
            
            for idx in self.dragging_group:
                new_x = self.points[idx][0] + dx
                new_y = self.points[idx][1] + dy
                self.points[idx] = [max(0, min(w, new_x)), max(0, min(h, new_y))]
            
            self.last_mouse_x, self.last_mouse_y = event.x, event.y

        self.process_warp(quality=cv2.INTER_NEAREST)

    def on_release(self, event):
        if self.dragging_idx is not None or self.dragging_group:
            self.dragging_idx = None
            self.dragging_group = []
            self.process_warp(quality=cv2.INTER_LINEAR)

    def reset_grid(self):
        if self.base_points:
            self.points = [p[:] for p in self.base_points]
            self.process_warp()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCorrector(root)
    root.mainloop()
