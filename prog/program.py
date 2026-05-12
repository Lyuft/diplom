# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2, numpy as np
from PIL import Image, ImageTk
from typing import List, Optional

class ImageCorrector:
    def __init__(self, root):
        self.root = root
        self.root.title("Аффинные преобразования — Диплом")
        self.root.geometry("1400x900")
        
        self.original_img = self.processed_img = None
        self.points, self.base_points, self.triangles = [], [], []
        self.undo_stack, self.redo_stack = [], []
        self.selected_indices = set()
        self.dragging_group = []
        self.current_scale = 1.0
        self.image_offset_x = self.image_offset_y = 0
        self.first_shift_point = None
        
        # ДОБАВЬТЕ ЭТИ СТРОКИ:
        self.is_scaling = False  # Флаг режима масштабирования
        self.is_alt = False      # Флаг нажатия Alt
        
        self.setup_ui()
        self.bind_events()

    def setup_ui(self):
        # Основной контейнер
        main = tk.Frame(self.root, bg="#1e1e1e")
        main.pack(fill=tk.BOTH, expand=True)

        # Панель инструментов (нижняя часть)
        tools = tk.Frame(main, bg="#2d2d2d", pady=5)
        tools.pack(side=tk.BOTTOM, fill=tk.X)
        
        cmds = [("Открыть фото", self.load_image, "#3c3c3c"), 
                ("Назад", self.undo, "#3c3c3c"), 
                ("Вперед", self.redo, "#3c3c3c"), 
                ("Сброс изменений", self.reset_grid, "#ff3366")]
        
        for t, c, bg in cmds:
            tk.Button(tools, text=t, command=c, bg=bg, fg="white", 
                      relief=tk.FLAT, padx=10).pack(side=tk.LEFT, padx=5)

        self.grid_size_var = tk.IntVar(value=4)
        tk.Spinbox(tools, from_=1, to=15, width=5, textvariable=self.grid_size_var).pack(side=tk.LEFT, padx=5)
        
        # Кнопка создания сетки (исправлен параметр шрифта)
        tk.Button(tools, text="Создать сетку", command=self.init_grid, 
                  bg="#15ff00", fg="black", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        
        tk.Button(tools, text="Сохранить результат", command=self.save_image, 
                  bg="#007acc", fg="white").pack(side=tk.RIGHT, padx=10)

        # Правая панель с инструкцией
        info_panel = tk.Frame(main, bg="#252526", width=260, padx=15, pady=15)
        info_panel.pack(side=tk.RIGHT, fill=tk.Y)
        info_panel.pack_propagate(False)

        tk.Label(info_panel, text="УПРАВЛЕНИЕ", fg="#15ff00", bg="#252526", 
                 font=("Arial", 11, "bold")).pack(pady=(0, 10), anchor="w")
        
        help_text = (
            "ЛКМ по точке: Перемещение\n"
            "ЛКМ по грани: Выбор грани\n"
            "ЛКМ внутри треугольника: Выбор области\n\n"
            "Ctrl + ЛКМ: Мультивыбор\n"
            "Shift + ЛКМ: Диапазон\n\n"
            "Горячие клавиши:\n"
            "Ctrl + A: Выбрать всё\n"
            "Alt + Колесо: Масштаб\n"
            "Ctrl + Z / Y: Отмена/Повтор\n"
            "R: Сброс сетки\n"
        )
        
        tk.Label(info_panel, text=help_text, fg="#cccccc", bg="#252526", 
                 justify=tk.LEFT, font=("Arial", 10), wraplength=230).pack(anchor="w")

        # Область с холстами
        self.canvases = []
        titles = ["Исходное изображение", "Результат коррекции"]
        for title in titles:
            f = tk.LabelFrame(main, text=f" {title} ", fg="#888", bg="#1e1e1e", font=("Arial", 9))
            f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=5)
            c = tk.Canvas(f, bg="#2b2b2b", highlightthickness=0)
            c.pack(fill=tk.BOTH, expand=True)
            self.canvases.append(c)
        
        self.left_canvas, self.right_canvas = self.canvases

    def bind_events(self):
        self.left_canvas.bind("<Button-1>", self.on_click)
        self.left_canvas.bind("<B1-Motion>", self.on_drag)
        self.left_canvas.bind("<MouseWheel>", self.on_wheel_scaling)
        
        # Горячие клавиши
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Control-a>", lambda e: self.select_all())
        
        # Биндим клавишу R (регистронезависимо через <Key-r>)
        self.root.bind("<Key-r>", lambda e: self.reset_grid())
        
        self.left_canvas.bind("<Configure>", lambda e: self.update_displays())

    def img_to_canvas(self, p):
        return (p[0] * self.current_scale + self.image_offset_x, 
                p[1] * self.current_scale + self.image_offset_y)

    def canvas_to_img(self, x, y):
        return ((x - self.image_offset_x) / self.current_scale, 
                (y - self.image_offset_y) / self.current_scale)

    def save_state(self):
        if self.points:
            self.undo_stack.append([p[:] for p in self.points])
            if len(self.undo_stack) > 50: self.undo_stack.pop(0)
            self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append([p[:] for p in self.points])
            self.points = self.undo_stack.pop()
            self.process_warp()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append([p[:] for p in self.points])
            self.points = self.redo_stack.pop()
            self.process_warp()

    def init_grid(self):
        if self.original_img is None: return
        self.save_state()
        n = self.grid_size_var.get()
        h, w = self.original_img.shape[:2]
        self.points = [[float(x), float(y)] for y in np.linspace(0, h, n + 1) for x in np.linspace(0, w, n + 1)]
        self.base_points = [p[:] for p in self.points]
        self.triangles = []
        c = n + 1
        for j in range(n):
            for i in range(n):
                p1, p2, p3, p4 = j*c+i, j*c+i+1, (j+1)*c+i, (j+1)*c+i+1
                self.triangles.extend([(p1, p2, p3), (p2, p4, p3)])
        self.process_warp()

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            img = cv2.imread(path)
            self.original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.processed_img = self.original_img.copy()
            self.points, self.undo_stack, self.redo_stack = [], [], []
            self.update_displays()

    def process_warp(self, q=cv2.INTER_LINEAR):
        if self.original_img is None or not self.points: return
        h, w = self.original_img.shape[:2]
        out = np.zeros_like(self.original_img)
        for tri in self.triangles:
            s_pts, d_pts = np.float32([self.points[i] for i in tri]), np.float32([self.base_points[i] for i in tri])
            x, y, wb, hb = cv2.boundingRect(d_pts)
            d_rect = d_pts - (x, y)
            mask = np.zeros((hb, wb, 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(d_rect), (1, 1, 1), 16, 0)
            
            sx, sy, sw, sh = cv2.boundingRect(s_pts)
            patch = self.original_img[max(0, sy):min(h, sy+sh), max(0, sx):min(w, sx+sw)]
            if patch.size == 0: continue
            
            m = cv2.getAffineTransform(np.float32(s_pts - (sx, sy)), np.float32(d_rect))
            warped = cv2.warpAffine(patch, m, (wb, hb), flags=q, borderMode=cv2.BORDER_REFLECT_101)
            
            sy_sl, sx_sl = slice(max(0, y), min(y+hb, h)), slice(max(0, x), min(x+wb, w))
            ph, pw = out[sy_sl, sx_sl].shape[:2]
            out[sy_sl, sx_sl] = (out[sy_sl, sx_sl] * (1 - mask[:ph, :pw]) + warped[:ph, :pw] * mask[:ph, :pw]).astype(np.uint8)
        self.processed_img = out
        self.update_displays()

    def update_displays(self):
        if self.original_img is None: return
        cw, ch = self.left_canvas.winfo_width(), self.left_canvas.winfo_height()
        h, w = self.original_img.shape[:2]
        self.current_scale = min(cw/w, ch/h) * 0.95
        sw, sh = int(w * self.current_scale), int(h * self.current_scale)
        self.image_offset_x, self.image_offset_y = (cw-sw)//2, (ch-sh)//2

        for canv, img in [(self.left_canvas, self.original_img), (self.right_canvas, self.processed_img)]:
            p_img = ImageTk.PhotoImage(Image.fromarray(cv2.resize(img, (sw, sh))))
            setattr(self, f"{canv}_img", p_img)
            canv.delete("all")
            canv.create_image(cw//2, ch//2, image=p_img)

        for tri in self.triangles:
            pts = [c for i in tri for c in self.img_to_canvas(self.points[i])]
            self.left_canvas.create_polygon(pts, fill="", outline="#15ff00")

        for i, p in enumerate(self.points):
            sx, sy = self.img_to_canvas(p)
            sel = i in self.selected_indices
            r = 5 if sel else 4
            self.left_canvas.create_oval(sx-r, sy-r, sx+r, sy+r, fill="#f3ff00" if sel else "#ff3366", outline="white")

    def on_click(self, e):
        self.left_canvas.focus_set()
        if not self.points: return
        self.save_state()
        
        # В Windows Alt часто определяется маской 0x20000 или состоянием 131072, 
        # но стандартная проверка (e.state & 0x0020) обычно достаточна.
        mx, my = self.canvas_to_img(e.x, e.y)
        
        # 1. Поиск ближайшей точки (приоритет)
        clicked_idx = next((i for i, p in enumerate(self.points) 
                           if np.hypot(p[0]-mx, p[1]-my) < 15/self.current_scale), None)
        
        if clicked_idx is not None:
            if e.state & 0x0004: # Ctrl зажат
                self.selected_indices.symmetric_difference_update([clicked_idx])
            elif e.state & 0x0001 and self.first_shift_point is not None: # Shift зажат
                for i in range(min(self.first_shift_point, clicked_idx), max(self.first_shift_point, clicked_idx)+1):
                    self.selected_indices.add(i)
            else:
                if clicked_idx not in self.selected_indices: 
                    self.selected_indices = {clicked_idx}
            self.first_shift_point = clicked_idx
        
        else:
            # 2. Поиск грани (ребра)
            edge_found = False
            for tri in self.triangles:
                for i in range(3):
                    p1_idx, p2_idx = tri[i], tri[(i+1)%3]
                    p1, p2 = np.array(self.points[p1_idx]), np.array(self.points[p2_idx])
                    
                    # Расстояние от клика до прямой, проходящей через p1, p2
                    line_vec = p2 - p1
                    p_vec = np.array([mx, my]) - p1
                    line_len = np.linalg.norm(line_vec)
                    if line_len == 0: continue
                    
                    unit_line = line_vec / line_len
                    projection = np.dot(p_vec, unit_line)
                    
                    if 0 <= projection <= line_len:
                        dist = np.linalg.norm(p_vec - projection * unit_line)
                        if dist < 8 / self.current_scale: # Порог попадания в линию
                            self.selected_indices = {p1_idx, p2_idx}
                            edge_found = True
                            break
                if edge_found: break

            # 3. Поиск треугольника (если не попали в точку или ребро)
            if not edge_found:
                for tri in self.triangles:
                    pts = [np.array(self.points[i]) for i in tri]
                    # Проверка через барицентрические координаты
                    v0, v1, v2 = pts[1]-pts[0], pts[2]-pts[0], np.array([mx, my])-pts[0]
                    det = (v0[0]*v1[1] - v0[1]*v1[0])
                    if abs(det) < 1e-9: continue
                    u = (v2[0]*v1[1] - v1[0]*v2[1]) / det
                    v = (v0[0]*v2[1] - v2[0]*v0[1]) / det
                    if u >= 0 and v >= 0 and (u + v) <= 1:
                        self.selected_indices = set(tri)
                        break
                else:
                    # Если кликнули в пустоту без Ctrl/Shift — сбрасываем выделение
                    if not (e.state & 0x0005): 
                        self.selected_indices.clear()

        self.dragging_group = list(self.selected_indices)
        self.last_mx, self.last_my = mx, my
        self.update_displays()

    def on_drag(self, e):
        if not self.dragging_group: return
        mx, my = self.canvas_to_img(e.x, e.y)
        h, w = self.original_img.shape[:2]
        
        if self.is_scaling:
            k = np.hypot(mx-self.scale_origin[0], my-self.scale_origin[1]) / self.start_dist
            init_pts = self.undo_stack[-1]
            for i in self.selected_indices:
                new_p = self.scale_origin + (np.array(init_pts[i]) - self.scale_origin) * k
                self.points[i] = [np.clip(new_p[0], 0, w), np.clip(new_p[1], 0, h)]
        else:
            dx, dy = mx - self.last_mx, my - self.last_my
            for i in self.dragging_group:
                self.points[i][0] = np.clip(self.points[i][0] + dx, 0, w)
                self.points[i][1] = np.clip(self.points[i][1] + dy, 0, h)
            self.last_mx, self.last_my = mx, my
        self.process_warp(q=cv2.INTER_NEAREST)

    def on_wheel_scaling(self, e):
        # В Windows Alt может приходить как 131072 или иметь бит 0x20000
        # Проверяем стандартный 0x0020 и расширенный вариант
        is_alt_pressed = (e.state & 0x0020) != 0 or (e.state & 131072) != 0
        
        if not is_alt_pressed or not self.selected_indices: 
            return
        
        self.save_state()
        
        # Для Windows: e.delta > 0 — крутим от себя (увеличение)
        k = 1.05 if e.delta > 0 else 0.95
        
        # Центр масштабирования — среднее арифметическое выбранных точек
        pts_coords = np.array([self.points[i] for i in self.selected_indices])
        center = np.mean(pts_coords, axis=0)
        
        for i in self.selected_indices:
            p = np.array(self.points[i])
            # Формула: Новая_точка = Центр + (Вектор_до_точки * Коэффициент)
            new_p = center + (p - center) * k
            
            # Ограничиваем, чтобы точки не улетали за края картинки
            h, w = self.original_img.shape[:2]
            self.points[i] = [np.clip(new_p[0], 0, w), np.clip(new_p[1], 0, h)]
            
        self.process_warp()

    def reset_grid(self):
        if self.base_points:
            self.save_state()
            self.points = [p[:] for p in self.base_points]
            self.selected_indices.clear()
            self.process_warp()

    def save_image(self):
        if self.processed_img is None: return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path: cv2.imwrite(path, cv2.cvtColor(self.processed_img, cv2.COLOR_RGB2BGR))

    def select_all(self):
        self.selected_indices = set(range(len(self.points)))
        self.update_displays()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCorrector(root)
    root.mainloop()