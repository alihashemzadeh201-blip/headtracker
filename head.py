import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import customtkinter as ctk
from PIL import Image
import tkinter as tk

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]

L_EYE_LEFT = 33
L_EYE_RIGHT = 133
L_EYE_TOP = 159
L_EYE_BOTTOM = 145

R_EYE_LEFT = 362
R_EYE_RIGHT = 263
R_EYE_TOP = 386
R_EYE_BOTTOM = 374

def eye_aspect_ratio(landmarks, eye_indices):
    try:
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
        vert1 = ((p2.x - p6.x)**2 + (p2.y - p6.y)**2) ** 0.5
        vert2 = ((p3.x - p5.x)**2 + (p3.y - p5.y)**2) ** 0.5
        horiz = ((p1.x - p4.x)**2 + (p1.y - p4.y)**2) ** 0.5
        if horiz == 0: return 0
        return (vert1 + vert2) / (2.0 * horiz)
    except: return 0

class HeadTrackerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Face & Eye Tracker - All in One")
        self.geometry("1000x650")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.is_active = False
        self.center_nx = None
        self.center_ny = None
        self.last_click_time = 0
        self.is_blinking = False
        self.smooth_mouse_x = screen_w / 2
        self.smooth_mouse_y = screen_h / 2
        
        self.calibrating = False
        self.calib_points = [(0, 0), (screen_w, 0), (screen_w, screen_h), (0, screen_h)]
        self.calib_index = 0
        self.calib_window = None
        self.calib_canvas = None
        self.calib_timer = 0
        self.calib_data_x = []
        self.calib_data_y = []
        
        self.min_gaze_x, self.max_gaze_x = 0.3, 0.7
        self.min_gaze_y, self.max_gaze_y = 0.3, 0.7
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.cap = cv2.VideoCapture(0)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0, minsize=350)
        self.grid_rowconfigure(0, weight=1)
        
        self.cam_frame = ctk.CTkFrame(self, corner_radius=10)
        self.cam_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.lbl_cam = ctk.CTkLabel(self.cam_frame, text="Loading Camera...")
        self.lbl_cam.pack(expand=True, fill="both", padx=5, pady=5)
        
        self.bind("<KeyPress-e>", lambda e: self.toggle_active())
        self.bind("<KeyPress-E>", lambda e: self.toggle_active())

        self.settings_frame = ctk.CTkFrame(self, corner_radius=10)
        self.settings_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        lbl_title = ctk.CTkLabel(self.settings_frame, text="Settings", font=ctk.CTkFont(size=20, weight="bold"))
        lbl_title.pack(pady=10)
        
        self.btn_toggle = ctk.CTkButton(self.settings_frame, text="ENABLE MOUSE (E)", 
                                        fg_color="#C2185B", hover_color="#9C1448", height=40,
                                        command=self.toggle_active)
        self.btn_toggle.pack(fill="x", padx=20, pady=(0, 15))
        
        self.scroll = ctk.CTkScrollableFrame(self.settings_frame, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=10)
        
        self.track_mode = ctk.StringVar(value="Head Tracking")
        ctk.CTkRadioButton(self.scroll, text="Head Tracking (Joystick)", variable=self.track_mode, value="Head Tracking").pack(anchor="w", pady=5)
        ctk.CTkRadioButton(self.scroll, text="Eye Tracking (Absolute)", variable=self.track_mode, value="Eye Tracking").pack(anchor="w", pady=5)

        self.btn_calib = ctk.CTkButton(self.scroll, text="Calibrate Eye Tracking", command=self.start_calibration)
        self.btn_calib.pack(fill="x", pady=10)

        self.sens_var = ctk.DoubleVar(value=0.3)
        self.dz_var = ctk.DoubleVar(value=0.015)
        self.eye_smooth_var = ctk.DoubleVar(value=0.85)
        self.bc_var = ctk.DoubleVar(value=0.22)
        self.bo_var = ctk.DoubleVar(value=0.26)
        self.cd_var = ctk.DoubleVar(value=0.8)
        
        self.sw_wink = ctk.BooleanVar(value=True)
        self.sw_mirror = ctk.BooleanVar(value=True)
        self.sw_inv_y = ctk.BooleanVar(value=False)
        self.sw_face_zoom = ctk.BooleanVar(value=True)

        self.add_slider("Head Sensitivity:", self.sens_var, 0.01, 1.0)
        self.add_slider("Head Deadzone:", self.dz_var, 0.001, 0.1)
        self.add_slider("Eye Smoothing:", self.eye_smooth_var, 0.1, 0.99)
        self.add_slider("Wink (Close):", self.bc_var, 0.1, 0.4)
        self.add_slider("Wink (Open):", self.bo_var, 0.15, 0.5)
        self.add_slider("Cooldown (s):", self.cd_var, 0.1, 3.0)

        ctk.CTkSwitch(self.scroll, text="Auto Face Zoom", variable=self.sw_face_zoom).pack(anchor="w", pady=5)
        ctk.CTkSwitch(self.scroll, text="Enable Wink Click", variable=self.sw_wink).pack(anchor="w", pady=5)
        ctk.CTkSwitch(self.scroll, text="Mirror Camera", variable=self.sw_mirror).pack(anchor="w", pady=5)
        ctk.CTkSwitch(self.scroll, text="Invert Y Axis", variable=self.sw_inv_y).pack(anchor="w", pady=5)
        
        self.update_frame()

    def add_slider(self, text, variable, vmin, vmax):
        lbl = ctk.CTkLabel(self.scroll, text=text)
        lbl.pack(anchor="w", pady=(10,0))
        def update_label(val):
            lbl.configure(text=f"{text} {float(val):.3f}")
        sld = ctk.CTkSlider(self.scroll, from_=vmin, to=vmax, variable=variable, command=update_label)
        sld.pack(fill="x", pady=(0,5))
        update_label(variable.get())
        
    def toggle_active(self):
        self.is_active = not self.is_active
        if self.is_active:
            self.btn_toggle.configure(text="DISABLE MOUSE (E)", fg_color="#388E3C", hover_color="#2E7D32")
        else:
            self.btn_toggle.configure(text="ENABLE MOUSE (E)", fg_color="#C2185B", hover_color="#9C1448")
            self.center_nx = None
            self.center_ny = None

    def start_calibration(self):
        if self.calibrating: return
        self.calibrating = True
        self.calib_index = 0
        self.calib_data_x.clear()
        self.calib_data_y.clear()
        
        self.calib_window = tk.Toplevel(self)
        self.calib_window.attributes("-fullscreen", True)
        self.calib_window.configure(bg="black")
        self.calib_window.attributes("-topmost", True)
        self.calib_canvas = tk.Canvas(self.calib_window, width=screen_w, height=screen_h, bg="black", highlightthickness=0)
        self.calib_canvas.pack()
        
        self.calib_timer = time.time()
        self.calib_step()

    def calib_step(self):
        if not self.calibrating: return
        
        self.calib_canvas.delete("all")
        if self.calib_index < len(self.calib_points):
            cx, cy = self.calib_points[self.calib_index]
            r = 30
            if cx == 0: cx += r + 10
            elif cx == screen_w: cx -= r + 10
            if cy == 0: cy += r + 10
            elif cy == screen_h: cy -= r + 10
                
            self.calib_canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill="red")
            self.calib_canvas.create_text(screen_w/2, screen_h/2, text="Look at the RED DOT", fill="white", font=("Arial", 30))
            
            if time.time() - self.calib_timer > 2.0:
                self.calib_index += 1
                self.calib_timer = time.time()
            self.after(50, self.calib_step)
        else:
            if len(self.calib_data_x) > 0 and len(self.calib_data_y) > 0:
                self.min_gaze_x = np.percentile(self.calib_data_x, 5)
                self.max_gaze_x = np.percentile(self.calib_data_x, 95)
                self.min_gaze_y = np.percentile(self.calib_data_y, 5)
                self.max_gaze_y = np.percentile(self.calib_data_y, 95)
            
            self.calibrating = False
            self.calib_window.destroy()

    def get_eye_gaze_ratio(self, landmarks, side):
        if side == "left":
            left_pt, right_pt = landmarks[L_EYE_LEFT], landmarks[L_EYE_RIGHT]
            top_pt, bot_pt = landmarks[L_EYE_TOP], landmarks[L_EYE_BOTTOM]
            iris_indices = LEFT_IRIS
        else:
            left_pt, right_pt = landmarks[R_EYE_LEFT], landmarks[R_EYE_RIGHT]
            top_pt, bot_pt = landmarks[R_EYE_TOP], landmarks[R_EYE_BOTTOM]
            iris_indices = RIGHT_IRIS
            
        iris_pts = [landmarks[i] for i in iris_indices]
        iris_center_x = sum([p.x for p in iris_pts]) / 4
        iris_center_y = sum([p.y for p in iris_pts]) / 4
        
        eye_width = right_pt.x - left_pt.x
        eye_height = bot_pt.y - top_pt.y
        if eye_width == 0 or eye_height == 0: return 0.5, 0.5
        
        ratio_x = (iris_center_x - left_pt.x) / eye_width
        ratio_y = (iris_center_y - top_pt.y) / eye_height
        
        return ratio_x, ratio_y

    def update_frame(self):
        if not self.cap.isOpened(): return
        success, frame = self.cap.read()
        if not success:
            self.after(10, self.update_frame)
            return

        current_time = time.time()
        if self.sw_mirror.get(): frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        crop_box = None
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                nx, ny = landmarks[4].x, landmarks[4].y
                
                if self.sw_face_zoom.get():
                    xs = [lm.x for lm in landmarks]
                    ys = [lm.y for lm in landmarks]
                    min_x, max_x = max(0, min(xs)), min(1, max(xs))
                    min_y, max_y = max(0, min(ys)), min(1, max(ys))
                    
                    pad_x, pad_y = 0.05, 0.05
                    crop_box = (
                        int(max(0, (min_x - pad_x) * w)),
                        int(max(0, (min_y - pad_y) * h)),
                        int(min(w, (max_x + pad_x) * w)),
                        int(min(h, (max_y + pad_y) * h))
                    )
                
                if self.track_mode.get() == "Eye Tracking":
                    for idx in LEFT_IRIS + RIGHT_IRIS:
                        cv2.circle(rgb_frame, (int(landmarks[idx].x * w), int(landmarks[idx].y * h)), 1, (255, 255, 0), -1)

                dot_color = (0, 255, 0) if self.is_active else (255, 0, 0)
                cv2.circle(rgb_frame, (int(nx * w), int(ny * h)), 6, dot_color, -1)

                lx, ly = self.get_eye_gaze_ratio(landmarks, "left")
                rx, ry = self.get_eye_gaze_ratio(landmarks, "right")
                avg_x, avg_y = (lx + rx) / 2, (ly + ry) / 2

                if self.calibrating:
                    if time.time() - self.calib_timer > 0.5:
                        self.calib_data_x.append(avg_x)
                        self.calib_data_y.append(avg_y)

                if self.is_active and not self.calibrating:
                    mode = self.track_mode.get()
                    if mode == "Head Tracking":
                        if self.center_nx is None or self.center_ny is None:
                            self.center_nx, self.center_ny = nx, ny
                        cpx, cpy = int(self.center_nx * w), int(self.center_ny * h)
                        npx, npy = int(nx * w), int(ny * h)
                        cv2.circle(rgb_frame, (cpx, cpy), 4, (0, 100, 255), -1)
                        cv2.line(rgb_frame, (cpx, cpy), (npx, npy), (0, 100, 255), 1)

                        dz = self.dz_var.get()
                        dx, dy = nx - self.center_nx, ny - self.center_ny
                        dx = np.sign(dx) * (abs(dx) - dz) if abs(dx) > dz else 0
                        dy = np.sign(dy) * (abs(dy) - dz) if abs(dy) > dz else 0
                        if self.sw_inv_y.get(): dy = -dy

                        sens = self.sens_var.get()
                        move_x, move_y = dx * screen_w * sens, dy * screen_h * sens
                        if abs(move_x) > 0 or abs(move_y) > 0:
                            try: pyautogui.moveRel(int(move_x), int(move_y), _pause=False)
                            except: pass
                            
                    elif mode == "Eye Tracking":
                        mapped_x = (avg_x - self.min_gaze_x) / (self.max_gaze_x - self.min_gaze_x + 0.0001)
                        mapped_y = (avg_y - self.min_gaze_y) / (self.max_gaze_y - self.min_gaze_y + 0.0001)
                        
                        mapped_x = np.clip(mapped_x, 0, 1)
                        mapped_y = np.clip(mapped_y, 0, 1)

                        target_mouse_x = mapped_x * screen_w
                        target_mouse_y = mapped_y * screen_h
                        if self.sw_inv_y.get(): target_mouse_y = screen_h - target_mouse_y
                        
                        smooth_factor = self.eye_smooth_var.get()
                        self.smooth_mouse_x = (smooth_factor * self.smooth_mouse_x) + ((1 - smooth_factor) * target_mouse_x)
                        self.smooth_mouse_y = (smooth_factor * self.smooth_mouse_y) + ((1 - smooth_factor) * target_mouse_y)
                        
                        try: pyautogui.moveTo(int(self.smooth_mouse_x), int(self.smooth_mouse_y), _pause=False)
                        except: pass

                    if self.sw_wink.get():
                        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                        bc, bo = self.bc_var.get(), self.bo_var.get()
                        is_left = left_ear < bc and right_ear > bo
                        is_right = right_ear < bc and left_ear > bo
                        
                        if is_left or is_right:
                            if not self.is_blinking and (current_time - self.last_click_time > self.cd_var.get()):
                                try:
                                    pyautogui.click(button="left")
                                    self.is_blinking = True
                                    self.last_click_time = current_time
                                except: pass
                        else: self.is_blinking = False
                        
                        ec = (255, 255, 0) if self.is_blinking else (255, 255, 255)
                        cv2.putText(rgb_frame, f"L:{left_ear:.2f} R:{right_ear:.2f} {'[CLICK]' if self.is_blinking else ''}",
                                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ec, 2)
                else:
                    self.center_nx = self.center_ny = None
                    self.is_blinking = False
        else:
            self.center_nx = self.center_ny = None
            cv2.putText(rgb_frame, "Face not detected", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        if crop_box is not None:
            rgb_frame = rgb_frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]

        img = Image.fromarray(rgb_frame)
        frame_w, frame_h = self.cam_frame.winfo_width(), self.cam_frame.winfo_height()
        
        if frame_w > 10 and frame_h > 10:
            aspect = img.width / img.height
            new_w, new_h = frame_w, int(frame_w / aspect)
            if new_h > frame_h:
                new_h, new_w = frame_h, int(frame_h * aspect)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
        self.lbl_cam.configure(image=ctk_img, text="")
        self.after(10, self.update_frame)

    def on_closing(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = HeadTrackerApp()
    app.mainloop()
