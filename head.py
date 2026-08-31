import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import customtkinter as ctk
from PIL import Image

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

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
        
        self.title("HeadTracker - All in One")
        self.geometry("1000x550")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.is_active = False
        self.center_nx = None
        self.center_ny = None
        self.last_click_time = 0
        self.is_blinking = False
        
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
        
        title_font = ctk.CTkFont(size=20, weight="bold")
        lbl_title = ctk.CTkLabel(self.settings_frame, text="Settings", font=title_font)
        lbl_title.pack(pady=10)
        
        self.btn_toggle = ctk.CTkButton(self.settings_frame, text="ENABLE MOUSE (E)", 
                                        fg_color="#C2185B", hover_color="#9C1448", height=40,
                                        command=self.toggle_active)
        self.btn_toggle.pack(fill="x", padx=20, pady=(0, 15))
        
        self.scroll = ctk.CTkScrollableFrame(self.settings_frame, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=10)
        
        self.sens_var = ctk.DoubleVar(value=0.3)
        self.dz_var = ctk.DoubleVar(value=0.015)
        self.bc_var = ctk.DoubleVar(value=0.22)
        self.bo_var = ctk.DoubleVar(value=0.26)
        self.cd_var = ctk.DoubleVar(value=0.8)
        
        self.sw_wink = ctk.BooleanVar(value=True)
        self.sw_mirror = ctk.BooleanVar(value=False)
        self.sw_inv_y = ctk.BooleanVar(value=False)

        self.add_slider("Sensitivity:", self.sens_var, 0.01, 1.0)
        self.add_slider("Deadzone:", self.dz_var, 0.001, 0.1)
        self.add_slider("Wink (Close):", self.bc_var, 0.1, 0.4)
        self.add_slider("Wink (Open):", self.bo_var, 0.15, 0.5)
        self.add_slider("Cooldown (s):", self.cd_var, 0.1, 3.0)

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

    def update_frame(self):
        if not self.cap.isOpened():
            return
            
        success, frame = self.cap.read()
        if not success:
            self.after(10, self.update_frame)
            return

        current_time = time.time()
        
        if self.sw_mirror.get():
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                nx, ny = landmarks[4].x, landmarks[4].y
                
                dot_color = (0, 255, 0) if self.is_active else (255, 0, 0)
                cv2.circle(rgb_frame, (int(nx * w), int(ny * h)), 6, dot_color, -1)

                if self.is_active:
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

                    if self.sw_wink.get():
                        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                        
                        bc = self.bc_var.get()
                        bo = self.bo_var.get()
                        
                        is_left = left_ear < bc and right_ear > bo
                        is_right = right_ear < bc and left_ear > bo
                        
                        if is_left or is_right:
                            if not self.is_blinking and (current_time - self.last_click_time > self.cd_var.get()):
                                try:
                                    pyautogui.click(button='left')
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
        
        img = Image.fromarray(rgb_frame)
        
        frame_w = self.cam_frame.winfo_width()
        frame_h = self.cam_frame.winfo_height()
        
        if frame_w > 10 and frame_h > 10:
            aspect = w / h
            new_w = frame_w
            new_h = int(frame_w / aspect)
            if new_h > frame_h:
                new_h = frame_h
                new_w = int(frame_h * aspect)
                
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
