import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# تنظیمات اولیه
screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# ==========================================
# تنظیمات حرکت موس (حالت جوی‌استیک)
# ==========================================
sensitivity = 0.3   # حساسیت سرعت حرکت (هرچه کمتر، کندتر - قابل تغییر)
deadzone = 0.015     # منطقه خنثی در مرکز (تا زمانی که سر از این مقدار بیشتر نچرخد حرکت نمی‌کند)
y_invert = False

# ==========================================
# تنظیمات کلیک با چشمک یک طرفه
# ==========================================
click_cooldown = 0.8
last_click_time = 0
blink_closed_thresh = 0.22
blink_open_thresh = 0.26
is_blinking = False

# وضعیت فعال بودن
mouse_active = False
center_nx = None
center_ny = None

# نقاط مربوط به چشم‌ها در MediaPipe
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices):
    try:
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
        vert1 = ((p2.x - p6.x)**2 + (p2.y - p6.y)**2)**0.5
        vert2 = ((p3.x - p5.x)**2 + (p3.y - p5.y)**2)**0.5
        horiz = ((p1.x - p4.x)**2 + (p1.y - p4.y)**2)**0.5
        if horiz == 0: return 0
        return (vert1 + vert2) / (2.0 * horiz)
    except:
        return 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    # Mirror کردن تصویر غیرفعال شد تا دوربین برعکس نباشه
    # frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            nose_tip = landmarks[4]
            nx, ny = nose_tip.x, nose_tip.y
            
            nose_px_x = int(nx * w)
            nose_px_y = int(ny * h)
            
            color = (0, 255, 0) if mouse_active else (0, 0, 255)
            cv2.circle(frame, (nose_px_x, nose_px_y), 6, color, -1)
            
            # رسم مرکز و خط جوی‌استیک
            if mouse_active and center_nx is not None and center_ny is not None:
                center_px_x = int(center_nx * w)
                center_px_y = int(center_ny * h)
                cv2.circle(frame, (center_px_x, center_px_y), 4, (255, 0, 0), -1)
                cv2.line(frame, (center_px_x, center_px_y), (nose_px_x, nose_px_y), (255, 0, 0), 1)

            if mouse_active:
                if center_nx is None or center_ny is None:
                    # اولین باری که E زده میشه، موقعیت فعلی سر به عنوان مرکز ثبت میشه
                    center_nx = nx
                    center_ny = ny
                    
                dx = nx - center_nx
                dy = ny - center_ny
                
                # اعمال منطقه خنثی (Deadzone) برای جلوگیری از لرزش
                if abs(dx) > deadzone:
                    dx = np.sign(dx) * (abs(dx) - deadzone)
                else:
                    dx = 0
                    
                if abs(dy) > deadzone:
                    dy = np.sign(dy) * (abs(dy) - deadzone)
                else:
                    dy = 0
                    
                if y_invert: dy = -dy
                
                # محاسبه سرعت حرکت بر اساس فاصله سر از مرکز
                move_x = dx * screen_w * sensitivity
                move_y = dy * screen_h * sensitivity
                
                if abs(move_x) > 0 or abs(move_y) > 0:
                    try:
                        pyautogui.moveRel(int(move_x), int(move_y), _pause=False)
                    except Exception as e:
                        pass
                
                # بررسی وضعیت چشم‌ها برای کلیک
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                
                # شرط چشمک یک طرفه: یک چشم بسته و چشم دیگه کاملا باز باشه
                is_left_wink = left_ear < blink_closed_thresh and right_ear > blink_open_thresh
                is_right_wink = right_ear < blink_closed_thresh and left_ear > blink_open_thresh
                
                if (is_left_wink or is_right_wink):
                    if not is_blinking and (current_time - last_click_time > click_cooldown):
                        try:
                            pyautogui.click(button='left')
                            is_blinking = True
                            last_click_time = current_time
                        except:
                            pass
                else:
                    is_blinking = False
                        
                cv2.putText(frame, f"L:{left_ear:.2f} R:{right_ear:.2f} {'[CLICK!]' if is_blinking else ''}",
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255) if is_blinking else (255, 255, 255), 2)
            else:
                # ریست کردن متغیرها وقتی غیرفعاله
                center_nx = None
                center_ny = None
                is_blinking = False
    else:
        center_nx = None
        center_ny = None
        cv2.putText(frame, "Face not detected", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    status_text = "ACTIVE (Joystick Mode)" if mouse_active else "INACTIVE (Press E)"
    status_color = (0, 255, 0) if mouse_active else (0, 0, 255)
    cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, "E: Toggle | Q/Esc: Exit", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Head Mouse", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q'), ord('Q')]:
        break
    elif key in [ord('e'), ord('E')]:
        mouse_active = not mouse_active
        center_nx = None
        center_ny = None
        is_blinking = False

cap.release()
cv2.destroyAllWindows()
