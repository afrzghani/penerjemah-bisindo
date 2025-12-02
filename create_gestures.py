import cv2
import numpy as np
import os
import sqlite3
import random
import mediapipe as mp

image_x, image_y = 50, 50

class MediaPipeHandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def detect_hand(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        h, w = frame.shape[:2]
        hand_mask = np.zeros((h, w), dtype=np.uint8)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(hand_mask, (px, py), 8, 255, -1)

                connections = self.mp_hands.HAND_CONNECTIONS
                
                for connection in connections:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    start_point = hand_landmarks.landmark[start_idx]
                    end_point = hand_landmarks.landmark[end_idx]
                    
                    start_px = int(start_point.x * w)
                    start_py = int(start_point.y * h)
                    end_px = int(end_point.x * w)
                    end_py = int(end_point.y * h)
                    
                    cv2.line(hand_mask, (start_px, start_py), (end_px, end_py), 255, 10)
                
                palm_center = hand_landmarks.landmark[0]
                seed_x = int(palm_center.x * w)
                seed_y = int(palm_center.y * h)
                
                mask_copy = hand_mask.copy()
                cv2.floodFill(mask_copy, None, (seed_x, seed_y), 255)
                hand_mask = cv2.bitwise_or(hand_mask, mask_copy)
        
        return hand_mask
    
    def release(self):
        self.hands.close()

mediapipe_detector = MediaPipeHandDetector()

def init_create_folder_database():
    if not os.path.exists("gestures"):
        os.mkdir("gestures")
    if not os.path.exists("gesture_db.db"):
        conn = sqlite3.connect("gesture_db.db")
        create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
        conn.execute(create_table_cmd)
        conn.commit()

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def store_in_db(g_id, g_name):
    conn = sqlite3.connect("gesture_db.db")
    cmd = "INSERT INTO gesture (g_id, g_name) VALUES (%s, \'%s\')" % (g_id, g_name)
    try:
        conn.execute(cmd)
    except sqlite3.IntegrityError:
        choice = input("g_id sudah ada. Ingin mengubah? (y/n): ")
        if choice.lower() == 'y':
            cmd = "UPDATE gesture SET g_name = \'%s\' WHERE g_id = %s" % (g_name, g_id)
            conn.execute(cmd)
        else:
            print("...")
            return
    conn.commit()

def store_images(g_id):
    total_pics = 200
    cam = cv2.VideoCapture(0)
    x, y, w, h = 300, 100, 300, 300

    create_folder("gestures/"+str(g_id))
    pic_no = 0
    flag_start_capturing = False
    frames = 0
    
    print("\n=================")
    print("1. Posisikan tangan di dalam kotak hijau")
    print("2. Tekan 'C' untuk mulai/berhenti capture")
    print("3. Tahan gesture sampai 200 gambar tercapture")
    print("4. Tekan 'Q' untuk keluar")
    print("=================\n")
    
    while True:
        ret, img = cam.read()
        if not ret:
            break
            
        img = cv2.flip(img, 1)
        hand_mask = mediapipe_detector.detect_hand(img)
        thresh = hand_mask[y:y+h, x:x+w]
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000 and frames > 50:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                pic_no += 1
                save_img = thresh[y1:y1+h1, x1:x1+w1]
                
                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, (0, 0, 0))
                
                save_img = cv2.resize(save_img, (image_x, image_y))
                
                rand = random.randint(0, 10)
                if rand % 2 == 0:
                    save_img = cv2.flip(save_img, 1)

                cv2.imwrite("gestures/"+str(g_id)+"/"+str(pic_no)+".jpg", save_img)
                cv2.putText(img, "Mengambil...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 0), 3)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        counter_color = (0, 255, 0) if flag_start_capturing else (127, 127, 255)
        cv2.putText(img, f"{pic_no}/{total_pics}", (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, counter_color, 2)

        status = "CAPTURING" if flag_start_capturing else "READY (Press C)"
        status_color = (0, 255, 0) if flag_start_capturing else (0, 200, 255)
        cv2.putText(img, status, (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        cv2.imshow("Menangkap gestur", img)
        cv2.imshow("Mask", thresh)
        
        keypress = cv2.waitKey(1)
        
        if keypress == ord('c'):
            flag_start_capturing = not flag_start_capturing
            if not flag_start_capturing:
                frames = 0
            print(f"Mengambil gestur: {flag_start_capturing}")
        
        if keypress == ord('q'):
            print("Keluar")
            break
            
        if flag_start_capturing:
            frames += 1
        
        if pic_no == total_pics:
            print(f"\nBerhasil menangkap {total_pics} gambar!")
            break
    
    mediapipe_detector.release()
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    init_create_folder_database()
    
    print("\n" + "="*50)
    print("Membuat Dataset Gesture")
    print("="*50)
    
    while True:
        print("\n" + "-"*50)
        g_id = input("Masukkan ID Gesture (atau 'q' untuk keluar): ")
        
        if g_id.lower() == 'q':
            print("Keluar dari program...")
            break
        
        try:
            g_id = int(g_id)
        except ValueError:
            print("ID harus berupa angka!")
            continue
        
        g_name = input("Masukkan Nama Gesture: ")
        
        if not g_name.strip():
            print("Nama gesture tidak boleh kosong!")
            continue
        
        print(f"\nMembuat dataset untuk:")
        print(f"  ID: {g_id}")
        print(f"  Nama: {g_name}")
        print(f"  Total gambar: 200")
        
        store_in_db(g_id, g_name)
        store_images(g_id)
        
        print(f"\nDataset untuk gesture '{g_name}' selesai!")
        
        lanjut = input("\nLanjut ke gesture berikutnya? (y/n): ")
        if lanjut.lower() != 'y':
            break