import cv2, pickle
import numpy as np
import tensorflow as tf
import os
import sqlite3
from keras.models import load_model
from threading import Thread
import mediapipe as mp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model('cnn_model_keras2.h5')

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

                mp_hands = mp.solutions.hands
                connections = mp_hands.HAND_CONNECTIONS
                
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

def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

image_x, image_y = get_image_size()

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	img = img / 255.0
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed, verbose=0)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
	conn = sqlite3.connect("gesture_db.db")
	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
	cursor = conn.execute(cmd)
	for row in cursor:
		return row[0]

def get_pred_from_contour(contour, thresh):
	x1, y1, w1, h1 = cv2.boundingRect(contour)
	save_img = thresh[y1:y1+h1, x1:x1+w1]
	
	if w1 > h1:
		save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
	elif h1 > w1:
		save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, (0, 0, 0))
	
	pred_probab, pred_class = keras_predict(model, save_img)
	
	text = get_pred_text_from_db(pred_class)
	if text is None:
		text = ""
	
	return text, pred_probab

x, y, w, h = 300, 100, 300, 300

def get_img_contour_thresh_mediapipe(img):
    img = cv2.flip(img, 1)
    
    hand_mask = mediapipe_detector.detect_hand(img)
    
    thresh = hand_mask[y:y+h, x:x+w]
    
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    return img, contours, thresh

def create_modern_ui(img, thresh, pred_text="", word="", confidence=0, status="Ready"):
	canvas_height = 720
	canvas_width = 1280
	canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
	
	for i in range(canvas_height):
		gradient_val = int(15 + (i / canvas_height) * 10)
		canvas[i, :] = (gradient_val, gradient_val, gradient_val)
	
	header_height = 100
	for i in range(header_height):
		intensity = int((1 - i/header_height) * 100)
		canvas[i, :] = (0, 150 - intensity, 255 - intensity)
	
	cv2.putText(canvas, "Penerjemah Sign Language BISINDO", (50, 60), 
	cv2.FONT_HERSHEY_TRIPLEX, 1.3, (255, 255, 255), 3)
	
	status_color = (0, 255, 0) if pred_text else (100, 100, 100)
	cv2.circle(canvas, (canvas_width - 80, 50), 18, status_color, -1)
	cv2.circle(canvas, (canvas_width - 80, 50), 18, (255, 255, 255), 2)
	cv2.putText(canvas, "LIVE", (canvas_width - 150, 60), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
	
	img_resized = cv2.resize(img, (640, 480))
	canvas[130:610, 30:670] = img_resized
	
	cv2.rectangle(canvas, (28, 128), (672, 612), (0, 200, 255), 3)
	cv2.rectangle(canvas, (25, 125), (675, 615), (50, 50, 50), 1)
	
	cv2.rectangle(canvas, (30, 620), (250, 650), (0, 200, 255), -1)
	cv2.putText(canvas, "ZONA DETEKSI", (50, 642), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
	
	thresh_resized = cv2.resize(thresh, (300, 225))
	thresh_colored = cv2.cvtColor(thresh_resized, cv2.COLOR_GRAY2BGR)
	canvas[130:355, 950:1250] = thresh_colored
	cv2.rectangle(canvas, (948, 128), (1252, 357), (80, 80, 80), 2)
	
	cv2.rectangle(canvas, (950, 360), (1120, 385), (80, 80, 80), -1)
	cv2.putText(canvas, "Masking Tangan", (960, 378), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
	
	results_y = 400
	results_h = 300
	cv2.rectangle(canvas, (700, results_y), (1250, results_y + results_h), (30, 30, 30), -1)
	cv2.rectangle(canvas, (700, results_y), (1250, results_y + results_h), (0, 200, 255), 2)
	
	cv2.rectangle(canvas, (700, results_y), (1250, results_y + 50), (0, 150, 200), -1)
	cv2.putText(canvas, "HASIL DETEKSI", (730, results_y + 35), 
	cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)
	
	cv2.putText(canvas, "Gesture:", (730, results_y + 90), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
	
	display_text = pred_text if pred_text else "---"
	text_color = (0, 255, 150) if pred_text else (100, 100, 100)
	cv2.putText(canvas, display_text, (730, results_y + 140), 
	cv2.FONT_HERSHEY_TRIPLEX, 1.8, text_color, 3)
	
	if confidence > 0:
		cv2.putText(canvas, "Kepercayaan:", (730, results_y + 180), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
		
		bar_width = 400
		bar_height = 25
		bar_x = 730
		bar_y = results_y + 195
		
		cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
		cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), 2)
		
		conf_width = int(bar_width * confidence)
		conf_color = (0, 255, 0) if confidence > 0.7 else (0, 200, 255) if confidence > 0.5 else (0, 150, 255)
		cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), conf_color, -1)
		
		cv2.putText(canvas, f"{int(confidence*100)}%", (bar_x + bar_width + 15, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
	
	cv2.putText(canvas, "Kata:", (730, results_y + 245), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
	
	display_word = word if word else "..."
	if len(display_word) > 20:
		display_word = display_word[:20] + "..."
	cv2.putText(canvas, display_word, (730, results_y + 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

	return canvas

def gesture_recognition_mode(cam):
	text = ""
	word = ""
	count_same_frame = 0
	confidence = 0
	status = "Siap mendeteksi gesture..."
	
	while True:
		img = cam.read()[1]
		img = cv2.resize(img, (640, 480))

		img, contours, thresh = get_img_contour_thresh_mediapipe(img)
		
		old_text = text
		
		if len(contours) > 0:
			contour = max(contours, key=cv2.contourArea)
			area = cv2.contourArea(contour)
			
			if area > 1000:
				text, conf = get_pred_from_contour(contour, thresh)
				confidence = conf
				
				if old_text == text:
					count_same_frame += 1
				else:
					count_same_frame = 0
				
				if count_same_frame > 20 and text:
					word = word + text
					
					count_same_frame = 0
					status = f"Ditambahkan '{text}' ke kata"
				elif text:
					status = f"Mendeteksi: {text} ({int(confidence*100)}%)"
			
			elif area < 500:
				text = ""
				confidence = 0
		else:
			text = ""
			confidence = 0
		
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		cv2.putText(img, "Zona Deteksi", (x+5, y-10), 
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
		
		display = create_modern_ui(img, thresh, text, word, confidence, status)
		
		cv2.imshow("Penerjamah BISINDO", display)
		
		keypress = cv2.waitKey(1)
		
		if keypress == ord('q'):
			break
		elif keypress == ord('r'):
			word = ""
			text = ""
			confidence = 0
			status = "Reset - Siap untuk deteksi baru"
		elif keypress == 32:
			word += " "
			status = "Spasi ditambahkan"
	
	return 0

def recognize():
	cam = cv2.VideoCapture(0)
	if not cam.read()[0]:
		cam = cv2.VideoCapture(0)
	
	gesture_recognition_mode(cam)
	
	mediapipe_detector.release()
	cam.release()
	cv2.destroyAllWindows()

keras_predict(model, np.zeros((50, 50), dtype=np.uint8))
recognize()