import cv2
import numpy as np
import os
import time
import threading
from collections import deque
import pickle

class RealTimeFaceDetection:
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        # Initialize face cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Detection parameters
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        # For face tracking and recognition
        self.known_faces = {}  # Dictionary to store face templates
        self.face_id_counter = 0
        self.tracked_faces = []
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.processing_times = deque(maxlen=10)
        
        # Threading
        self.processing_thread = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.detected_faces = []
        
        # Face tracking
        self.trackers = []
        self.tracker_ids = []
        self.next_id = 0
        
    def detect_faces(self, frame, detect_eyes=True):
        """Detect faces using OpenCV cascade classifiers"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect frontal faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Detect profile faces
        profiles = self.profile_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        # Combine face detections
        all_faces = []
        
        # Add frontal faces
        for (x, y, w, h) in faces:
            face_info = {
                'bbox': (x, y, w, h),
                'type': 'frontal',
                'eyes': [],
                'confidence': self.min_neighbors  # Use min_neighbors as confidence proxy
            }
            
            # Detect eyes within face region for validation
            if detect_eyes:
                face_roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(face_roi_gray, 1.1, 3)
                face_info['eyes'] = [(ex+x, ey+y, ew, eh) for (ex, ey, ew, eh) in eyes]
                face_info['confidence'] += len(eyes) * 2  # Boost confidence if eyes detected
            
            all_faces.append(face_info)
        
        # Add profile faces (avoid duplicates)
        for (x, y, w, h) in profiles:
            # Check if this profile overlaps significantly with any frontal face
            overlap = False
            for face_info in all_faces:
                fx, fy, fw, fh = face_info['bbox']
                if self.calculate_overlap((x, y, w, h), (fx, fy, fw, fh)) > 0.3:
                    overlap = True
                    break
            
            if not overlap:
                face_info = {
                    'bbox': (x, y, w, h),
                    'type': 'profile',
                    'eyes': [],
                    'confidence': self.min_neighbors
                }
                all_faces.append(face_info)
        
        return all_faces
    
    def calculate_overlap(self, box1, box2):
        """Calculate overlap ratio between two bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection area
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def extract_face_features(self, frame, bbox):
        """Extract simple features from face region"""
        x, y, w, h = bbox
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return None
        
        # Resize to standard size
        face_roi = cv2.resize(face_roi, (100, 100))
        
        # Convert to grayscale
        if len(face_roi.shape) == 3:
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_roi
        
        # Calculate histogram features
        hist = cv2.calcHist([face_gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Calculate LBP (Local Binary Pattern) features
        lbp = self.calculate_lbp(face_gray)
        lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()
        
        # Combine features
        features = np.concatenate([hist, lbp_hist])
        
        return features
    
    def calculate_lbp(self, image):
        """Calculate Local Binary Pattern"""
        rows, cols = image.shape
        lbp_image = np.zeros((rows-2, cols-2), dtype=np.uint8)
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = image[i, j]
                pattern = 0
                
                # Check 8 neighbors
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        pattern |= (1 << k)
                
                lbp_image[i-1, j-1] = pattern
        
        return lbp_image
    
    def compare_faces(self, features1, features2):
        """Compare two face feature vectors"""
        if features1 is None or features2 is None:
            return 0.0
        
        # Calculate cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, similarity)  # Ensure non-negative
    
    def add_known_face(self, frame, bbox, name):
        """Add a face to the known faces database"""
        features = self.extract_face_features(frame, bbox)
        if features is not None:
            if name not in self.known_faces:
                self.known_faces[name] = []
            
            self.known_faces[name].append(features)
            print(f"✓ Added face sample for {name}")
            
            # Keep only last 5 samples per person
            if len(self.known_faces[name]) > 5:
                self.known_faces[name] = self.known_faces[name][-5:]
            
            return True
        return False
    
    def recognize_face(self, frame, bbox, threshold=0.7):
        """Try to recognize a face"""
        features = self.extract_face_features(frame, bbox)
        if features is None:
            return "Unknown", 0.0
        
        best_match = "Unknown"
        best_score = 0.0
        
        for name, face_features_list in self.known_faces.items():
            scores = []
            for known_features in face_features_list:
                score = self.compare_faces(features, known_features)
                scores.append(score)
            
            if scores:
                avg_score = np.mean(scores)
                if avg_score > best_score and avg_score > threshold:
                    best_score = avg_score
                    best_match = name
        
        return best_match, best_score
    
    def save_known_faces(self, filename="known_faces_opencv.pkl"):
        """Save known faces to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.known_faces, f)
        print(f"Known faces saved to {filename}")
    
    def load_known_faces(self, filename="known_faces_opencv.pkl"):
        """Load known faces from file"""
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    self.known_faces = pickle.load(f)
                print(f"✓ Loaded {len(self.known_faces)} known faces")
                return True
            except Exception as e:
                print(f"Error loading known faces: {e}")
        return False
    
    def process_frame_threaded(self, frame, detection_enabled=True, recognition_enabled=True):
        """Process frame in separate thread"""
        start_time = time.time()
        
        detected_faces = []
        
        # Only detect faces if detection is enabled
        if detection_enabled:
            # Detect faces
            detected_faces = self.detect_faces(frame, detect_eyes=True)
            
            # Only recognize faces if recognition is enabled
            if recognition_enabled:
                for face_info in detected_faces:
                    bbox = face_info['bbox']
                    name, confidence = self.recognize_face(frame, bbox)
                    face_info['name'] = name
                    face_info['recognition_confidence'] = confidence
            else:
                # Set default values when recognition is disabled
                for face_info in detected_faces:
                    face_info['name'] = "Detection Only"
                    face_info['recognition_confidence'] = 0.0
        
        # Update results
        with self.frame_lock:
            self.detected_faces = detected_faces
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
    
    def draw_button(self, frame, x, y, width, height, text, is_active, color_active=(0, 200, 0), color_inactive=(0, 0, 200)):
        """Draw a clickable button on the frame"""
        color = color_active if is_active else color_inactive
        text_color = (255, 255, 255)
        
        # Draw button background
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)
        
        # Draw button text
        font_scale = 0.6
        thickness = 1
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = x + (width - text_size[0]) // 2
        text_y = y + (height + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        
        return (x, y, width, height)
    
    def is_point_in_button(self, x, y, button_rect):
        """Check if a point is inside a button"""
        bx, by, bw, bh = button_rect
        return bx <= x <= bx + bw and by <= y <= by + bh

    def start_camera(self, camera_index=0, width=1280, height=720):
        """Start real-time face detection"""
        cap = cv2.VideoCapture(camera_index)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("\n🎥 Starting real-time face detection (OpenCV only)...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'a' - Add person (click on face first)")
        print("  'r' - Reset detection parameters")
        print("  'f' - Toggle FPS display")
        print("  'e' - Toggle eye detection")
        print("  'l' - Load known faces")
        print("  'Space' - Pause/Resume")
        print("  '+/-' - Adjust detection sensitivity")
        print("  Click buttons to toggle detection/recognition")
        
        show_fps = True
        show_eyes = True
        paused = False
        selected_face = None
        process_every_n_frames = 2
        frame_count = 0
        
        # Toggle switches for detection and recognition
        face_detection_enabled = True
        face_recognition_enabled = True
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_face, face_detection_enabled, face_recognition_enabled
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if clicked on buttons first
                detection_button_rect = (width - 220, 10, 100, 40)
                recognition_button_rect = (width - 110, 10, 100, 40)
                
                if self.is_point_in_button(x, y, detection_button_rect):
                    face_detection_enabled = not face_detection_enabled
                    print(f"Face Detection: {'ON' if face_detection_enabled else 'OFF'}")
                    return
                
                if self.is_point_in_button(x, y, recognition_button_rect):
                    face_recognition_enabled = not face_recognition_enabled
                    print(f"Face Recognition: {'ON' if face_recognition_enabled else 'OFF'}")
                    return
                
                # Check if clicked on a face
                with self.frame_lock:
                    current_faces = self.detected_faces.copy()
                
                for i, face_info in enumerate(current_faces):
                    fx, fy, fw, fh = face_info['bbox']
                    if fx <= x <= fx + fw and fy <= y <= fy + fh:
                        selected_face = i
                        print(f"Selected face {i}")
                        break
        
        cv2.namedWindow('Real-Time Face Detection')
        cv2.setMouseCallback('Real-Time Face Detection', mouse_callback)
        
        frame_times = deque(maxlen=30)
        
        # Load existing known faces
        self.load_known_faces()
        
        try:
            while True:
                frame_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                if not paused:
                    frame_count += 1
                    
                    # Process frame every N frames (only if detection is enabled)
                    if face_detection_enabled and frame_count % process_every_n_frames == 0:
                        if self.processing_thread is None or not self.processing_thread.is_alive():
                            self.processing_thread = threading.Thread(
                                target=self.process_frame_threaded,
                                args=(frame, face_detection_enabled, face_recognition_enabled)
                            )
                            self.processing_thread.start()
                    elif not face_detection_enabled:
                        # Clear detected faces when detection is disabled
                        with self.frame_lock:
                            self.detected_faces = []
                
                # Draw results (only if detection is enabled)
                display_frame = frame.copy()
                
                # Draw control buttons
                detection_button_rect = self.draw_button(display_frame, width - 220, 10, 100, 40, 
                                                       "Detection", face_detection_enabled)
                recognition_button_rect = self.draw_button(display_frame, width - 110, 10, 100, 40, 
                                                         "Recognition", face_recognition_enabled)
                
                if face_detection_enabled:
                    with self.frame_lock:
                        current_faces = self.detected_faces.copy()
                    
                    for i, face_info in enumerate(current_faces):
                        x, y, w, h = face_info['bbox']
                        face_type = face_info['type']
                        confidence = face_info['confidence']
                        
                        # Color based on recognition
                        if face_recognition_enabled and 'name' in face_info and face_info['name'] not in ["Unknown", "Detection Only"]:
                            color = (0, 255, 0)  # Green for recognized
                            name = face_info['name']
                            rec_conf = face_info['recognition_confidence']
                            label = f"{name} ({rec_conf:.2f})"
                        elif face_recognition_enabled and 'name' in face_info and face_info['name'] == "Unknown":
                            color = (255, 0, 0)  # Blue for unknown
                            label = f"Unknown {face_type.title()}"
                        else:
                            color = (255, 165, 0)  # Orange for detection only
                            label = f"{face_type.title()} (conf: {confidence})"
                        
                        # Highlight selected face
                        if selected_face == i:
                            color = (255, 255, 0)  # Cyan
                            cv2.rectangle(display_frame, (x-3, y-3), (x+w+3, y+h+3), color, 3)
                        
                        # Draw face rectangle
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                        
                        # Draw label
                        label_y = y - 10 if y - 10 > 10 else y + h + 25
                        cv2.putText(display_frame, label, (x, label_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Draw eyes if detected and enabled
                        if show_eyes and 'eyes' in face_info:
                            for (ex, ey, ew, eh) in face_info['eyes']:
                                cv2.rectangle(display_frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 1)
                else:
                    current_faces = []
                
                # Show stats
                if show_fps:
                    frame_time = time.time() - frame_start
                    frame_times.append(frame_time)
                    
                    if len(frame_times) > 1:
                        fps = 1.0 / np.mean(frame_times)
                        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Processing time
                    if self.processing_times:
                        avg_proc = np.mean(self.processing_times) * 1000
                        cv2.putText(display_frame, f"Process: {avg_proc:.1f}ms", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    
                    # Face count
                    face_count = len(current_faces)
                    cv2.putText(display_frame, f"Faces: {face_count}", (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    
                    # Known people
                    cv2.putText(display_frame, f"Known: {len(self.known_faces)}", (10, 120),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    
                    # Detection sensitivity
                    cv2.putText(display_frame, f"MinNeighbors: {self.min_neighbors}", (10, 150),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Status indicators
                    status_y = 180
                    det_status = "ON" if face_detection_enabled else "OFF"
                    det_color = (0, 255, 0) if face_detection_enabled else (0, 0, 255)
                    cv2.putText(display_frame, f"Detection: {det_status}", (10, status_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, det_color, 1)
                    
                    rec_status = "ON" if face_recognition_enabled else "OFF"
                    rec_color = (0, 255, 0) if face_recognition_enabled else (0, 0, 255)
                    cv2.putText(display_frame, f"Recognition: {rec_status}", (10, status_y + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, rec_color, 1)
                
                # Pause indicator
                if paused:
                    cv2.putText(display_frame, "PAUSED", (display_frame.shape[1]//2 - 60, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('Real-Time Face Detection', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"face_detection_{int(time.time())}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Frame saved as {filename}")
                elif key == ord('a'):
                    if not face_detection_enabled:
                        print("Please enable face detection first")
                    elif selected_face is not None and selected_face < len(current_faces):
                        name = input("\nEnter person's name: ").strip()
                        if name:
                            face_info = current_faces[selected_face]
                            if self.add_known_face(frame, face_info['bbox'], name):
                                self.save_known_faces()
                                print(f"Added {name} and saved to database")
                                if not face_recognition_enabled:
                                    print("Note: Enable face recognition to see the recognition results")
                        selected_face = None
                    else:
                        print("Please click on a face first, then press 'a'")
                elif key == ord('r'):
                    self.min_neighbors = 5
                    self.scale_factor = 1.1
                    print("Reset detection parameters")
                elif key == ord('f'):
                    show_fps = not show_fps
                elif key == ord('e'):
                    show_eyes = not show_eyes
                    print(f"Eye detection: {'ON' if show_eyes else 'OFF'}")
                elif key == ord('l'):
                    self.load_known_faces()
                elif key == ord(' '):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('+') or key == ord('='):
                    self.min_neighbors = min(10, self.min_neighbors + 1)
                    print(f"Increased sensitivity: minNeighbors = {self.min_neighbors}")
                elif key == ord('-'):
                    self.min_neighbors = max(1, self.min_neighbors - 1)
                    print(f"Decreased sensitivity: minNeighbors = {self.min_neighbors}")
        
        except KeyboardInterrupt:
            print("\nStopping...")
        
        finally:
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join()
            cap.release()
            cv2.destroyAllWindows()
            print("Camera stopped")

def main():
    # Initialize the detector
    detector = RealTimeFaceDetection(
        scale_factor=1.1,        # How much the image size is reduced at each scale
        min_neighbors=5,         # How many neighbors each face should have to be detected
        min_size=(30, 30)       # Minimum possible face size
    )
    
    # Start the camera
    detector.start_camera(camera_index=0, width=1280, height=720)

if __name__ == "__main__":
    main()