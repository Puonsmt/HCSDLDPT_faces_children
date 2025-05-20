# scripts/feature_extraction.py
import os
import numpy as np
import cv2
import mediapipe as mp
import pickle
import psycopg2
import io
from psycopg2 import Binary
import pandas as pd
from skimage.feature import hog
from PIL import Image, ImageOps


# Database connection parameters
DB_PARAMS = {
    "host": "localhost",
    "database": "child_images_fn",
    "user": "postgres",
    "password": "tranphuong"  # Thay bằng mật khẩu của bạn
}

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def extract_hog_features(image_path, resize_dim=(200, 200)):
    """
    Trích xuất đặc trưng HOG từ ảnh PIL.

    Args:
        image (PIL.Image): Ảnh đầu vào.
        resize_dim (tuple): Kích thước ảnh sau khi resize.

    Returns:
        np.array: Vector đặc trưng HOG.
    """
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path  # nếu đã là ảnh dạng np.array rồi

    # Chuyển từ PIL Image sang numpy array
    # image_np = np.array(image)

    # Nếu ảnh là RGB, chuyển sang grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) == 2:
        gray = image
    else:
        return None  # Không hợp lệ

    # Resize ảnh
    gray_resized = cv2.resize(gray, resize_dim)

    # Trích xuất HOG
    features = hog(
        gray_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )

    return np.array(features)

def extract_color_histogram(image_path, bins=(8, 8, 8)):
    """
    Trích xuất đặc trưng histogram màu từ ảnh.

    Args:
        image (PIL.Image hoặc numpy.ndarray)
        bins (tuple): Số lượng bin cho mỗi kênh màu (H, S, V)

    Returns:
        np.array: Vector histogram màu chuẩn hóa
    """
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path  # nếu đã là ảnh dạng np.array rồi

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_landmark_features(image_path, max_points=468):
    """
    Trích xuất đặc trưng landmark từ ảnh đầu vào.

    Args:
        image (PIL.Image hoặc numpy.ndarray)
    Returns:
        np.array: Vector các tọa độ landmark (flattened)
    """
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path  # nếu đã là ảnh dạng np.array rồi

    results = face_mesh.process(image)

    h, w, _ = image.shape
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        coords = []
        for i in range(max_points):
            pt = landmarks.landmark[i]
            coords.extend([pt.x * w, pt.y * h])  # flatten (x1, y1, x2, y2, ...)
        return np.array(coords)
    return np.zeros(max_points * 2)

def store_features_in_db(image_id, hog, color_hist, landmark):
    """
    Lưu hoặc cập nhật các đặc trưng vào bảng image_features.
    """
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()

        # Chuyển các vector numpy sang list Python để psycopg2 tự chuyển sang ARRAY
        hog_list = hog.tolist() if hog is not None else None
        color_hist_list = color_hist.tolist() if color_hist is not None else None
        landmark_list = landmark.tolist() if landmark is not None else None

        cursor.execute(
                """
                INSERT INTO image_features_1 (image_id, hog, color_hist, landmark)
                VALUES (%s, %s, %s, %s)
                """,
                (image_id, hog_list, color_hist_list, landmark_list)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error storing features in database: {e}")
        return False

def extract_and_store_features(metadata_path):
    """
    Trích xuất và lưu embedding, landmarks, color_histogram cho từng ảnh vào bảng image_features.
    """

    # Load metadata
    metadata = pd.read_csv(metadata_path)

    for _, row in metadata.iterrows():
        image_id = row['id']
        image_path = row['path']

        #print(f"Processing image: {image_path}")

        # Trích xuất các đặc trưng
        hog = extract_hog_features(image_path)
        color_hist = extract_color_histogram(image_path)
        landmark = extract_landmark_features(image_path)
        # print(hog)


        # Chỉ lưu nếu có hog (bắt buộc)
        if hog is not None:
            store_features_in_db(
                image_id,
                hog=hog,
                color_hist=color_hist,
                landmark=landmark
            )
        else:
            print(f"Skipping image {image_path} because embedding is missing.")

if __name__ == "__main__":
    processed_dir = "../data/processed_images"  # Đường dẫn tới thư mục chứa ảnh đã xử lý
    metadata_path = os.path.join(processed_dir, 'metadata.csv')
    extract_and_store_features(metadata_path)
