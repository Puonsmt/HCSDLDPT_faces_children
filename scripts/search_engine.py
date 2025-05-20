# scripts/search_engine.py
import os
import numpy as np
import cv2
import mediapipe as mp
import pickle
import psycopg2
import io
from psycopg2 import Binary
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processing import parse_filename

# Từ feature_extraction.py
from feature_extraction import (
    extract_color_histogram,
    extract_landmark_features,
    extract_hog_features
)

# Database connection parameters
DB_PARAMS = {
    "host": "localhost",
    "database": "child_images_fn",
    "user": "postgres",
    "password": "tranphuong"  # Thay bằng mật khẩu của bạn
}

epsilon = 1e-10  # để tránh chia cho 0


def minmax_normalize(similarities):
    min_val = np.min(similarities)
    max_val = np.max(similarities)
    return (similarities - min_val) / (max_val - min_val + epsilon)


def load_feature_vectors():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()

        cursor.execute("""
                       SELECT f.image_id, i.image_path, f.hog, f.color_hist, f.landmark
                       FROM image_features_1 f
                                JOIN face_images i ON f.image_id = i.id
                       """)

        results = cursor.fetchall()

        # Khởi tạo dictionary để lưu trữ kết quả
        data = {
            'image_ids': [],
            'file_paths': [],
            'full_paths': [],
            'feature_vectors': {
                'hog': [],
                'color_hist': [],
                'landmark': []
            }
        }

        # Xử lý kết quả
        for row in results:
            image_id = row[0]
            file_path = row[1]
            full_path = os.path.join('../data/processed_images', file_path)
            hog_vector = np.array(row[2])
            color_hist_vector = np.array(row[3])
            landmark_vector = np.array(row[4])

            # Thêm image_id và paths
            data['image_ids'].append(image_id)
            data['file_paths'].append(file_path)
            data['full_paths'].append(full_path)

            # Thêm các vector đặc trưng
            data['feature_vectors']['hog'].append(hog_vector)
            data['feature_vectors']['color_hist'].append(color_hist_vector)
            data['feature_vectors']['landmark'].append(landmark_vector)

        # Chuyển đổi list thành numpy array
        data['image_ids'] = np.array(data['image_ids'])
        data['file_paths'] = np.array(data['file_paths'])
        data['full_paths'] = np.array(data['full_paths'])
        data['feature_vectors']['hog'] = np.array(data['feature_vectors']['hog'])
        data['feature_vectors']['color_hist'] = np.array(data['feature_vectors']['color_hist'])
        data['feature_vectors']['landmark'] = np.array(data['feature_vectors']['landmark'])

        return data

    except Exception as e:
        print(f"Lỗi khi tải vector đặc trưng: {e}")
        return None


def compute_similarity(query_vector, all_vectors, metric='cosine'):
    """Compute similarity between query vector and all stored vectors."""
    if metric == 'cosine':
        # Normalize vectors for cosine similarity
        query_norm = np.linalg.norm(query_vector)
        all_norms = np.linalg.norm(all_vectors, axis=1)

        # Avoid division by zero
        query_norm = max(query_norm, 1e-10)
        all_norms = np.maximum(all_norms, 1e-10)

        # Compute dot product and then cosine similarity
        dot_products = np.dot(all_vectors, query_vector)
        similarities = dot_products / (all_norms * query_norm)

    elif metric == 'euclidean':
        # Compute Euclidean distance
        distances = np.linalg.norm(all_vectors - query_vector, axis=1)
        # Convert to similarity (smaller distance = higher similarity)
        similarities = 1 / (1 + distances)

    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return similarities


def extract_query_features(query_image_path, feature_type='hog'):
    """Extract features from query image based on feature type."""

    if feature_type == 'hog':
        return extract_hog_features(query_image_path)
    elif feature_type == 'color_hist':
        return extract_color_histogram(query_image_path)
    elif feature_type == 'landmark':
        return extract_landmark_features(query_image_path)
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")


def get_image_similarity(query_image_path, metric='cosine'):
    """Find the top_k most similar images to the query image."""
    try:
        weights = {
            'hog': 0.3,
            'color_hist': 0.2,
            'landmark': 0.5
        }

        # Tải toàn bộ đặc trưng đã có trong database về
        data = load_feature_vectors()

        # Compute similarities
        query_vector_hog = extract_query_features(query_image_path, 'hog')
        similarities_hog_1 = compute_similarity(query_vector_hog, data['feature_vectors']['hog'], metric)
        similarities_hog = minmax_normalize(similarities_hog_1)

        query_vector_color_hist = extract_query_features(query_image_path, 'color_hist')
        similarities_color_hist_1 = compute_similarity(query_vector_color_hist, data['feature_vectors']['color_hist'],
                                                       metric)
        similarities_color_hist = minmax_normalize(similarities_color_hist_1)

        query_vector_landmark = extract_query_features(query_image_path, 'landmark')
        similarities_landmark_1 = compute_similarity(query_vector_landmark, data['feature_vectors']['landmark'], metric)
        similarities_landmark = minmax_normalize(similarities_landmark_1)

        similarities = (
                weights['hog'] * similarities_hog +
                weights['color_hist'] * similarities_color_hist +
                weights['landmark'] * similarities_landmark
        )

        # Return top matches
        results = []
        for i, (image_id, file_path, full_path, sim) in enumerate(zip(
                data['image_ids'], data['file_paths'], data['full_paths'], similarities)):
            # Lưu chi tiết độ tương đồng của từng đặc trưng
            similarity_details = {
                'hog': float(similarities_hog[i] * 100),
                'color_hist': float(similarities_color_hist[i] * 100),
                'landmark': float(similarities_landmark[i] * 100)
            }

            results.append({
                'image_id': image_id,
                'file_path': file_path,
                'full_path': full_path,
                'similarity': float(sim),
                'similarity_details': similarity_details
            })

        if metric == 'cosine':
            results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:3]
        else:
            # Sắp xếp theo similarity tăng dần và lấy top 3
            results = sorted(results, key=lambda x: x['similarity'])[:3]

        final_results = []
        for r in results:
            # Trích xuất tên file từ đường dẫn
            filename = os.path.basename(r['file_path'])

            # Sử dụng hàm parse_filename đã import
            age, gender, race = parse_filename(filename)

            # Chuyển đổi gender từ số sang text
            if gender is not None:
                gender = 'Nữ' if gender == 1 else 'Nam'

            # Tạo dictionary kết quả
            final_results.append({
                'image_id': r['image_id'],
                'file_name': r['file_path'],
                'filename': filename,
                'path': f'/static/images/{filename}',
                'similarity': r['similarity'],
                'similarity_details': r['similarity_details'],
                'age': age,
                'gender': gender,
            })

        return final_results

    except Exception as e:
        print(f"Error finding similar images: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    query_image = "../data/test_image.jpg"  # Path to test image

    metric = 'cosine'

    results = get_image_similarity(query_image, metric)

    if results:
        print("\nTop similar images:")
        for i, result in enumerate(results):
            print(f"{i + 1}. {result['file_name']} (Similarity: {result['similarity']:.4f})")
    else:
        print("No similar images found")