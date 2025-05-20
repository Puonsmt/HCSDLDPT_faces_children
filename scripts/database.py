# scripts/database.py
import os
import psycopg2
from psycopg2 import sql


def create_database():
    # Kết nối đến PostgreSQL server
    conn = psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="tranphuong"  # Thay bằng mật khẩu của bạn
    )
    conn.autocommit = True
    cursor = conn.cursor()

    # Tạo database child_images_fn nếu chưa tồn tại
    try:
        cursor.execute("CREATE DATABASE child_images_fn")
        print("Database created successfully")
    except psycopg2.errors.DuplicateDatabase:
        print("Database already exists")

    cursor.close()
    conn.close()

    # Kết nối đến database mới tạo
    conn = psycopg2.connect(
        host="localhost",
        database="child_images_fn",
        user="postgres",
        password="tranphuong"  # Thay bằng mật khẩu của bạn
    )
    cursor = conn.cursor()

    # Tạo bảng images
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_images(
        id SERIAL PRIMARY KEY,           -- id tự động tăng
        image_path TEXT NOT NULL,        -- Đường dẫn ảnh
        age INT NOT NULL,                -- Tuổi
        gender INT NOT NULL,             -- Giới tính (0: Nam, 1: Nữ)
        race INT NOT NULL                -- Chủng tộc (ví dụ: 0: Châu Á, 1: Châu Âu, v.v.)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS image_features_1(
        id SERIAL PRIMARY KEY,  -- id tự động tăng
        image_id INT NOT NULL, -- Khóa ngoại liên kết với bảng face_images
        hog FLOAT8[] NOT NULL, -- HOG (Histogram of Oriented Gradients) lưu dưới dạng JSON
        color_hist FLOAT8[] NOT NULL, -- Màu sắc histogram lưu dưới dạng JSON
        landmark FLOAT8[] NOT NULL, -- Dữ liệu landmark (điểm đặc trưng của khuôn mặt) lưu dưới dạng JSON
        FOREIGN KEY(image_id) REFERENCES face_images(id) ON DELETE CASCADE
    )
    """)


    conn.commit()

    print("Tables created successfully")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    create_database()