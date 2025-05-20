# scripts/data_processing.py
import os
import shutil
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import psycopg2
import re

# Database connection parameters
DB_PARAMS = {
    "host": "localhost",
    "database": "child_images_fn",
    "user": "postgres",
    "password": "tranphuong"  # Thay bằng mật khẩu của bạn
}


def parse_filename(filename):
    """Parse UTKFace filename to extract age, gender, race information."""
    try:
        parts = filename.split('_')
        age = int(parts[0])
        gender = int(parts[1])  # 0 for male, 1 for female
        race = int(parts[2])
        return age, gender, race
    except:
        return None, None, None


def process_images(raw_dir, processed_dir, max_age=10, target_size=(200, 200)):
    """
    Process UTKFace images:
    1. Select only children images (age <= max_age)
    2. Resize to consistent dimensions
    3. Save processed images to processed_dir
    4. Insert metadata into database
    """
    os.makedirs(processed_dir, exist_ok=True)

    # Connect to database
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()

    # List to store metadata
    metadata = []

    # Process each image
    for filename in os.listdir(raw_dir):
        if not (filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg')):
            continue

        age, gender, race = parse_filename(filename)

        if age is None or age > max_age:
            continue  # Skip non-children or invalid files

        # Read image
        img_path = os.path.join(raw_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not read image: {filename}")
            continue

        # Resize image
        img_resized = cv2.resize(img, target_size)

        # Save processed image
        processed_path = os.path.join(processed_dir, filename)
        cv2.imwrite(processed_path, img_resized)

        # Store metadata
        gender_label = 'female' if gender == 1 else 'male'

        # Insert into database
        try:
            cursor.execute(
                """
                INSERT INTO face_images (image_path, age, gender, race)
                VALUES (%s, %s, %s, %s) RETURNING id
                """,
                (filename, age, gender, race)
            )

            image_id = cursor.fetchone()[0]
            metadata.append({
                'id': image_id,
                'file_name': filename,
                'age': age,
                'gender': gender_label,
                'race': race,
                'path': processed_path
            })


        except psycopg2.errors.UniqueViolation:
            # Skip duplicate files
            conn.rollback()
            print(f"Skipping duplicate file: {filename}")

    # Commit changes and close connection
    conn.commit()
    cursor.close()
    conn.close()

    # Create metadata DataFrame and save to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(processed_dir, 'metadata.csv'), index=False)

    print(f"Processed {len(metadata)} children face images")

    return metadata_df


if __name__ == "__main__":
    raw_dir = "../data/raw_images"  # Đường dẫn tới thư mục chứa ảnh gốc
    processed_dir = "../data/processed_images"  # Đường dẫn tới thư mục chứa ảnh đã xử lý
    metadata = process_images(raw_dir, processed_dir)
    print(f"Total children images: {len(metadata)}")