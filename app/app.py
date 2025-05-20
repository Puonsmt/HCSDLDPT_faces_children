import os
import sys
import time
from flask import Flask, request, render_template, url_for, redirect, flash, send_from_directory

# Thêm đường dẫn để import từ thư mục scripts
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
scripts_dir = os.path.join(parent_dir, 'scripts')
sys.path.append(scripts_dir)

from scripts.search_engine import get_image_similarity

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Giới hạn file 16MB

# Đường dẫn đến thư mục processed_images
PROCESSED_IMAGES_DIR = os.path.join(parent_dir, 'data', 'processed_images')

# Tạo thư mục uploads nếu chưa tồn tại
os.makedirs(os.path.join(app.root_path, app.config['UPLOAD_FOLDER']), exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash('Không tìm thấy ảnh trong form')
        return redirect(url_for('index'))

    file = request.files['image']

    if file.filename == '':
        flash('Không có file nào được chọn')
        return redirect(url_for('index'))

    if file:
        try:
            # Lưu file đã upload
            filename = os.path.basename(file.filename)
            upload_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # Tạo URL cho ảnh đã tải lên
            upload_image_url = url_for('static', filename=f'uploads/{filename}')

            # Đo thời gian tìm kiếm
            start_time = time.time()

            # Tìm ảnh tương tự
            results = get_image_similarity(upload_path, metric='cosine')

            # Tính thời gian tìm kiếm
            search_time = time.time() - start_time
            search_time_ms = round(search_time * 1000, 2)  # Convert to milliseconds

            if not results:
                flash('Không tìm thấy ảnh tương đồng hoặc có lỗi khi xử lý ảnh')
                return redirect(url_for('index'))

            # Cập nhật kết quả với URL mới cho ảnh
            for result in results:
                result['image_url'] = url_for('serve_data_image', filename=result['filename'])

            return render_template('results.html',
                                   upload_image_url=upload_image_url,
                                   results=results,
                                   search_time=search_time_ms)

        except Exception as e:
            flash(f'Có lỗi xảy ra: {str(e)}')
            return redirect(url_for('index'))


@app.route('/data-images/<filename>')
def serve_data_image(filename):
    """Phục vụ ảnh từ thư mục data/processed_images"""
    if os.path.exists(os.path.join(PROCESSED_IMAGES_DIR, filename)):
        return send_from_directory(PROCESSED_IMAGES_DIR, filename)
    else:
        return f"Không tìm thấy file: {filename}", 404


if __name__ == '__main__':
    app.run(debug=True)
