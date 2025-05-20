## HCSDLDPT_faces_children
Đây là một dự án học tập cho môn học "Hệ Cơ Sở Dữ Liệu Đa Phương Tiện" (HCSDLDPT), với yêu cầu:

Xây dựng hệ CSDL lưu trữ và tìm kiếm ảnh mặt trẻ em.

1.Hãy xây dựng/sưu tầm một bộ dữ liệu ảnh gồm ít nhất 150 files ảnh chân dung trẻ em, các ảnh có cùng kích thước, vật trong ảnh có cùng tỉ lệ khung hình (SV tùy chọn định dạng ảnh).

2.Hãy xây dựng một bộ thuộc tính để nhận diện ảnh mặt trẻ em từ bộ dữ liệu đã thu thập. Trình bày cụ thể vể lý do lựa chọn cùng giá trị thông tin của các thuộc tính được sử dụng.

3.Xây dựng hệ thống tìm kiếm ảnh mặt trẻ em với đầu vào là một ảnh mới về mặt trẻ em (ảnh của người đã có và không có trong dữ liệu), đầu ra là 3 ảnh giống nhất, xếp thứ tự giảm dần về độ tương đồng nội dung với ảnh đầu vào.

  a.Trình bày sơ đồ khối của hệ thống và quy trình thực hiện yêu cầu của đề bài.
  
  b.Trình bày quá trình trích rút, lưu trữ và sử dụng các thuộc tính để tìm kiếm ảnh trong hệ thống.
  
4.Demo hệ thống và đánh giá kết quả đã đạt được.

## 📁 Cấu trúc thư mục
app/: Chứa mã nguồn chính của ứng dụng.

data/: Chứa dữ liệu thô và dữ liệu sau xử lý 

scripts/: Bao gồm các script hỗ trợ như tiền xử lý dữ liệu, trích rút, và tìm kiếm.

requirements.txt: Danh sách các thư viện và phiên bản cần thiết để chạy dự án.

## 🧪 Yêu cầu hệ thống
Python 3.8 trở lên

Các thư viện được liệt kê trong requirements.txt

## 🚀 Cài đặt và chạy
# Sao chép kho lưu trữ về máy:
git clone https://github.com/Puonsmt/HCSDLDPT_faces_children.git

cd HCSDLDPT_faces_children

# Tạo môi trường ảo và cài đặt các phụ thuộc: 
python -m venv venv

pip install -r requirements.txt


# Chạy ứng dụng:
python database.py

python dataprocessing.py

python feature_extraction.py

python app/main.py
