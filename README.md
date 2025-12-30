## Dự đoán nguy cơ mắc bệnh tiều đường - Machine learning cơ bản
- Bài toán: Dự đoán mắc bệnh tiểu đường
Bệnh tiểu đường (Diabetic) là một trong những bệnh mãn tính phổ biến, ảnh hưởng nghiêm trọng đến sức khỏe con người
Việc phát hiện sớm nguy cơ mắc bệnh dựa trên các chỉ số y tế giúp hỗ trợ bác sĩ và bệnh nhân trong việc phòng ngừa và điều trị kịp thời.

 Mục tiêu: Xây dựng mô hình học máy để dự đoán khả năng mắc bệnh tiểu đường của một người
 - So sánh các mô hình thực hiện để đánh giá độ phù hợp và hiệu quả
 - Xây dựng pipeline hoàn chỉnh từ tiền xử lý dữ liệu -> huấn luyện -> đánh giá -> dư đoán
-------------------------------------------------------------------------------------------------------
## Dataset: 
Nguồn dữ liệu DiaBD A Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh
Nguồn tải dataset: https://data.mendeley.com/datasets/m8cgwxs9s6/2

Mô tả các cột:
- age: Tuổi
- gender: Giới tính
- pulse_rate: Nhịp tim (Lần/phút)
- systolic_bp: Huyết áp tâm thu
- diastolic_bp: Huyết áp tâm trương
- glucose: Mức đường huyết
- height: Chiều cao
- height: Cân nặng
- bmi: Chỉ số khối cơ thể
- family_diabetes: Tiền sử bị bệnh tiểu đường trong gia đình
- hypertensive: Cá nhân mắc bệnh tăng huyết áp
- family_hypertension: Tiền sử tăng huyết áp trong gia đình
- cardiovascular_disease: Cá nhân mắc bệnh tim mạch
- stroke: Cá nhân mắc bệnh độ quỵ
# Đầu ra
- diabetic: Có bị tiểu đường không
------------------------------------------------------------------------------------------------------
## Pipeline xử lý dữ liệu
Pipeline của hệ thống gồm các bước sau:
Data Loading -> 
Data Cleaning & Preprocessing ->
Train/Test Split ->
Train Model ->
Evaluate Model 0 ->
Inference (Dự đoán dữ liệu mới)

Chi tiết:
- xử lý giá trị thiểu
- Chuẩn hóa dự liệu bằng StandardScaler
- Chia tập train & test
- Huấn luyện các mô hình
- Đánh giá bằng các metric phù hợp
  * sử dụng các chỉ số: Accuracy, Confusion Matrix (Ma trận nhầm lẫn)
- Dự đoán: Xây dựng hàm 'Du_doan_tieu_duong()' để nhập thông số và trả về xác suất mắc bệnh
-----------------------------------------------------------------------------------------------------
## Các mô hình sử dụng
*Random Forest: được chọn sử dụng trong bài vì khả năng xử lý tốt dữ liệu bảng, chống overfitting tốt hơn Decision Tree và cho độ chính xác cao, phù hợp với yêu cầu bài toán.
*Logistic Regression: (Mô hình chính) được chọn trong bài vì khả năng dự đoán đúng người bệnh thực sự, nó 
có tỉ lệ bỏ sót của nó thấp hơn random forest.

## Kết quả
                    Accuracy    Precision   Recall  F1
Logistic Regression:0.9137      0.3462      0.4030  0.3724
Random Forest:      0.7884      0.2023      0.7910  0.3222

- Accuracy (Độ chính xác):Metric này đo lường độ đúng đắn tổng thể của các dự đoán mô hình. Nó là tỷ lệ các trường hợp dự đoán đúng (cả true positive và true negative) trên tổng số trường hợp
- Precision (Độ chính xác dự đoán danh tính): Metric này cho biết bao nhiêu phần trăm các trường hợp dự đoán dương tính (ví dụ: dự đoán có nguy cơ) thực sự là dương tính. Nó tập trung vào việc giảm thiểu false positive
- Recall (Độ nhạy): Metric này đo lường bao nhiêu phần trăm các trường hợp dương tính thực tế (ví dụ: người thực sự có nguy cơ) mà mô hình xác định đúng. Nó tập trung vào việc giảm thiểu false negative
- F1 (Điểm F1): Đây là trung bình hài hòa của Precision và Recall, cung cấp một measure cân bằng khi bạn quan tâm đến cả hai. Nó hữu ích khi các lớp không cân bằng
----------------------------------------------------------------------------
## Hướng dẫn chạy
- Hướng dẫn chạy model: Cần cài đặt Python trên thiết bị của bạn
Cài python: https://www.python.org/downloads/ 

- Môi trường chuẩn bị: VsCode hoặc googlelab
- Trước khi chạy cần cài các thư viện phù hợp, cài trên terminal nhấn tổ hợp phím win + R và gõ cmd và gõ:
  * numpy: pip install numpy
  * pandas: pip install pandas
  * matplotlib: pip install matplotlib
  * Smote: pip install imbalanced-learn
  * sklearn: pip install scikit-learn
 
## Chạy huấn luyện Training
1. Đảm bảo file dữ liệu Train.csv nằm cùng thư mục với file notebook DuDoanTieuDuong.ipynb.
2. Mở file DuDoanTieuDuong.ipynb bằng Jupyter Notebook hoặc VS Code, gg 
3. Chọn Run All hoặc chạy lần lượt các cell từ trên xuống dưới để thực hiện quy trình:
   * Load dữ liệu và Tiền xử lý (Làm sạch, xử lý Outlier, điền giá trị thiếu).
   * Trực quan hóa dữ liệu EDA
   * Huấn luyện mô hình (RF, LR)

 ## Chạy Demo
Ở cuối file notebook, một hàm dự đoán du_doan_tieu_duong() đã được viết sẵn để bạn kiểm tra thử trên dữ liệu mới.
Cách sử dụng: Nhập các thông số sức khỏe vào hàm để nhận kết quả dự báo:
# Ví dụ chạy thử
du_doan_tieu_duong(
    age=45,                 # Tuổi
    gender=1,               # Giới tính (0: Nữ, 1: Nam)
    pulse_rate=70,          # Nhịp tim
    systolic_bp=120,        # Huyết áp tâm thu
    diastolic_bp=80,        # Huyết áp tâm trương
    glucose=7.5,            # Đường huyết (mmol/L) - Quan trọng
    height=1.65,            # Chiều cao (m)
    weight=65.0,            # Cân nặng (kg)
    bmi=23.8,               # BMI
    family_diabetes=1,      # Tiền sử gia đình (0/1)
    hypertensive=0,         # Huyết áp cao (0/1)
    family_hypertension=0,  # Tiền sử gia đình HA cao (0/1)
    cardiovascular_disease=0, # Bệnh tim mạch (0/1)
    stroke=0                # Đột quỵ (0/1)
)

## Cấu trúc thư mục dự án
Cấu trúc tổ chức file
DuDoanBenhTieuDuong-ML/
│
├── app/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   └── utils.py
│
├── demo/
│   ├── demo_inference.py
│   └── demo.ipynb
│
├── data/
│   ├── sample_data.csv
│   └── README.md
│
├── reports/
│   └── BaoCao_DuDoanTieuDuong.pdf
│
├── slides/
│   └── Slide_DuDoanTieuDuong.pptx
│
├── requirements.txt
├── README.md
└── .gitignore
       

Sở đồ cáu trúc dự án để đảm báo mã nguồn chạy chính xác:
* Load dữ liệu
* Tiền xử lý
* Vẽ biểu đồ
* Huấn luyện mô hình
* Áp dụng thuật toán
* Ứng dụng demo
-------------------------------------------------------------------------------------------------------
## Tác giả
Họ tên: Phạm Tùng Dương
Mã sinh viên: 10123071
Mã lớp: 12423TN
