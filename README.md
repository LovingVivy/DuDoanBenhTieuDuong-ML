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


