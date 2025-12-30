import joblib
import pandas as pd

# Load model và các file cần thiết
model = joblib.load("../model_tieuduong_best.pkl")
scaler = joblib.load("../scaler.pkl")
columns = joblib.load("../columns.pkl")

cols_to_scale = [
    'age', 'pulse_rate', 'systolic_bp', 'diastolic_bp',
    'glucose', 'height', 'weight', 'bmi'
]
def demo():
    # Dữ liệu demo (1 bệnh nhân mẫu)
    sample = pd.DataFrame([[
        42, 1, 66, 110, 73,
        5.88, 1.63, 72.2, 25.75,
        0, 1, 0, 0, 0
    ]], columns=columns
                         )
    # Chuẩn hóa
    sample_scaled = sample.copy()
    sample_scaled[cols_to_scale] = scaler.transform(sample[cols_to_scale])
  
    # Dự đoán
    prob = model.predict_proba(sample_scaled)[0][1]
    result = "CÓ NGUY CƠ TIỂU ĐƯỜNG" if prob >= 0.5 else "BÌNH THƯỜNG"

    print("===== DEMO DỰ ĐOÁN TIỂU ĐƯỜNG =====")
    print(f"Xác suất mắc bệnh: {prob:.2%}")
    print(f"Kết luận: {result}")

if __name__ == "__main__":
    demo()

