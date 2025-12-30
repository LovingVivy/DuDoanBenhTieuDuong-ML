# Khai báo thư viện
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Hiển thị dữ liệu
df = pd.read_csv(r'C:/Users/Duong/Documents/DataEngineer/tieuDuong/tieuduong/Train.csv')
df.head()
df.tail()

df.info()

# Tiền xử lý dữ liệu
#Xử lý bất thường
df = df.drop(df[df['height'] < 1.0].index).reset_index(drop=True)
df = df.dropna().reset_index(drop=True)
df['diabetic'] = df['diabetic'].map({'No': 0, 'Yes': 1})
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

df = df.copy()
-chỉ số khối trong cơ thể không nhỏ hơn 60
df = df[df['bmi'] <= 60]
-đường trong cơ thể không thể nhỏ hơn 25 mmol/l 
df = df[df['glucose'] <= 25]
-huyết áp
df = df[df['systolic_bp'] >= 70]   # <70 là sốc
df = df[df['systolic_bp'] <= 250] # <= 250 thì đi viện

selected_cols = ['age', 'glucose', 'bmi', 'systolic_bp', 'diastolic_bp', 
                 'family_diabetes', 'hypertensive', 'stroke', 'diabetic']

# Xử lý trùng lặp
num_duplicates = df.duplicated().sum()
print(f"Số lượng dòng trùng lặp: {num_duplicates}")
if num_duplicates > 0:
    display(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)))

# Xử lý giá trị thiếu
-Kiểm tra tổng số giá trị thiếu ở mỗi cột
missing_info = df.isnull().sum()
print("Số lượng giá trị thiếu ở mỗi cột:")
print(missing_info[missing_info > 0])

-Liệt kê các cột số cần điền giá trị
cols_to_fix = ['age', 'pulse_rate', 'systolic_bp', 'diastolic_bp', 'glucose', 'height', 'weight', 'bmi']

-Điền giá trị thiếu bằng trung vị của cột đó
for col in cols_to_fix:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())
print("Đã xử lý xong giá trị thiếu bằng phương pháp điền trung vị.")

-Xử lý nhiễu
def handle_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
    return df_clean

cols_to_check = ['age', 'pulse_rate', 'systolic_bp', 'diastolic_bp', 'glucose', 'height', 'weight', 'bmi']
df = handle_outliers_iqr(df, cols_to_check)
print("Outliers handled using IQR method (Capping).")

print("\nKiểu dữ liệu:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# kiểm tra giá trị
df.nunique().sort_values()
df.isna().sum()
df['diabetic'].value_counts().head(5)

# Thống kê dữ liệu
glucose_mean = df['glucose'].mean()
high_glucose_df = df[df['glucose'] > glucose_mean]
num_high_glucose = len(high_glucose_df)
total_people = len(df)
percentage = (num_high_glucose / total_people) * 100
print(f"Mức Glucose trung bình: {glucose_mean:.2f} mmol/L")
print(f"Số lượng người có mức Glucose cao hơn trung bình: {num_high_glucose} người")
print(f"Tỷ lệ: {percentage:.2f}% trên tổng số dữ liệu")
print("\nThống kê tình trạng bệnh trong nhóm Glucose cao:")
print(high_glucose_df['diabetic'].value_counts())

# Lọc ra những người có chỉ số BMI cao hơn mức trung bình
bmi_mean = df['bmi'].mean()
high_bmi_df = df[df['bmi'] > bmi_mean]
num_high_bmi = len(high_bmi_df)
total_people = len(df)
percentage = (num_high_bmi / total_people) * 100
print(f"Trung bình BMI: {bmi_mean:.2f}")
print(f"Số người có mức BMI > trung bình: {num_high_bmi}")
print(f"Phần trăm: {percentage:.2f}% của tổng dữ liệu")
print("\nDiabetic số người :")
print(high_bmi_df['diabetic'].value_counts())

# biểu đồ tương quan
fig = plt.figure(figsize=(20, 15))
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Ma trận tương quan', fontsize=14)
plt.show()

# biểu đồ hiển thị số người mắc bệnh trong dữ liệu
fig = plt.figure(figsize=(20, 15))
plt.subplot(3, 3, 9)
sizes = df['diabetic'].value_counts()
plt.pie(sizes, labels=['Không bị', 'Bị tiểu đường'], autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], startangle=90)
plt.title('Tỷ lệ bị tiểu đường trong dữ liệu', fontweight='bold')
plt.tight_layout()
plt.show()

# Biểu đồ hiển thị các feature quan trọng
-Chỉ giữ lại cột số để train model
df_model = df.select_dtypes(include=['float64', 'int64'])
X = df_model.drop('diabetic', axis=1)
y = df_model['diabetic']
-Feature importance
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 7))
sns.barplot(x=feat_imp.values[:10], y=feat_imp.index[:10], palette="viridis")
plt.title('TOP 10 YẾU TỐ QUAN TRỌNG NHẤT DỰ ĐOÁN BỆNH TIỂU ĐƯỜNG\n'
          '(theo Random Forest)', fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
plt.xlabel('Độ quan trọng', fontsize=14)
plt.ylabel('Yếu tố', fontsize=14)
for i, v in enumerate(feat_imp.values[:10]):
    plt.text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold', fontsize=12)
plt.xlim(0, max(feat_imp.values[:10]) * 1.15)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# Chia dữ liệu thành 2 phần test & train
df_model = df.select_dtypes(include=['number'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
cols_to_scale = ['age', 'pulse_rate', 'systolic_bp', 'diastolic_bp', 'glucose',
                 'height', 'weight', 'bmi']
X_train_scaled = X_train.copy() 
X_test_scaled  = X_test.copy()
X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test_scaled[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])

# Smote dữ liệu
smote = SMOTE(random_state=42)
-sinh dữ liệu
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("Sau khi SMOTE:", y_train_smote.value_counts())


# Huấn luyện mô hình
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, PrecisionRecallDisplay,
    ConfusionMatrixDisplay
)
def evaluate_clf(model, model_name, X_test, y_test):
    """
    Hàm đánh giá mô hình phân loại.
    Trả về: accuracy, precision, recall, f1 + vẽ 2 biểu đồ quan trọng.
    (Đã loại bỏ ROC AUC và ROC Curve theo yêu cầu)
    """

    y_pred = model.predict(X_test)

    # Lấy xác suất dự đoán (chỉ dùng cho Precision-Recall Curve)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    print(f"\n===== {model_name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    # Chỉ vẽ 2 biểu đồ
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axs[0])
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axs[1])
    axs[0].set_title("Precision-Recall Curve")
    axs[1].set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Lưu vào bảng kết quả (không có roc_auc)
    return {
        "modelname": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }
-Cập nhật DataFrame để phù hợp
model_metrics = pd.DataFrame(columns=[
    "modelname", "accuracy", "precision", "recall", "f1"
])

# Model Logistic Regression (trước somte)
logreg_before = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
logreg_before.fit(X_train_scaled, y_train)
result1 = evaluate_clf(logreg_before, "Logistic Regression (Trước SMOTE)", X_test_scaled, y_test)

# Sau smote
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f"Sau SMOTE:")
logreg_after = LogisticRegression(max_iter=1000, random_state=42) 
logreg_after.fit(X_train_smote, y_train_smote)
result2 = evaluate_clf(logreg_after, "Logistic Regression (Sau SMOTE)", X_test_scaled, y_test)

# Mô hình Random Forest
X_res, y_res = smote.fit_resample(X_train_scaled, y_train)
rf = RandomForestClassifier(n_estimators=800, random_state=42, n_jobs=-1)
rf.fit(X_res, y_res)
metrics = evaluate_clf(rf, 'Random Forest', X_test_scaled, y_test)
model_metrics = pd.concat([model_metrics, pd.DataFrame([metrics])], ignore_index=True)

## Đánh giá mô hình
import pandas as pd

-Tạo lại bảng sạch (chỉ còn các cột cần thiết, không có roc_auc)
model_metrics = pd.DataFrame(columns=["modelname", "accuracy", "precision", "recall", "f1"])
-Áp dụng SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)
-Logistic Regression
log = LogisticRegression(max_iter=1000, random_state=42)
log.fit(X_res, y_res)
metrics_log = evaluate_clf(log, "Logistic", X_test_scaled, y_test)
model_metrics = pd.concat([model_metrics, pd.DataFrame([metrics_log])], ignore_index=True)
-Random Forest
rf = RandomForestClassifier(n_estimators=800, random_state=42, n_jobs=-1)
rf.fit(X_res, y_res)
metrics_rf = evaluate_clf(rf, "Random Forest", X_test_scaled, y_test)
model_metrics = pd.concat([model_metrics, pd.DataFrame([metrics_rf])], ignore_index=True)
-Chuẩn bị hiển thị bảng
model_metrics = model_metrics.set_index('modelname')
-Sắp xếp theo f1 (hoặc bạn có thể chọn accuracy/recall tùy ý, vì không còn roc_auc)
-Ở đây mình chọn sort theo f1 descending - thường là metric quan trọng nhất trong bài toán imbalance
display(model_metrics.sort_values('f1', ascending=False)
        .style.background_gradient(cmap='Greens', subset=['f1'])  # Tô màu cột f1
        .highlight_max(axis=0, props='font-weight:bold; color:gold;')  # Highlight giá trị max từng cột
        .set_caption("SO SÁNH 2 MÔ HÌNH SAU SMOTE")
        .format("{:.4f}"))

# Demo
import joblib
import pandas as pd
import numpy as np

joblib.dump(rf, 'model_tieuduong_best.pkl')
joblib.dump(scaler, 'scaler.pkl') 
joblib.dump(X_train_scaled.columns, 'columns.pkl')

def du_doan_tieu_duong(
    age, gender, pulse_rate, systolic_bp, diastolic_bp,
    glucose, height, weight, bmi,
    family_diabetes, hypertensive, family_hypertension,
    cardiovascular_disease, stroke
):
    model = joblib.load('model_tieuduong_best.pkl')
    scaler = joblib.load('scaler.pkl')
    cols = joblib.load('columns.pkl')
    
    # Tạo dữ liệu 1 người
    data = pd.DataFrame([[
        age, gender, pulse_rate, systolic_bp, diastolic_bp,
        glucose, height, weight, bmi,
        family_diabetes, hypertensive, family_hypertension,
        cardiovascular_disease, stroke
    ]], columns=cols)
    
    data_scaled = data.copy()
    data_scaled[cols_to_scale] = scaler.transform(data[cols_to_scale])
    
    # Dự đoán
    prob = model.predict_proba(data_scaled)[0][1]
    pred = "CÓ NGUY CƠ TIỂU ĐƯỜNG" if prob >= 0.5 else "BÌNH THƯỜNG"
    
    print(f"XÁC SUẤT BỊ TIỂU ĐƯỜNG: {prob:.1%}")
    print(f"KẾT LUẬN: {pred}")
    if prob >= 0.7:
        print("CẢNH BÁO MỨC ĐỎ - Khuyên đi khám ngay!")
    elif prob >= 0.5:
        print("CẢNH BÁO MỨC VÀNG - Cần theo dõi sát")
  
du_doan_tieu_duong(
    age=42, gender=1, pulse_rate=66, systolic_bp=110, diastolic_bp=73,
    glucose=5.88, height=1.63, weight=72.2, bmi=25.75,
    family_diabetes=0, hypertensive=1, family_hypertension=0,
    cardiovascular_disease=0, stroke=0
)
