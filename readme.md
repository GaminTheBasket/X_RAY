# Chest X-Ray Classification (Pneumonia vs Normal)

Dự án này thực hiện **phân loại ảnh X-quang phổi** thành hai nhãn:  
- **PNEUMONIA**  
- **NORMAL**

---

## Các bước thực hiện
- Tiền xử lý ảnh (resize, histogram equalization).
- Tạo dataset cân bằng theo tỷ lệ 80-10-10.
- Data augmentation khác nhau cho từng lớp.
- Trích xuất đặc trưng:  
  - HOG (Histogram of Oriented Gradients)  
  - LBP (Local Binary Pattern)  
  - GLCM (Gray Level Co-occurrence Matrix)  
  - Color Features (Mean, Std, Skewness, Kurtosis)  
- Cân bằng dữ liệu bằng **SMOTE**.
- Giảm chiều dữ liệu bằng **PCA**.
- Huấn luyện nhiều mô hình ML (SVM, Random Forest, Logistic Regression, Naive Bayes, KNN, Decision Tree).
- Đánh giá mô hình bằng:  
  - Accuracy  
  - Confusion Matrix  
  - ROC-AUC  
  - Precision-Recall Curve  

---

## Cấu trúc dự án
├── chest_xray.py # Mã nguồn chính: tiền xử lý, trích xuất đặc trưng, huấn luyện, đánh giá
├── chest_xray_ml_model.pkl # Mô hình ML đã huấn luyện
├── chest_xray_scaler.pkl # Scaler để chuẩn hóa dữ liệu
├── chest_xray_pca.pkl # PCA transformer
├── README.md # Tài liệu hướng dẫn

---

## Cài đặt môi trường

Yêu cầu Python >= 3.10. Cài các thư viện:

```bash
#pip install scikit-learn scikit-image imbalanced-learn opencv-python matplotlib seaborn joblib
Dataset

Do dataset X-ray có dung lượng lớn, không được đính kèm repo.
Bạn có thể tải về từ Kaggle:
Chest X-ray Pneumonia Dataset

Cấu trúc thư mục mong đợi:
chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
