# Nhận Diện Món Ăn Việt Nam

Ứng dụng web sử dụng CNN để nhận diện 10 món ăn truyền thống Việt Nam từ hình ảnh người dùng tải lên.
---

## Tính năng

- Tải ảnh lên dưới dạng (JPG, JPEG, PNG) và nhận diện món ăn bằng AI.
- Hiển thị món ăn dự đoán và độ tin cậy  
- Hiển thị top 3 dự đoán có xác suất cao nhất

---

## Món ăn hỗ trợ

| | | |
|---|---|---|
| Bánh mì | Bánh xèo | Bún bò Huế |
| Cơm tấm | Gỏi cuốn | Hủ tiếu |
| Mì Quảng | Nem chua | Phở |
| Xôi xéo | | |

---

## Cấu trúc thư mục

```
├── app.py    
│         
├── images/
│   ├── train/
│   ├── validate/
│   └── test/
│    
├── notebooks/
│   └── train.ipynb 
│          
├── models/
│   └── food_model.h5
│       
└── requirements.txt
```

---

## Chi tiết mô hình

Mô hình được huấn luyện bằng TensorFlow/Keras trên Google Colab (GPU T4).

**Kiến trúc mô hình:** 
- Input: ảnh RGB kích thước 128×128 
- Data augmentation: Random Flip, Random Rotation (±10%), Random Zoom (±10%)
- 3 khối CNN: Conv2D (32→64→128 filters), BatchNormalization, MaxPooling2D
- GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.5) → Dense(10, Softmax)
- Trainable params: 111,498 

**Quá trình huấn luyện:**
- Dataset: 6699 ảnh train / 956 ảnh validate / 1920 ảnh test với 10 lớp món ăn
- Optimizer: Adam 
- Loss: Categorical Crossentropy
- Epochs: 40 (tự động lưu model tốt nhất bằng `val_accuracy`)

**Kết quả mô hình:**

| Tập dữ liệu | Độ chính xác |
|-------------|--------------|
| Train       | 73.23%       |
| Validation  | 68.51%       |
| Test        | 68.18%       |

---

## Công nghệ sử dụng

- Python 3.12.10
- TensorFlow / Keras
- Streamlit
- Pillow
- NumPy
- Google Colab

---

## Cách chạy

# Cài dependencies
```bash
pip install -r requirements.txt
```

# Chạy ứng dụng
```bash
streamlit run app.py
```

---

## Hướng dẫn sử dụng

1. Chạy ứng dụng bằng lệnh `streamlit run app.py`
2. Upload ảnh món ăn (JPG/PNG/JPEG)
3. Chờ AI phân tích hình ảnh
4. Xem kết quả dự đoán, độ tin cậy, và top 3 dự đoán

---

## Hạn chế

- Các món có hình dạng tương tự nhau (như Phở, Hủ tiếu, Bún bò Huế) dễ bị nhầm lẫn.
- Độ chính xác có thể giảm đối với ảnh mờ, ánh sáng kém, nhiều món ăn trong cùng một ảnh.
- Dataset tương đối nhỏ nên mô hình chưa đạt độ chính xác cao.

