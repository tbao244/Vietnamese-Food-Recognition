import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Nhận diện món ăn Việt Nam", page_icon="🍜")
st.title("🍜 Ứng Dụng Nhận Diện Món Ăn Việt Nam")
st.write("Upload một bức ảnh đồ ăn và để AI đoán xem đó là món gì nhé!")

CLASS_NAMES = [
    'Banh mi', 'Banh xeo', 'Bun bo Hue', 'Com tam', 'Goi cuon', 
    'Hu tieu', 'Mi quang', 'Nem chua', 'Pho', 'Xoi xeo'
]

with st.expander("📌 Xem danh sách các món ăn AI có thể nhận diện"):
    supported_foods = ", ".join(CLASS_NAMES)
    st.info(f"Hiện tại phiên bản này đang hỗ trợ nhận diện **10 món ăn** sau:\n\n{supported_foods}")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/food_model.h5')
model = load_model()

uploaded_file = st.file_uploader("Chọn một bức ảnh đồ ăn...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh bạn vừa tải lên", use_column_width=True)
    st.write("⏳ AI đang phân tích...")
    img = image.resize((128, 128))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0) 
    
    predictions = model.predict(img_array)
    confidence = np.max(predictions[0])
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    
    st.markdown("---")
    st.success(f"🍲 **Dự đoán:** {predicted_class}")
    st.info(f"📊 **Độ tin cậy:** {confidence * 100:.2f}%")