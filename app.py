
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AI Traffic Sign Recognition",
    page_icon="🚦"
)

# Load trained CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "traffic_sign_cnn.keras",
        compile=False
    )

model = load_model()

# Traffic sign information
sign_info = {
    0: ("Speed Limit 20 km/h",
        "Maximum permitted speed is 20 km/h.",
        "Reduce your speed and drive carefully."),

    1: ("Speed Limit 30 km/h",
        "Maximum permitted speed is 30 km/h.",
        "Reduce your speed and drive carefully."),

    2: ("Speed Limit 50 km/h",
        "Maximum permitted speed is 50 km/h.",
        "Do not exceed 50 km/h."),

    14: ("Stop",
         "You must come to a complete stop.",
         "Stop and check the road before proceeding."),

    17: ("No Entry",
         "Entry is prohibited.",
         "Do not enter this road."),

    18: ("General Caution",
         "Warning of a potentially dangerous situation.",
         "Slow down and drive carefully."),

    25: ("Road Work",
         "Road construction or maintenance is ahead.",
         "Slow down and follow temporary signs."),

    27: ("Pedestrians",
         "Pedestrians may be present ahead.",
         "Slow down and watch for pedestrians."),

    40: ("Roundabout",
         "A roundabout is ahead.",
         "Slow down and follow the roundabout direction.")
}

# Website title
st.title("🚦 AI-Based Traffic Sign Recognition")
st.write("Upload a traffic sign image and let the CNN model recognize it.")

# Image upload
uploaded_file = st.file_uploader(
    "Upload a traffic sign image",
    type=["jpg", "jpeg", "png", "jfif"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Traffic Sign",
        width=350
    )

    if st.button("🔍 Recognize Traffic Sign"):

        # Resize image
        image_resized = image.resize((32, 32))

        # Convert to NumPy array
        img_array = np.array(image_resized)

        # Normalize
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array, verbose=0)

        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        st.subheader("🚦 Recognition Result")

        if predicted_class in sign_info:

            name, meaning, alert = sign_info[predicted_class]

            st.success(f"Sign: {name}")

            st.write(f"**Confidence:** {confidence:.2f}%")

            st.info(f"📌 **Meaning:** {meaning}")

            st.warning(f"⚠️ **Safety Alert:** {alert}")

        else:

            st.write(f"Predicted Class: {predicted_class}")

            st.write(f"Confidence: {confidence:.2f}%")

            st.warning(
                "Information for this class is not available."
            )
