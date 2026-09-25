
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

    0: {
        "name": "Speed Limit 20 km/h",
        "meaning": "Maximum permitted speed is 20 km/h.",
        "alert": "Keep your speed at or below 20 km/h."
    },

    1: {
        "name": "Speed Limit 30 km/h",
        "meaning": "Maximum permitted speed is 30 km/h.",
        "alert": "Keep your speed at or below 30 km/h."
    },

    2: {
        "name": "Speed Limit 50 km/h",
        "meaning": "Maximum permitted speed is 50 km/h.",
        "alert": "Keep your speed at or below 50 km/h."
    },

    3: {
        "name": "Speed Limit 60 km/h",
        "meaning": "Maximum permitted speed is 60 km/h.",
        "alert": "Keep your speed at or below 60 km/h."
    },

    4: {
        "name": "Speed Limit 70 km/h",
        "meaning": "Maximum permitted speed is 70 km/h.",
        "alert": "Keep your speed at or below 70 km/h."
    },

    5: {
        "name": "Speed Limit 80 km/h",
        "meaning": "Maximum permitted speed is 80 km/h.",
        "alert": "Keep your speed at or below 80 km/h."
    },

    6: {
        "name": "End of Speed Limit 80 km/h",
        "meaning": "The previous 80 km/h speed restriction has ended.",
        "alert": "Follow the next applicable speed restriction."
    },

    7: {
        "name": "Speed Limit 100 km/h",
        "meaning": "Maximum permitted speed is 100 km/h.",
        "alert": "Keep your speed at or below 100 km/h."
    },

    8: {
        "name": "Speed Limit 120 km/h",
        "meaning": "Maximum permitted speed is 120 km/h.",
        "alert": "Keep your speed at or below 120 km/h."
    },

    9: {
        "name": "No Passing",
        "meaning": "Overtaking other vehicles is prohibited.",
        "alert": "Do not overtake vehicles in this section."
    },

    10: {
        "name": "No Passing for Heavy Vehicles",
        "meaning": "Heavy vehicles are prohibited from overtaking.",
        "alert": "Heavy vehicles should not overtake here."
    },

    11: {
        "name": "Right-of-Way at Next Intersection",
        "meaning": "You have priority at the upcoming intersection.",
        "alert": "Proceed carefully and watch for other road users."
    },

    12: {
        "name": "Priority Road",
        "meaning": "You are travelling on a priority road.",
        "alert": "Continue carefully and observe other traffic."
    },

    13: {
        "name": "Yield",
        "meaning": "Give way to traffic with priority.",
        "alert": "Slow down and yield when necessary."
    },

    14: {
        "name": "Stop",
        "meaning": "You must come to a complete stop.",
        "alert": "Stop completely before proceeding."
    },

    15: {
        "name": "No Vehicles",
        "meaning": "Vehicles are not permitted beyond this sign.",
        "alert": "Do not enter with a vehicle."
    },

    16: {
        "name": "No Vehicles Over 3.5 Tons",
        "meaning": "Vehicles exceeding 3.5 tons are prohibited.",
        "alert": "Heavy vehicles must not enter."
    },

    17: {
        "name": "No Entry",
        "meaning": "Entry is prohibited from this direction.",
        "alert": "Do not enter this road from this direction."
    },

    18: {
        "name": "General Caution",
        "meaning": "There is a general hazard or danger ahead.",
        "alert": "Slow down and drive with extra caution."
    },

    19: {
        "name": "Dangerous Curve Left",
        "meaning": "A dangerous curve to the left is ahead.",
        "alert": "Reduce speed and prepare for the left curve."
    },

    20: {
        "name": "Dangerous Curve Right",
        "meaning": "A dangerous curve to the right is ahead.",
        "alert": "Reduce speed and prepare for the right curve."
    },

    21: {
        "name": "Double Curve",
        "meaning": "Two successive curves are ahead.",
        "alert": "Slow down and drive carefully through the curves."
    },

    22: {
        "name": "Bumpy Road",
        "meaning": "The road surface ahead may be uneven or bumpy.",
        "alert": "Reduce speed and drive carefully."
    },

    23: {
        "name": "Slippery Road",
        "meaning": "The road may be slippery.",
        "alert": "Slow down and drive carefully."
    },

    24: {
        "name": "Road Narrows on the Right",
        "meaning": "The road becomes narrower on the right side.",
        "alert": "Reduce speed and watch for nearby vehicles."
    },

    25: {
        "name": "Road Work",
        "meaning": "Road construction or maintenance work is ahead.",
        "alert": "Slow down and watch for workers and obstacles."
    },

    26: {
        "name": "Traffic Signals",
        "meaning": "Traffic signals are ahead.",
        "alert": "Be prepared to stop and follow the traffic lights."
    },

    27: {
        "name": "Pedestrians",
        "meaning": "Pedestrians may be present or crossing ahead.",
        "alert": "Slow down and watch carefully for pedestrians."
    },

    28: {
        "name": "Children Crossing",
        "meaning": "Children may be crossing the road ahead.",
        "alert": "Slow down and watch carefully for children."
    },

    29: {
        "name": "Bicycles Crossing",
        "meaning": "Bicycles may be crossing or entering the road.",
        "alert": "Slow down and watch for cyclists."
    },

    30: {
        "name": "Ice or Snow",
        "meaning": "Ice or snow may make the road slippery.",
        "alert": "Reduce speed and drive carefully."
    },

    31: {
        "name": "Wild Animals Crossing",
        "meaning": "Wild animals may cross the road ahead.",
        "alert": "Slow down and watch for animals."
    },

    32: {
        "name": "End of All Speed and Passing Restrictions",
        "meaning": "The previous speed and passing restrictions have ended.",
        "alert": "Follow the next applicable traffic signs."
    },

    33: {
        "name": "Turn Right Ahead",
        "meaning": "You must turn right ahead.",
        "alert": "Prepare to turn right and follow the road safely."
    },

    34: {
        "name": "Turn Left Ahead",
        "meaning": "You must turn left ahead.",
        "alert": "Prepare to turn left and follow the road safely."
    },

    35: {
        "name": "Ahead Only",
        "meaning": "Traffic must continue straight ahead.",
        "alert": "Continue straight and follow the marked direction."
    },

    36: {
        "name": "Go Straight or Right",
        "meaning": "You may continue straight or turn right.",
        "alert": "Follow one of the permitted directions."
    },

    37: {
        "name": "Go Straight or Left",
        "meaning": "You may continue straight or turn left.",
        "alert": "Follow one of the permitted directions."
    },

    38: {
        "name": "Keep Right",
        "meaning": "Keep to the right side of the traffic island or obstacle.",
        "alert": "Pass safely on the right."
    },

    39: {
        "name": "Keep Left",
        "meaning": "Keep to the left side of the traffic island or obstacle.",
        "alert": "Pass safely on the left."
    },

    40: {
        "name": "Roundabout",
        "meaning": "A roundabout is ahead and traffic must follow its direction.",
        "alert": "Slow down and follow the roundabout correctly."
    },

    41: {
        "name": "End of No Passing",
        "meaning": "The previous no-passing restriction has ended.",
        "alert": "Follow the applicable road rules and signs."
    },

    42: {
        "name": "End of No Passing for Heavy Vehicles",
        "meaning": "The previous no-passing restriction for heavy vehicles has ended.",
        "alert": "Heavy vehicles should follow the applicable traffic rules."
    }
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
