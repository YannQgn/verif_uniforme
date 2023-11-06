# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Détection d'objet en utilisant YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.title("Détection d'objet en utilisant YOLOv8")
st.sidebar.header("ML Model Config")

model_type = st.sidebar.radio(
    "Tâche :", ['Détection'])

confidence = float(st.sidebar.slider(
    "Confiance du modèle", 25, 100, 40)) / 100

if model_type == 'Détection':
    model_path = Path(settings.DETECTION_MODEL)

model = helper.load_model(model_path)

st.sidebar.header("Image/Caméra")
source_radio = st.sidebar.radio(
    "Choisir la source", settings.SOURCES_LIST)

source_img = None
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choisissez une image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        if source_img is None:
            default_image_path = str(settings.DEFAULT_IMAGE)
            default_image = PIL.Image.open(default_image_path)
            st.image(default_image_path, caption="Image par défaut",
                        use_column_width=True)
        else:
            uploaded_image = PIL.Image.open(source_img)
            st.image(source_img, caption="Image téléchargée",
                    use_column_width=True)


    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Image détectée',
                     use_column_width=True)
        else:
            if st.sidebar.button('Détecter les objets'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Image détectée',
                         use_column_width=True)
                try:
                    with st.expander("Résultat :"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("Pas d'image téléchargée !")

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model, 1)

elif source_radio == settings.WEBCAM2:
    helper.play_webcam(confidence, model, 2)

else:
    st.error("Veuillez choisir une source valide !")
