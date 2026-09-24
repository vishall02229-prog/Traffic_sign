# Traffic Sign Classification - Streamlit

This project is a Streamlit version of the original Flask traffic-sign classifier.

## Important compatibility note

The included `Traffic.h5` model was saved with an older TensorFlow/Keras format. The project therefore pins **Python 3.11 + TensorFlow 2.15.1**. Do not change TensorFlow to the latest version unless the model is converted/retrained.

## Run locally

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload all files in this folder.
3. In Streamlit Community Cloud, choose the repository and `app.py` as the main file.
4. Keep `runtime.txt` in the repository so the app uses Python 3.11.
5. Deploy.

The app accepts JPG/JPEG/PNG images, resizes them to 30x30 RGB, normalizes pixels to 0-1, and predicts one of 43 traffic-sign classes.
