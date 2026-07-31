from fastapi import FastAPI, UploadFile , File
from fastapi.responses import JSONResponse
import shutil
import numpy as np
import cv2
import os
from keras.models import load_model
from collections import deque
import config

app = FastAPI(title='Violence Detection API')

UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

# loadong the model
MODEL_PATH = 'model/MobBiLSTM_model.h5'
MoBiLSTM_model = load_model(MODEL_PATH)


# Frame extraction Function

def extract_frames(video_path , sequence_length=config.SEQUENCE_LENGTH):
    frames_list =[]
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(video_frames_count // sequence_length, 1)

    for frame_counter in range(sequence_length):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES,frame_counter * skip_frames_window)
        success , frame = video_reader.read()

        if not success:
            break
        resized_frame = cv2.resize(frame,(config.IMAGE_WIDTH,config.IMAGE_HEIGHT))  
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)

    video_reader.release()
    return np.array(frames_list)




# Prediction Function

def predict_video_class(video_path):
    frames = extract_frames(video_path)
    if len(frames) != config.SEQUENCE_LENGTH:
        return {'error':f'Video has insufficient frames ({len(frames)}). Minimum required: {config.SEQUENCE_LENGTH}'}

    frames_expanded = np.expand_dims(frames,axis=0)
    predictions = MoBiLSTM_model.predict(frames_expanded)[0]
    predicted_label_index = np.argmax(predictions)
    predicted_class = config.CLASSES_LIST[predicted_label_index]
    confidence = float(predictions[predicted_label_index])

    return {
        "prediction": predicted_class,
        "confidence": confidence
    }


#Routes

@app.get('/')
def home():
    return {'message':'Violence Detection API Running'}


@app.post('/predict')
async def predict(file:UploadFile=File(...)):
    try:
        filepath = os.path.join(UPLOAD_DIR,file.filename)
        with open(filepath,'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_video_class(filepath)
        os.remove(filepath)

        return JSONResponse(result)
    except Exception as e:
        return JSONResponse(
            {
                'error':str(e)
            },
            status_code=500
        )