from keras.models import load_model
import cv2
import numpy as np
import cvzone

np.set_printoptions(suppress=True)

model = load_model("keras_Model.h5", compile=False)

class_names = ['normal', 'tumor']

img = cv2.imread('yes/Y103.jpg')

if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
    image = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    texto1 = f"Classe: {class_name}"
    texto2 = f"Taxa de acerto: {str(np.round(confidence_score * 100))[:-2]} %"

    print(texto1, texto2)

    cvzone.putTextRect(img, texto1, (50, 50), scale=3)
    cvzone.putTextRect(img, texto2, (50, 100), scale=3)

    # Listen to the keyboard for presses.
    cv2.imshow('Detector de Tumor Cerebral', img)
    cv2.waitKey(0)
else:
    print("Erro: Imagem inválida ou vazia.")
