# face_recognition_model.py
import os
import cv2
import face_recognition
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle 

def capture_images(user_name, dataset_path):
    """Captura imagens da webcam e salva no diretório especificado para o usuário."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao acessar a câmera!")
        return

    print(f"Capturando imagens para o usuário '{user_name}'. Olhe para a câmera e pressione 'q' para sair.")
    max_images = 30 
    
    user_path = os.path.abspath(os.path.join(dataset_path, user_name))
    
    if not os.path.exists(user_path):
        print(f"Criando diretório para o usuário: {user_path}")
        os.makedirs(user_path)

    count = 0
    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar imagem da webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]
                image_path = os.path.join(user_path, f"{user_name}_{count}.jpg")
                cv2.imwrite(image_path, face_image)
                print(f"Imagem salva: {image_path}")
                count += 1
        else:
            print("Nenhum rosto detectado.")

        cv2.imshow('Captura de Imagens', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def prepare_dataset(dataset_path):
    """Prepara o dataset, convertendo as imagens em encodings."""
    X, y = [], []
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                try:
                    image = cv2.imread(image_path)
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_image)
                    if face_locations:
                        encodings = face_recognition.face_encodings(rgb_image, face_locations)
                        for encoding in encodings:
                            X.append(encoding)
                            y.append(person_name)
                except Exception as e:
                    print(f"Erro ao processar a imagem {image_path}: {e}")
    return np.array(X), np.array(y)

def train_model(dataset_path):
    """Treina o modelo KNN com o dataset de imagens."""
    X, y = prepare_dataset(dataset_path)

    if X.size == 0:
        print("Erro: Não há dados suficientes para treinamento!")
        return None

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn

def live_recognition(model):
    """Reconhecimento em tempo real usando a webcam."""
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Erro ao acessar a câmera!")
        return

    print("Reconhecimento ao vivo iniciado. Pressione 'q' para encerrar.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Falha ao capturar imagem. Verifique a câmera.")
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            try:
                name = "Desconhecido"
                if model:
                    matches = model.predict([face_encoding])
                    if matches:
                        name = matches[0]
            except Exception as e:
                print(f"Erro durante a previsão: {e}")
                name = "Erro"

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Reconhecimento em Tempo Real', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Reconhecimento encerrado pelo usuário.")
            break

    video_capture.release()
    cv2.destroyAllWindows()


def save_model(model, model_path):
    """Salva o modelo treinado em um arquivo."""
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Modelo salvo em {model_path} com sucesso!")
    except Exception as e:
        print(f"Erro ao salvar o modelo: {e}")
