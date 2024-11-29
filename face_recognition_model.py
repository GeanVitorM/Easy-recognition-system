import os
import cv2
import face_recognition
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def capture_images(user_name, dataset_path):
    """Captura imagens da webcam e salva no diretório especificado para o usuário."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao acessar a câmera!")
        return

    print(f"Capturando imagens para o usuário '{user_name}'. Olhe para a câmera e pressione 'q' para sair.")
    count = 0
    max_images = 30  # Número máximo de imagens
    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar imagem da webcam.")
            break

        # Verifica se a imagem está vazia ou com dimensões erradas
        if frame is None or frame.size == 0:
            print("Imagem inválida capturada, tentando novamente.")
            continue

        # Converte a imagem de BGR (OpenCV) para RGB (usado pelo face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # Detectar rostos
            face_locations = face_recognition.face_locations(rgb_frame)
        except Exception as e:
            print(f"Erro ao detectar rostos: {e}")
            continue

        if face_locations:
            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]
                image_path = os.path.join(dataset_path, f"{user_name}_{count}.jpg")
                while os.path.exists(image_path):  # Evita sobrescrever as imagens
                    count += 1
                    image_path = os.path.join(dataset_path, f"{user_name}_{count}.jpg")
                cv2.imwrite(image_path, face_image)
                print(f"Imagem salva como: {image_path}")
                count += 1
        else:
            print("Nenhum rosto detectado. Tente novamente.")

        # Exibe o frame com a detecção da câmera
        cv2.imshow('Captura de Imagens', frame)

        # Encerra quando pressionar a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def prepare_dataset(dataset_path):
    """Prepara o dataset, convertendo as imagens em encodings."""
    X, y = [], []
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):  # Verifica se é um diretório (nome do usuário)
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Erro ao ler a imagem {image_path}. Pulando.")
                        continue
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_image)
                    if not face_locations:
                        print(f"Nenhuma face detectada na imagem {image_path}. Pulando.")
                        continue
                    encodings = face_recognition.face_encodings(rgb_image, face_locations)
                    for encoding in encodings:
                        X.append(encoding)
                        y.append(person_name)
                except Exception as e:
                    print(f"Erro ao processar a imagem {image_path}: {e}")
                    continue

    return np.array(X), np.array(y)

def train_model(dataset_path):
    """Treina o modelo KNN com o dataset de imagens."""
    X, y = prepare_dataset(dataset_path)

    if X.size == 0:
        print("Erro: Não há dados suficientes para treinamento!")
        return None

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    print("Modelo treinado com sucesso!")
    return knn

def live_recognition(model):
    """Reconhece rostos em tempo real usando a webcam."""
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Erro ao acessar a câmera!")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Falha ao capturar imagem da webcam.")
            break
        
        rgb_frame = frame[:, :, ::-1]  # Converte de BGR para RGB
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Desconhecido"
            # Verifica se há correspondência no modelo
            if model is not None:
                matches = model.predict([face_encoding])
                if len(matches) > 0:
                    name = matches[0]
            # Desenha um retângulo ao redor do rosto
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Exibe o frame com a detecção
        cv2.imshow('Reconhecimento em Tempo Real', frame)

        # Encerra quando pressionar a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
