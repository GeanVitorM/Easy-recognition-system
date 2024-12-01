import os
import cv2
import numpy as np
import face_recognition
from sklearn.svm import SVC
import pickle

# Função para cadastro de fotos de usuários
def cadastrar_usuario(usuario_nome, dataset_path, num_fotos=30):
    """Cadastra um novo usuário capturando fotos e salvando no dataset."""
    usuario_path = os.path.join(dataset_path, usuario_nome)
    os.makedirs(usuario_path, exist_ok=True)
    
    camera = cv2.VideoCapture(0)
    contador_fotos = 0

    print(f"Iniciando captura de imagens para o usuário: {usuario_nome}")
    
    while contador_fotos < num_fotos:
        ret, frame = camera.read()
        if not ret:
            print("Erro ao acessar a câmera.")
            break

        cv2.imshow("Captura de Imagens", frame)
        k = cv2.waitKey(1) & 0xFF
        
        if k == ord('q'):  # Permitir sair manualmente
            print("Captura interrompida pelo usuário.")
            break

        # Salvar a imagem
        foto_path = os.path.join(usuario_path, f"{usuario_nome}_{contador_fotos}.jpg")
        cv2.imwrite(foto_path, frame)
        print(f"Imagem salva: {foto_path}")
        contador_fotos += 1

    camera.release()
    cv2.destroyAllWindows()

    if contador_fotos == num_fotos:
        print(f"{num_fotos} imagens capturadas para o usuário: {usuario_nome}")
    else:
        print(f"Captura encerrada com {contador_fotos} imagens.")


# Função para treinamento do modelo
def treinar_modelo(dataset_path, modelo_path):
    """Treina o modelo de reconhecimento facial com base no dataset."""
    print("Treinando o modelo...")
    X, y = [], []

    for root, dirs, files in os.walk(dataset_path):
        if files:
            person_name = os.path.basename(os.path.normpath(root))
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file)
                    try:
                        image = cv2.imread(image_path)
                        if image is None:
                            print(f"Falha ao carregar a imagem: {image_path}. Ignorando.")
                            continue

                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_image)
                        encodings = face_recognition.face_encodings(rgb_image, face_locations)

                        if not encodings:
                            print(f"Nenhum encoding encontrado para {image_path}. Ignorando.")
                            continue

                        for encoding in encodings:
                            X.append(encoding)
                            y.append(person_name)
                    except Exception as e:
                        print(f"Erro ao processar {image_path}: {e}")

    if len(set(y)) < 2:
        print("Erro: O dataset deve conter pelo menos duas pessoas para treinar o modelo.")
        return

    print(f"Encodings preparados: {len(X)}, Classes: {set(y)}")
    class_counts = {name: y.count(name) for name in set(y)}
    print(f"Distribuição de classes: {class_counts}")

    # Treinamento do modelo
    modelo = SVC(kernel='linear', probability=True)
    modelo.fit(X, y)

    # Salvar o modelo treinado
    with open(modelo_path, 'wb') as f:
        pickle.dump(modelo, f)
    
    print(f"Modelo treinado e salvo em: {modelo_path}")


# Função para reconhecimento ao vivo
def reconhecimento_ao_vivo(modelo_path):
    """Faz o reconhecimento facial ao vivo com o modelo treinado."""
    if os.path.exists(modelo_path):
        try:
            with open(modelo_path, 'rb') as f:
                modelo = pickle.load(f)
            print("Modelo carregado com sucesso!")

            # Iniciar captura ao vivo
            camera = cv2.VideoCapture(0)

            while True:
                ret, frame = camera.read()
                if not ret:
                    print("Erro ao acessar a câmera.")
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for encoding, location in zip(encodings, face_locations):
                    matches = modelo.predict([encoding])

                    if matches:
                        name = matches[0]
                        print(f"Reconhecido: {name}")

                        # Desenhar a caixa ao redor do rosto
                        top, right, bottom, left = location
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                cv2.imshow("Reconhecimento Facial Ao Vivo", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):  # Pressione 'q' para sair
                    break

            camera.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
    else:
        print(f"O modelo em {modelo_path} não foi encontrado. Certifique-se de treinar o modelo primeiro.")
