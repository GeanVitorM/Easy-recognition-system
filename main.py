#Main.py
import os
import cv2
from face_recognition_model import capture_images, train_model, live_recognition

def main():
    # Solicita o nome do usuário
    user_name = input("Digite o nome do usuário: ")

    # Criação da pasta para o usuário, caso não exista
    dataset_path = f'./dataset/{user_name}'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"Pasta do usuário '{user_name}' criada com sucesso!")

    # Captura das imagens
    print(f"Agora, adicione as imagens para o usuário '{user_name}' na pasta './dataset/{user_name}'.")
    capture_images(user_name, dataset_path)

    # Treinamento do modelo com as imagens capturadas
    print("Treinando o modelo...")
    model = train_model(dataset_path)
    print("Modelo treinado com sucesso!")

    # Reconhecimento em tempo real
    print("Iniciando reconhecimento ao vivo...")
    live_recognition(model)

if __name__ == "__main__":
    main()
