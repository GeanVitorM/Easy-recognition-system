# main.py
import os
import pickle  # Adicionando a importação do pickle
from face_recognition_model import capture_images, train_model, save_model, live_recognition

def main():
    """Controla o fluxo principal do programa."""
    print("Escolha a funcionalidade desejada:")
    print("1 - Treinamento do modelo (capturar imagens e treinar)")
    print("2 - Reconhecimento ao vivo")
    
    choice = input("Digite o número da funcionalidade desejada (1 ou 2): ")

    if choice == "1":
        # Função para treinamento do modelo
        print("Iniciando treinamento do modelo...")
        user_name = input("Digite o nome do usuário: ")
        dataset_path = f'./dataset/{user_name}'

        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            print(f"Pasta do usuário '{user_name}' criada com sucesso!")

        print(f"Agora, adicione as imagens para o usuário '{user_name}' na pasta './dataset/{user_name}'.")
        capture_images(user_name, dataset_path)

        print("Treinando o modelo...")
        model = train_model(dataset_path)

        if model:
            model_path = './Models/trained_model.pkl'
            save_model(model, model_path)  # Função que salva o modelo
            print(f"Modelo treinado e salvo em {model_path} com sucesso!")
        else:
            print("Falha ao treinar o modelo!")

    elif choice == "2":
        # Função para reconhecimento ao vivo
        print("Iniciando reconhecimento ao vivo...")
        model_path = './Models/trained_model.pkl'

        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                print("Modelo carregado com sucesso!")
                live_recognition(model)
            except Exception as e:
                print(f"Erro ao carregar o modelo: {e}")
        else:
            print(f"O modelo em {model_path} não foi encontrado. Certifique-se de treinar o modelo primeiro.")

    else:
        print("Opção inválida. Por favor, escolha 1 ou 2.")

if __name__ == "__main__":
    main()
