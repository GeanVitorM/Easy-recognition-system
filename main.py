import os
from face_recognition_model import cadastrar_usuario, treinar_modelo, reconhecimento_ao_vivo

def main():
    """Controla o fluxo principal do programa."""
    while True:
        print("\nEscolha a funcionalidade desejada:")
        print("1 - Cadastrar novo usuário (capturar fotos)")
        print("2 - Treinar modelo")
        print("3 - Reconhecimento facial ao vivo")
        print("0 - Encerrar")
        
        choice = input("Digite o número da funcionalidade desejada (1, 2, 3 ou 0): ").strip()

        dataset_path = "./dataset"  # Diretório do dataset
        modelo_path = "./Models/modelo.pkl"  # Caminho para salvar o modelo treinado

        if choice == "1":
            nome_usuario = input("Digite o nome do usuário: ").strip()
            cadastrar_usuario(nome_usuario, dataset_path)
        
        elif choice == "2":
            treinar_modelo(dataset_path, modelo_path)
        
        elif choice == "3":
            reconhecimento_ao_vivo(modelo_path)

        elif choice == "0":
            print("Encerrando o programa...")
            break  # Encerra o loop e o programa

        else:
            print("Opção inválida! Por favor, escolha 1, 2, 3 ou 0.")

if __name__ == "__main__":
    main()
