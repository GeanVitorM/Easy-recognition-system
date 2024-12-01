
# Sistema de Reconhecimento Facial

Uma breve descrição sobre o que esse projeto faz e para quem ele é

Este projeto utiliza a biblioteca **face_recognition** para reconhecimento facial em tempo real, juntamente com um classificador KNN (K-Nearest Neighbors) para identificar usuários com base em imagens capturadas da webcam. O sistema captura imagens da câmera, treina um modelo KNN com essas imagens e realiza o reconhecimento facial em tempo real.

## Funcionalidades

- **Captura de Imagens**: O sistema captura imagens da webcam, detecta rostos e as salva para treinamento.
- **Preparação do Dataset**: As imagens salvas são usadas para gerar os "encodings" faciais, que são características únicas de cada rosto.
- **Treinamento do Modelo**: Utiliza um classificador KNN para treinar um modelo com os encodings das faces.
- **Reconhecimento em Tempo Real**: A partir do modelo treinado, o sistema consegue identificar rostos capturados pela webcam em tempo real.

## Requisitos

- Python 3.x
- OpenCV
- face_recognition
- scikit-learn
- numpy
- pickle

### Instalação das dependências

Para instalar as dependências, execute o seguinte comando:

```bash
pip install opencv-python face_recognition scikit-learn numpy
```

### Funcionalidades:
- **Captura de Imagens**: Descreve como as imagens são capturadas da webcam para treinamento.
- **Preparação do Dataset**: Explica como as imagens capturadas são processadas para extrair características únicas (encodings) de cada rosto.
- **Treinamento do Modelo**: Descreve o uso do classificador KNN para treinar um modelo com os encodings extraídos das imagens.
- **Reconhecimento em Tempo Real**: Explica como o sistema faz a identificação dos rostos em tempo real com base no modelo treinado.
## Utilizando o projeto
* Iniciando o projeto

```bash
 py main.py
```

* Selecionando funcionalidade
```bash
1 - Treinamento do modelo (capturar imagens e treinar)
2 - Reconhecimento ao vivo
Digite o número da funcionalidade desejada (1 ou 2):
```

* Treinamento do modelo - 1:
```bash
Digite o número da funcionalidade desejada (1 ou 2): 1
Iniciando treinamento do modelo...
Digite o nome do usuário: User
Agora, adicione as imagens para o usuário 'User' na pasta './dataset/User'.
Capturando imagens para o usuário 'Gean'. Olhe para a câmera e pressione 'q' para sair.
Imagem salva: C:\Users\User\Documents\Easy recognition system\dataset\User\User\User.jpg
Imagem salva: C:\Users\User\Documents\Easy recognition system\dataset\User\User\User.jpg
Imagem salva: C:\Users\User\Documents\Easy recognition system\dataset\User\User\User.jpg
Imagem salva: C:\Users\User\Documents\Easy recognition system\dataset\User\User\User.jpg
Imagem salva: C:\Users\User\Documents\Easy recognition system\dataset\User\User\User.jpg
Imagem salva: C:\Users\User\Documents\Easy recognition system\dataset\User\User\User.jpg
Treinando o modelo...
Modelo salvo em ./Models/trained_model.pkl com sucesso!
Modelo treinado e salvo em ./Models/trained_model.pkl com sucesso!
```
* Reconhecimento ao vivo - 2:
```bash
Digite o número da funcionalidade desejada (1 ou 2): 2
Iniciando reconhecimento ao vivo...
Modelo carregado com sucesso!
Reconhecimento ao vivo iniciado. Pressione 'q' para encerrar.
```

* Em ambas as escolhas a camera sera aberta para reconhecimento.

## Requisitos

- Python 3.x
- OpenCV
- face_recognition
- scikit-learn
- numpy
- pickle
