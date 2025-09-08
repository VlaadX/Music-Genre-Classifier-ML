# **Classificação de Gêneros Musicais com Machine Learning**

![Capa do Projeto](https://github.com/VlaadX/Music-Genre-Classifier-ML/blob/main/imgs/dataset-cover.jpg)

A classificação de gêneros musicais é uma tarefa fundamental para a organização e curadoria de grandes bibliotecas de áudio. É a base para sistemas de recomendação de músicas e para a análise de tendências na indústria musical. O desafio é ensinar uma máquina a "ouvir" e a identificar padrões musicais complexos que definem gêneros como rock, jazz ou hip hop.

Este projeto foca em construir um pipeline completo de Machine Learning, do processamento de áudio à avaliação do modelo, para classificar áudios em diferentes gêneros musicais.

---
## **1.0 O Problema do Negócio: O Desafio da Classificação de Áudio**

No mundo da música digital, a capacidade de categorizar automaticamente uma faixa de áudio em seu gênero correto é um recurso de alto valor. Para serviços de streaming como o Spotify ou o YouTube Music, isso é crucial para construir playlists personalizadas e para que os usuários descubram novas músicas.

O maior desafio aqui é transformar a informação bruta do áudio (uma sequência de ondas sonoras) em um formato que um modelo de Machine Learning possa entender. O áudio é um tipo de dado sequencial e complexo, e métricas como a acurácia pura podem ser enganosas se o modelo não conseguir capturar as nuances acústicas de cada gênero. Por isso, a validação de modelos deve focar em:
- **Precision:** De todos os áudios que o modelo previu para um gênero, quantos estavam corretos?
- **Recall:** De todos os áudios de um gênero, quantos o modelo conseguiu identificar corretamente?
- **F1-Score:** Uma métrica que equilibra Precision e Recall, ideal para avaliar o desempenho em cada gênero.

---
## **2.0 Análise e Pré-processamento dos Dados de Áudio**

A etapa mais importante deste projeto foi o pré-processamento, que envolveu a extração de características significativas dos arquivos de áudio.

### **2.1 A Extração das Features: MFCCs**

Para converter o áudio em dados numéricos, optamos por extrair os **Coeficientes de Frequência Mel-Cepstral (MFCCs)**. Os MFCCs são amplamente utilizados em processamento de áudio pois simulam a percepção do som pelo ouvido humano. Eles representam o timbre do áudio, ignorando as informações de pitch e volume.

Cada arquivo de áudio foi processado para gerar uma série de MFCCs, que foram então transformados em um conjunto de dados tabulares.

### **2.2 O Processo de Pré-processamento**

O conjunto de dados foi dividido em conjuntos de treino e teste de forma **estratificada**. Isso é fundamental para a classificação de gêneros, pois garante que a proporção de cada gênero musical seja a mesma nos conjuntos de treino e teste, evitando que o modelo seja treinado com um desbalanceamento de classes que não existe na realidade.

---
## **3.0 Comparação e Avaliação dos Modelos**

A alma do projeto reside na comparação entre modelos. Nosso objetivo não era apenas encontrar o modelo com o melhor desempenho técnico, mas sim aquele que oferece a solução mais pragmática e benéfica para a classificação de áudio. Para este projeto, o modelo **XGBoost** foi o escolhido para a avaliação.

### **3.1 Análise Detalhada do Modelo XGBoost**

O modelo **XGBoost**, um algoritmo de Gradient Boosting, se mostrou robusto e eficaz na classificação de gêneros. A sua performance no conjunto de teste foi a seguinte:

| **Gênero** | **Precision** | **Recall** | **F1-Score** | **Suporte** |
| :--- | :--- | :--- | :--- | :--- |
| **blues** | 0.77 | 0.65 | 0.70 | 20 |
| **classical** | 0.81 | 0.85 | 0.83 | 20 |
| **country** | 0.67 | 0.80 | 0.73 | 20 |
| **disco** | 0.83 | 0.90 | 0.86 | 20 |
| **hiphop** | 0.67 | 0.50 | 0.57 | 20 |
| **jazz** | 0.73 | 0.75 | 0.74 | 20 |
| **metal** | 0.88 | 0.90 | 0.89 | 20 |
| **pop** | 0.67 | 0.60 | 0.63 | 20 |
| **reggae** | 0.85 | 0.85 | 0.85 | 20 |
| **rock** | 0.62 | 0.65 | 0.63 | 20 |
| **Acurácia (Geral)** | | | | |
| | | | **0.75** | **200** |

#### **Insights da Matriz de Confusão**

A matriz de confusão forneceu insights valiosos sobre o desempenho do modelo em cada gênero:

- **Performance Sólida:** Gêneros como `metal`, `disco` e `classical` foram classificados com alta precisão e recall, mostrando que o modelo é excelente em identificar suas características acústicas distintas.
- **Maior Desafio:** O modelo teve mais dificuldade em distinguir entre gêneros como `hiphop` e `rock`. Por exemplo, alguns áudios de `hiphop` foram classificados incorretamente como `rock`, e vice-versa. Isso mostra que, acusticamente, as características de ambos os gêneros podem ter sobreposição, e uma análise mais profunda de hiperparâmetros ou a utilização de outros modelos poderia melhorar esses resultados.
  ![Página Principal do Dashboard](https://github.com/VlaadX/Music-Genre-Classifier-ML/blob/main/imgs/output.png)
### **3.2 Conclusão do Modelo**

O modelo **XGBoost** se mostrou uma escolha robusta e eficaz para a classificação de gêneros musicais, alcançando uma **acurácia geral de 75%** e um bom desempenho em gêneros específicos. Embora um projeto completo possa incluir a otimização de hiperparâmetros e a comparação com outros modelos, como Redes Neurais, o XGBoost demonstrou um resultado inicial muito promissor, validando o pipeline de pré-processamento e extração de características.

---
## **4.0 Estrutura do Projeto e Tecnologias**

O projeto foi estruturado em um pipeline de 2 etapas claras, documentadas em notebooks Jupyter para garantir a reprodutibilidade.

* **Tecnologias:** Python 3.10+, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost.
* **Estrutura:**
    * `notebooks/` - Contém os notebooks para cada etapa (Extração de Features e Treinamento do Modelo).
    * `data/` - Armazena o dataset original e os arquivos processados (`X_train.csv`, etc.). **A pasta `data` não é versionada no Git.**

---
## **5.0 Como Executar o Projeto**

1.  **Instale as dependências:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost
    ```
2.  **Baixe o Dataset:**
    - Baixe o conjunto de dados GTZAN em [http://marsyas.info/downloads/datasets.html](http://marsyas.info/downloads/datasets.html).
    - Descompacte o arquivo e mova a pasta `genres` para dentro da sua pasta `data`, renomeando-a para `genres_original`.
3.  **Execute os notebooks na ordem correta:**
    - `01_feature_extraction.ipynb`
    - `02_music_classifier.ipynb`
