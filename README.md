# Pipeline Híbrida Quântico-Clássica para Agrupamento de Dados de Saúde Mental

Este repositório contém o código e a documentação para uma pipeline inovadora que combina técnicas de aprendizado de máquina quântica e clássica para agrupamento (clustering) de dados. Nosso objetivo é explorar como a extração de características quânticas pode aprimorar a performance de algoritmos de agrupamento tradicionais.

## Resumo do Projeto

### O que fizemos antes deste código:

Antes de chegarmos ao cerne da pipeline híbrida, realizamos as etapas essenciais de **pré-processamento e exploração dos dados**. Isso incluiu:

1.  **Carregamento dos Dados:** Leitura dos datasets de saúde mental, que provavelmente contêm informações diversas sobre indivíduos e seus estados de saúde mental.
2.  **Limpeza de Dados:** Tratamento de valores ausentes, remoção de duplicatas e correção de inconsistências.
3.  **Codificação de Variáveis Categóricas:** Transformação de variáveis textuais ou categóricas (como 'Gênero', 'Idade', etc.) em formatos numéricos que podem ser processados por algoritmos de ML (e.g., One-Hot Encoding, Label Encoding).
4.  **Normalização/Escalonamento:** Ajuste das escalas das características numéricas para evitar que características com grandes magnitudes dominem o processo de treinamento (e.g., `StandardScaler`).
5.  **Análise Exploratória de Dados (EDA):** Visualizações e estatísticas descritivas para entender a distribuição dos dados, identificar outliers e obter insights iniciais.

Após essas etapas, obtivemos o `drp` (dataset bruto processado), e em alguns casos, criamos o `dcp` (dataset com dimensionalidade reduzida e processado) através de técnicas como PCA prévias ao trecho atual do código.

### A Pipeline Híbrida em Detalhes:

Nós implementamos uma pipeline híbrida que combina o poder da computação quântica para extração de características com algoritmos clássicos de agrupamento. Pense nisso como um processo em várias etapas, onde os dados são primeiro transformados por um "cérebro quântico" e depois agrupados por "cérebros clássicos".

#### A Jornada dos Seus Dados Conosco: Do Bruto ao Agrupado 📊

Nossa pipeline começa com a preparação dos dados, seguida pela transformação quântica para extração de características. Em seguida, essas características aprimoradas são alimentadas em algoritmos de agrupamento clássicos, e os resultados são meticulosamente avaliados. **É crucial entender que nós testaremos esta pipeline com duas versões do seu conjunto de dados: o `drp` (dataset bruto) e o `dcp` (dataset processado e reduzido), para comparar o desempenho.**

#### 1. Carregamento e Pré-processamento dos Dados 🧹
*Abstração:* Antes de tudo, precisamos ter certeza de que os dados estão prontos para serem trabalhados. É como verificar se todos os ingredientes estão na cozinha antes de começar a cozinhar. 🍳

*Termos Técnicos:* Nós tentamos carregar a matriz numérica `X` a partir de um objeto `drp`. Utilizamos um **bloco `try-except`** para **tratamento de erros**, garantindo que o `drp` (provavelmente um **DataFrame Pandas** ou uma **matriz NumPy**) esteja definido. Se `drp` não for encontrado, o script é encerrado, evitando **erros de execução**. **Para esta execução, considere que `drp` representa seu conjunto de dados bruto ou completo. O mesmo processo será repetido para o `dcp`, que é o seu conjunto de dados já processado e com dimensionalidade reduzida.**

#### 2. Redução de Dimensionalidade com PCA 📉
*Abstração:* Às vezes, temos dados com muitas informações que podem ser redundantes. Aqui, estamos fazendo uma "compressão inteligente" dos dados, focando apenas no que é mais importante para o nosso "cérebro quântico" processar. 🧠💨

*Termos Técnicos:* Nós aplicamos a **Análise de Componentes Principais (PCA)** (`pca = PCA(n_components=n_qubits_actual)`) para reduzir a dimensionalidade dos dados de entrada (`X`). O número de componentes (`n_qubits_actual`) é determinado pelo número de **qubits** (`n_qubits`) configurado para o **circuito quântico variacional (VQC)**, limitando-o ao número original de características se `n_qubits` for maior. Essa etapa é crucial para mapear eficientemente as características clássicas para o espaço quântico. **Quando nós rodarmos o código com `drp`, o PCA será aplicado diretamente. Quando for executado com `dcp`, que já é um dataset reduzido e processado, o PCA ainda será aplicado para garantir que o número de características corresponda ao número de qubits definido, caso `n_qubits` seja menor que as características de `dcp`.**

#### 3. Divisão dos Dados em Treino e Teste 📚
*Abstração:* Para ter certeza de que o nosso "cérebro quântico" aprende bem, nós separamos uma parte dos dados para ele treinar e outra para ver se ele realmente aprendeu e generaliza. É como estudar para uma prova e depois fazer a prova. 📝

*Termos Técnicos:* Nós dividimos os dados reduzidos (`X_reduced`) em conjuntos de **treinamento** (`X_train`) e **teste** (`X_test`) usando `train_test_split` com uma proporção de 80/20. Embora o treinamento do VQC seja não supervisionado, essa divisão pode ser útil para avaliação posterior ou para cenários mais complexos onde o VQC poderia ser usado como um pré-processador para tarefas supervisionadas.

#### 4. O Coração Quântico: Circuito Quântico Variacional (VQC) ⚛️
*Abstração:* Esta é a parte "mágica" da nossa pipeline! Os dados comprimidos são enviados para um "computador quântico" simulado. Ele tem algumas "alavancas" (pesos) que são ajustadas para que os dados transformados sejam o mais "variados" possível, o que é bom para o agrupamento. 🪄

*Termos Técnicos:*
    * **Dispositivo Quântico:** Nós instanciamos um **dispositivo quântico simulado** (`qml.device("default.qubit", wires=n_qubits_actual)`).
    * **Circuito Quântico (`circuit`):** O `circuit` é o VQC em si. Ele utiliza **codificação de ângulos** (`qml.AngleEmbedding`) para mapear as características clássicas (os dados de entrada `x`) nos qubits, transformando-as em estados quânticos. Em seguida, camadas de **emaranhamento forte** (`qml.StronglyEntanglingLayers`) são aplicadas, contendo os **parâmetros variacionais** (os `weights` a serem otimizados). A saída é a **expectância de PauliZ no primeiro qubit** (`qml.expval(qml.PauliZ(0))`), que serve como a característica extraída.
    * **Nó Quântico (`qnode`):** O `qnode` encapsula o circuito quântico e o dispositivo, tornando-o executável e diferenciável.
    * **Função de Custo Não Supervisionada (`unsupervised_loss`):** O objetivo do treinamento do VQC é maximizar a variância das saídas do circuito para um **batch** de dados (`X_batch`). Isso é feito minimizando o negativo da variância (`-np.var(batch_preds)`). A intuição por trás disso é que, ao maximizar a variância, o VQC está efetivamente "espalhando" os dados no espaço de características, o que pode tornar os clusters mais discerníveis.
    * **Otimização:** Nós utilizamos um **otimizador de descida de gradiente** (`qml.GradientDescentOptimizer`) para ajustar iterativamente os `weights` do VQC, buscando minimizar a função de custo (ou seja, maximizar a variância das saídas). O processo ocorre em **batches** e por um número definido de **épocas** (`n_epochs`).

#### 5. Extração de Características VQC 🌟
*Abstração:* Depois que o nosso "cérebro quântico" é treinado e suas "alavancas" estão na posição ideal, nós passamos todos os seus dados por ele para obter uma nova e aprimorada versão das suas informações. ✨

*Termos Técnicos:* Após o treinamento, a função `get_vqc_features` é utilizada para aplicar o VQC otimizado a todo o conjunto de dados reduzido (`X_reduced`), gerando as **características extraídas pelo VQC** (`vqc_features`). Essas características são as representações dos dados em um espaço de características transformado quânticamente.

#### 6. Agrupamento Clássico (K-Means e Aglomerativo) 🤝
*Abstração:* Agora que nós temos as informações aprimoradas pelo computador quântico, nós usamos algoritmos clássicos para encontrar padrões e formar grupos. É como organizar um monte de objetos em caixas baseadas em suas semelhanças, mas com a ajuda do nosso "olhar quântico" que os deixou mais fáceis de ver. 📦

*Termos Técnicos:* Nós então submetemos as `vqc_features` (que são unidimensionais após a extração, daí o `reshape(-1, 1)`) a dois algoritmos de **agrupamento não supervisionado** clássicos:
    * **K-Means:** (`KMeans(n_clusters=n_clusters, n_init=10, random_state=74).fit(vqc_features)`) tenta particionar os dados em `n_clusters` pré-definidos, atribuindo cada ponto ao cluster com o centroide mais próximo.
    * **Agrupamento Aglomerativo (AgglomerativeClustering):** (`AgglomerativeClustering(n_clusters=n_clusters).fit(vqc_features)`) é um algoritmo de agrupamento hierárquico que constrói uma hierarquia de clusters, começando com cada ponto de dados como um cluster individual e, em seguida, mesclando pares de clusters.
Ambos são executados para diferentes opções de `n_clusters`.

#### 7. Avaliação e Coleta de Resultados 🏆
*Abstração:* Depois de tudo isso, precisamos ver o quão bem os agrupamentos foram feitos. Nós usamos algumas "réguas" para medir a qualidade dos grupos. Todos os detalhes de como o processo foi feito e o quão bem ele funcionou são anotados. 📝📊

*Termos Técnicos:* Para cada configuração de VQC e algoritmo de agrupamento, métricas de avaliação de cluster como o **Silhouette Score** (`silhouette_score`) e o **Davies-Bouldin Score** (`davies_bouldin_score`) são calculadas.
    * **Silhouette Score:** Mede o quão similar um objeto é ao seu próprio cluster (coesão) em comparação com outros clusters (separação). Valores próximos de 1 indicam clusters bem separados, 0 indica sobreposição, e valores negativos indicam que os objetos podem ter sido atribuídos ao cluster errado.
    * **Davies-Bouldin Score:** Mede a razão entre a dispersão intra-cluster e a separação inter-cluster. Um valor menor indica um melhor agrupamento.
Os tempos de execução para o treinamento do VQC e para os algoritmos de agrupamento clássicos também são registrados. Todos esses resultados são armazenados em uma `results_list` e, finalmente, formatados e exibidos em uma **tabela de desempenho** (`PrettyTable`) para facilitar a análise.

---

Nós acreditamos que esta pipeline demonstra uma abordagem promissora para alavancar os recursos da computação quântica para tarefas de aprendizado de máquina, particularmente em cenários onde a extração de características é um desafio para métodos clássicos. Nosso objetivo é ver se as características geradas pelo VQC podem levar a agrupamentos mais coerentes e bem separados, **comparando a performance tanto com o conjunto de dados bruto (`drp`) quanto com o conjunto de dados processado e reduzido (`dcp`)**.

### O que faremos depois deste código:

Após a execução desta pipeline e a coleta dos resultados, as próximas etapas incluirão:

1.  **Análise Comparativa:** Comparar o desempenho dos algoritmos de agrupamento nas características geradas pelo VQC (tanto para `drp` quanto para `dcp`) com o desempenho de algoritmos de agrupamento aplicados diretamente aos dados clássicos (com ou sem PCA clássico).
2.  **Otimização de Hiperparâmetros:** Refinar os parâmetros do VQC (número de qubits, camadas, taxa de aprendizado, etc.) e dos algoritmos de agrupamento (número de clusters) com base nos resultados obtidos.
3.  **Visualização dos Clusters:** Criar visualizações (e.g., gráficos de dispersão 2D/3D, t-SNE, UMAP) das características quânticas e dos clusters resultantes para uma compreensão mais intuitiva dos agrupamentos.
4.  **Interpretação dos Clusters:** Tentar atribuir significado aos clusters formados, examinando as características originais dos pontos de dados em cada grupo.
5.  **Relatório Final:** Documentar todas as descobertas, incluindo métricas de desempenho, tempos de execução e insights sobre a eficácia da abordagem híbrida quântico-clássica para este problema.