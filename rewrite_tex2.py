import re

with open('docs/relatorio/relatorio.tex', 'r', encoding='utf-8') as f:
    content = f.read()

replacement = r"""\part*{Parte II --- Tarefa 2: Dataset Atribuído (Consumo Energético)}
\addcontentsline{toc}{section}{Parte II --- Tarefa 2: Dataset Atribuído (Consumo Energético)}

\section{Metodologia}

A metodologia de análise de dados que será utilizada neste problema é a SEMMA: Sample, Explore, Modify, Model e Assess. A fase de Sample encontra-se já concretizada na forma do dataset atribuído, pelo que o presente relatório se foca nas fases seguintes. Assim, começou-se por realizar a fase de Explore ou Exploração, cujo objetivo é perceber potenciais tendências e problemas com os dados, para na fase de Modificação se fazerem transformações que visam corrigir os problemas identificados na fase de Exploração. Por fim, aplicam-se as fases de Modelação e Avaliação para cada um dos 5 modelos desenvolvidos para este problema.

\subsection{Visão geral do workflow desenvolvido no KNIME}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.9\textwidth]{Visão geral do workflow desenvolvido no KNIME.png}
    \caption{Visão geral do workflow desenvolvido no KNIME}
    \label{fig:workflow_geral_atribuio}
\end{figure}

De forma geral, o workflow está organizado num conjunto de metanodes que encapsulam os diferentes passos. O nó String to Number, que converte features numéricas que aparecem no dataset em formato string, encontra-se fora de qualquer metanode de forma a propagar os missing values nessas features para a Exploração e para a Modificação simultaneamente. No final, cada um dos 5 modelos recebe o output das transformações dos dados, sendo a Avaliação feita dentro de cada metanode.

\subsection{Exploração}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.9\textwidth]{Exploração.png}
    \caption{Exploração de dados}
    \label{fig:exploracao_atribuio}
\end{figure}

O primeiro passo da exploração dos dados foi a utilização do nó Statistics para verificar diversas estatísticas, destacando-se as seguintes observações:
\begin{itemize}
    \item \code{precipitation} e \code{rain} apresentam muitos missing values ($\sim 90$\%)
    \item \code{snowfall} tem todos os valores a 0 (mean = min = max = 0)
    \item \code{RowID} é redundante
    \item Todas as restantes features apresentam uma quantidade semelhante de missing values, cerca de 10\%. Por isso, deixa de ser viável remover as entradas com missing values, uma vez que, feito isso, o dataset final fica com 1300 entradas (apenas 15\% do total). Portanto, foram avaliadas, para cada modelo, diferentes abordagens para imputar os missing values (média e mediana para os numéricos e moda ou valor fixo “Missing” para as strings).
\end{itemize}

De seguida, foi verificada a presença de outliers, recorrendo tanto a Numeric Outliers para obter resultados numéricos como ao \textit{Box Plot} para fazer uma exploração mais visual. Desta análise, retirou-se que várias features (como as do vento e as da voltagem) apresentam outliers em quantidade reduzida ($< 2.5$\%) e a \code{surface\_pressure} ligeiramente mais ($\sim 5$\%). No total, são 1700 as entradas com algum outlier, pelo que a sua remoção não foi considerada, pois perder-se-ia uma boa parte do dataset (cerca de 20\%). Portanto, na avaliação dos modelos, os outliers foram ou mantidos ou ajustados para o valor permitido mais próximo, de maneira a avaliar qual o melhor tratamento para cada modelo.

Por fim, foi ainda feita uma análise dos valores das features categóricas com Value Counter, para determinar se existem valores fora do padrão. Concluiu-se que as features \code{Diffuse\_Radiation} e \code{Direct\_Radiation} não apresentam valores anormais. Já a target, \code{Consumptions}, apresenta, em algumas entradas, valores com erros ortográficos (e.g., Hgh e Lw) ou escritos em português em vez de inglês (e.g., MedioAlto vs Medium-High Consumption).

\subsection{Transformação}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.9\textwidth]{Transformação.png}
    \caption{Transformação de dados}
    \label{fig:transformacao_atribuio}
\end{figure}

As primeiras transformações aplicadas aos dados centraram-se nos formatos das features, incluindo a conversão das colunas numéricas que se encontravam em string (\code{Medium Voltage} e \code{Total}), que foi realizada fora deste metanode, conforme descrito anteriormente, e a passagem das datas de string para um formato adequado, além da consequente extração dos campos mais relevantes para este problema (hora, dia da semana e mês). A feature original com a data completa foi removida para evitar introduzir ruído. Ainda, foram detectadas 4 linhas nas quais a data era inválida (e.g., 2025-11-32), pelo que essas entradas foram removidas, por ser uma quantidade muito reduzida.

De seguida, no seguimento da exploração realizada, foram removidas algumas features, sendo elas:
\begin{itemize}
    \item \code{RowID} (redundante)
    \item \code{precipitation} e \code{rain} ($\sim 90$\% missing values)
    \item \code{snowfall} (redundante, mean = min = max = 0)
\end{itemize}

Por fim, de acordo com o que foi possível perceber na exploração, foi aplicada uma Rule Engine para uniformizar o formato dos valores de \code{Consumptions}, resultando daqui 5 valores possíveis (Low, MediumLow, Medium, MediumHigh, High).

\section{Modelação e Avaliação}

É agora altura de desenvolver modelos para o presente problema. Foram concebidos 5 modelos de classificação, com 3 baseados em árvores e 2 redes neuronais. Algumas observações relevantes para as secções seguintes:
\begin{enumerate}
    \item A partição dos dados para treino e teste foi feita, sempre que possível, com recurso aos nós X-Partitioner e X-Aggregator, uma vez que o uso de cross-validation, embora mais custoso a nível computacional, permite aproveitar a totalidade do dataset e reduzir as chances do modelo sofrer de overfitting;
    \item Os nodos de cross-validation foram configurados com uma random seed, de modo a ser possível obter resultados reproduzíveis e sem variabilidade aleatória;
    \item Não foram testadas todas as combinações possíveis de transformações de features e configurações dos modelos, pois é uma quantidade impraticável. Por isso, a abordagem adotada foi: testar um conjunto de abordagens e as suas combinações; tirar daí o melhor resultado; realizar o próximo conjunto de testes com a combinação obtida anteriormente. Isto pode fazer com que não sejam detectadas as melhores combinações para cada modelo, mas torna este processo de refinamento mais exequível.
\end{enumerate}

\subsection{Decision Tree}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.9\textwidth]{Decision Tree.png}
    \caption{Resultados da Decision Tree}
    \label{fig:decision_tree_atribuio}
\end{figure}

Tal como mencionado na fase de exploração, os tratamentos de missing values testados foram média e mediana para os valores numéricos e, para os categóricos, moda e valor fixo “Missing”. Já os outliers, tal como dito anteriormente, foram mantidos ou ajustados para o valor permitido mais próximo.

\begin{table}[ht]
\centering
\caption{Avaliação de Outliers e Missing Values na Decision Tree}
\begin{tabular}{llccc}
\hline
\textbf{Outliers} & \textbf{Miss. (num.)} & \textbf{Miss. (cat.)} & \textbf{Accuracy} & \textbf{Cohen’s Kappa} \\
\hline
manter & média & moda & 0.884 & 0.852 \\
manter & média & fixo & 0.887 & 0.855 \\
manter & mediana & moda & 0.885 & 0.853 \\
manter & mediana & fixo & 0.888 & 0.857 \\
ajustar & média & moda & 0.885 & 0.852 \\
ajustar & média & fixo & 0.890 & 0.859 \\
ajustar & mediana & moda & 0.885 & 0.853 \\
ajustar & mediana & fixo & 0.889 & 0.858 \\
\hline
\end{tabular}
\end{table}

\begin{table}[ht]
\centering
\caption{Avaliação dos Parâmetros da Decision Tree}
\begin{tabular}{llccc}
\hline
\textbf{Quality Measure} & \textbf{Pruning Method} & \textbf{Min records per node} & \textbf{Accuracy} & \textbf{Cohen’s Kappa} \\
\hline
Gini index & No pruning & 2 & 0.890 & 0.859 \\
Gini index & No pruning & 4 & 0.897 & 0.869 \\
Gini index & No pruning & 8 & 0.919 & 0.897 \\
Gini index & No pruning & 12 & 0.927 & 0.906 \\
Gini index & MDL & 2 & 0.925 & 0.904 \\
Gini index & MDL & 4 & 0.925 & 0.904 \\
Gini index & MDL & 8 & 0.925 & 0.904 \\
Gini index & MDL & 12 & 0.926 & 0.905 \\
Gain ratio & No pruning & 2 & 0.888 & 0.857 \\
Gain ratio & No pruning & 4 & 0.904 & 0.878 \\
Gain ratio & No pruning & 8 & 0.918 & 0.895 \\
Gain ratio & No pruning & 12 & 0.922 & 0.901 \\
Gain ratio & MDL & 2 & 0.919 & 0.897 \\
Gain ratio & MDL & 4 & 0.919 & 0.897 \\
Gain ratio & MDL & 8 & 0.921 & 0.899 \\
Gain ratio & MDL & 12 & 0.921 & 0.899 \\
\hline
\end{tabular}
\end{table}

Os resultados obtidos mostram que o tratamento de missing values e outliers teve impacto reduzido no desempenho da Decision Tree, verificando-se diferenças muito pequenas entre as diferentes estratégias avaliadas (menos de 1\%). Em contrapartida, os hiperparâmetros estruturais da árvore tiveram mais influência no desempenho final. O melhor desempenho global foi obtido com Gini Index, sem pruning e com mínimo de 12 registos por nó, atingindo uma accuracy de 0.927 e um Cohen’s Kappa de 0.906.

\subsection{Random Forest}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.9\textwidth]{Random Forest.png}
    \caption{Resultados da Random Forest}
    \label{fig:random_forest_atribuio}
\end{figure}

Aqui foram testadas as mesmas combinações de tratamento de missing values e outliers que na Decision Tree.

\begin{table}[ht]
\centering
\caption{Avaliação de Outliers e Missing Values na Random Forest}
\begin{tabular}{llccc}
\hline
\textbf{Outliers} & \textbf{Miss. (num.)} & \textbf{Miss. (cat.)} & \textbf{Accuracy} & \textbf{Cohen’s Kappa} \\
\hline
manter & média & moda & 0.929 & 0.909 \\
manter & média & fixo & 0.929 & 0.909 \\
manter & mediana & moda & 0.930 & 0.910 \\
manter & mediana & fixo & 0.929 & 0.909 \\
ajustar & média & moda & 0.929 & 0.909 \\
ajustar & média & fixo & 0.928 & 0.908 \\
ajustar & mediana & moda & 0.929 & 0.909 \\
ajustar & mediana & fixo & 0.928 & 0.908 \\
\hline
\end{tabular}
\end{table}

\begin{table}[ht]
\centering
\caption{Avaliação do Número de Modelos na Random Forest}
\begin{tabular}{ccc}
\hline
\textbf{Number of models} & \textbf{Accuracy} & \textbf{Cohen’s Kappa} \\
\hline
50 & 0.929 & 0.909 \\
100 & 0.930 & 0.910 \\
150 & 0.930 & 0.910 \\
200 & 0.929 & 0.909 \\
250 & 0.929 & 0.909 \\
\hline
\end{tabular}
\end{table}

Tal como observado na Decision Tree, as diferentes estratégias de tratamento de missing values e outliers tiveram impacto praticamente nulo no desempenho da Random Forest. Relativamente aos hiperparâmetros, limitando o número de níveis das árvores reduziu-se o desempenho. O melhor método foi o Information Gain Ratio. Finalmente, a avaliação do número de modelos mostrou que aumentar o número de árvores acima de 100 não produziu melhorias relevantes.

\subsection{Gradient Boosted Trees}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.9\textwidth]{Gradient Boosted Trees.png}
    \caption{Resultados de Gradient Boosted Trees}
    \label{fig:gbt_atribuio}
\end{figure}

Ao contrário do que foi feito até agora, não foram avaliadas as estratégias de imputação de missing values de modo tão extenso, pois o modelo (XGBoost) possui mecanismos próprios.

\begin{table}[ht]
\centering
\caption{Avaliação de Outliers e Missing Values (GBT)}
\begin{tabular}{llcc}
\hline
\textbf{Outliers} & \textbf{Missing values} & \textbf{Accuracy} & \textbf{Cohen’s Kappa} \\
\hline
manter & XGBoost & 0.928 & 0.908 \\
manter & Surrogate & 0.897 & 0.868 \\
ajustar & XGBoost & 0.928 & 0.908 \\
ajustar & Surrogate & 0.896 & 0.867 \\
\hline
\end{tabular}
\end{table}

\begin{table}[ht]
\centering
\caption{Avaliação de Hiperparâmetros em GBT}
\begin{tabular}{cccccc}
\hline
\textbf{Learning Rate} & \textbf{Num Models} & \textbf{Limit Levels} & \textbf{Row Sampling} & \textbf{Accuracy} & \textbf{Kappa} \\
\hline
0.1 & 100 & 8 & 0.8 & 0.932 & 0.913 \\
0.05 & 200 & 8 & 0.8 & 0.932 & 0.912 \\
0.02 & 400 & 8 & 0.8 & 0.932 & 0.913 \\
\hline
\end{tabular}
\end{table}

Os resultados mostram que o XGBoost tem um resultado excelente comparativamente, mas a variação de learning rate e número de modelos demonstrou que as configurações obtiveram o mesmo desempenho máximo (0.932).

\subsection{Multilayer Perceptron (MLP)}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.9\textwidth]{Multilayer Perceptron.png}
    \caption{Resultados do Multilayer Perceptron}
    \label{fig:mlp_atribuio}
\end{figure}

A normalização (nomeadamente Z-score) e o tratamento One to Many das features demonstraram superioridade, atingindo accuracies elevadas (0.886). O aumento da profundidade (várias camadas) implicou degradações de desempenho no conjunto de teste.

\begin{table}[ht]
\centering
\caption{Avaliação da MLP - Número de iterações e Neurónios}
\begin{tabular}{ccccc}
\hline
\textbf{Nº iterations} & \textbf{Nº hidden layers} & \textbf{Nº hidden neurons} & \textbf{Accuracy} & \textbf{Cohen’s Kappa} \\
\hline
100 & 1 & 10 & 0.886 & 0.854 \\
200 & 1 & 10 & 0.900 & 0.873 \\
300 & 1 & 10 & 0.903 & 0.875 \\
\hline
\end{tabular}
\end{table}


\subsection{Keras}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.9\textwidth]{Keras.png}
    \caption{Resultados em Keras}
    \label{fig:keras_atribuio}
\end{figure}

O último modelo avaliado foi mais uma rede neuronal. A abordagem One to Many para Keras revelou-se apropriada.

\begin{table}[ht]
\centering
\caption{Estratégia e Epochs em Keras}
\begin{tabular}{lccc}
\hline
\textbf{Estratégia} & \textbf{Nº epochs} & \textbf{Accuracy} & \textbf{Cohen’s Kappa} \\
\hline
One to Many & 5 & 0.779 & 0.718 \\
One to Many & 10 & 0.803 & 0.747 \\
One to Many & 20 & 0.842 & 0.798 \\
Numerar & 20 & 0.684 & 0.592 \\
\hline
\end{tabular}
\end{table}

\section{Conclusão da Parte II}

Seguindo a metodologia SEMMA, a fase analítica permitiu tratar de valores omissos e outliers. De modo geral, os resultados indicam que os modelos baseados em árvores (Random Forest e Gradient Boosted Trees) apresentaram os melhores desempenhos e robustez.

\clearpage

\end{document}
"""

start_str = r"\part*{Parte II --- Tarefa 2: Dataset Atribuído (Consumo Energético)}"
idx = content.find(start_str)

if idx != -1:
    new_content = content[:idx] + replacement
    
    # We must also change \graphicspath to include imgs/atribuio/
    new_content = new_content.replace("{imgs/atribuido/}", "{imgs/atribuio/}")
    
    with open('docs/relatorio/relatorio.tex', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Replace successful")
else:
    print("String not found")

