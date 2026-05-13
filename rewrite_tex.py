import re

with open('docs/relatorio/relatorio.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# I will replace from \part*{Parte II to the end of the document
replacement = r"""\part*{Parte II --- Tarefa 2: Dataset Atribuído (Consumo Energético)}
\addcontentsline{toc}{section}{Parte II --- Tarefa 2: Dataset Atribuído (Consumo Energético)}

\section{Metodologia}

A metodologia de análise de dados que será utilizada neste problema é a SEMMA: Sample, Explore, Modify, Model e Assess. A fase de Sample encontra-se já concretizada na forma do dataset atribuído, pelo que o presente relatório se foca nas fases seguintes. Assim, começou-se por realizar a fase de Explore ou Exploração, cujo objetivo é perceber potenciais tendências e problemas com os dados, para na fase de Modificação se fazerem transformações que visam corrigir os problemas identificados na fase de Exploração. Por fim, aplicam-se as fases de Modelação e Avaliação para cada um dos 5 modelos desenvolvidos para este problema.

\subsection{Visão geral do workflow desenvolvido no KNIME}

De forma geral, o workflow está organizado num conjunto de metnodes que encapsulam os diferentes passos. O nó String to Number, que converte features numéricas que aparecem no dataset em formato string, encontra-se fora de qualquer metanode de forma a propagar os missing values nessas features para a Exploração e para a Modificação simultaneamente. No final, cada um dos 5 modelos recebe o output das transformações dos dados, sendo a Avaliação feita dentro de cada metanode (ver Figura~\ref{fig:workflow_geral} no Anexo).

\subsection{Exploração}

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
ajustar & média & fixo & 0.89 & 0.859 \\
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
\textbf{Quality Measure} & \textbf{Pruning Method} & \textbf{Min records / node} & \textbf{Accuracy} & \textbf{Cohen’s Kappa} \\
\hline
Gini index & No pruning & 12 & 0.927 & 0.906 \\
Gini index & MDL & 12 & 0.926 & 0.905 \\
Gain ratio & No pruning & 12 & 0.922 & 0.901 \\
\hline
\multicolumn{5}{l}{\footnotesize (Amostra dos melhores resultados para simplificação)}\\
\end{tabular}
\end{table}

Os resultados obtidos mostram que o tratamento de missing values e outliers teve impacto reduzido no desempenho da Decision Tree, verificando-se diferenças muito pequenas. Em contrapartida, os hiperparâmetros estruturais da árvore tiveram mais influência. O melhor desempenho global foi obtido com Gini Index, sem pruning e com mínimo de 12 registos por nó, atingindo uma accuracy de 0.927.

\subsection{Random Forest}

Na Random Forest testaram-se as mesmas combinações. A avaliação do número de modelos mostrou que aumentar o número de árvores acima de 100 não produziu melhorias relevantes. Atingiu-se accuracy na ordem dos 0.930 e Kappa de 0.910 (com Information Gain Ratio, limite de níveis off, e 100 modelos).

\subsection{Gradient Boosted Trees}

Foi verificado que o modelo dispõe de mecanismos próprios (XGBoost) para lidar com missing values, obtendo bons resultados (accuracy = 0.928) comparativamente ao uso de Surrogate (accuracy = 0.897).
Avaliando a learning rate e o número de modelos, os resultados apontam a uma accuracy que pode chegar a 0.932 (learning rate 0.05, 200 modelos) usando maior profundidade de árvores e row sampling.

\subsection{Multilayer Perceptron (MLP)}

A normalização (nomeadamente Z-score) e o tratamento One to Many das features demonstraram superioridade, atingindo accuracies até 0.886. 
Quanto à arquitetura, mais iterações contribuíram para melhorias, mas o aumento demasiado profundo (várias camadas) não implicou ganhos e muitas vezes degradou as performances nos testes pela complexidade acrescida na rede.

\subsection{Keras}

A adaptação final usou One to Many para o Target consumos, logrando accuracy de 0.842 com 3 hidden layers (64, 32, 16 units) após 20 epochs. Uma abordagem de numeração provou-se muito inferior inicialmente, devido aos métodos de regularização contidos.

\section{Conclusão da Parte II}

Seguindo a metodologia SEMMA, a fase analítica permitiu tratar de valores omissos e outliers. De modo geral, os resultados indicam que modelos de gradient boosting e florestas aleatórias demonstraram os melhores comportamentos e robustness perante variabilidade nos tratamentos de dados. As redes neuronais denotam forte dependência nestas vertentes e, embora o modelo baseado em Keras introduza imensa flexibilidade, os seus custos e limitações demonstraram uma vantagem marginal em relação aos métodos baseados em árvores.

\clearpage

\section*{Anexos (Comum)}
\addcontentsline{toc}{section}{Anexos}
\setcounter{figure}{0}

\begin{figure}[ht]
    \centering
    % Se existirem as imagens, pode-se incluir aqui
    % \includegraphics[width=0.9\textwidth]{image1.png}
    \caption{Visão geral do Workflow Desenvolvido (Exploração e Modelação)}
    \label{fig:workflow_geral}
\end{figure}

\end{document}
"""

start_str = r"\part*{Parte II --- Tarefa 2: Dataset Atribuído (Consumo Energético)}"
idx = content.find(start_str)

if idx != -1:
    new_content = content[:idx] + replacement
    with open('docs/relatorio/relatorio.tex', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Replace successful")
else:
    print("String not found")

