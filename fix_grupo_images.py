import re

with open('docs/relatorio/relatorio.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Make sure graphicspath includes the grupo folder.
text = re.sub(r'\\graphicspath\{\{imgs/atribuio/\}\}', r'\\graphicspath{{imgs/atribuio/}{imgs/grupo/}}', text)

# 1. Dataset understanding workflow
text = text.replace(r'\subsection{Compreensão dos dados}', 
r'''\subsection{Compreensão dos dados}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.9\textwidth]{Data Understanding (Screenshot).png}
    \caption{Workflow da fase de Compreensão dos Dados}
    \label{fig:workflow_data_und}
\end{figure}''')

# 2. Data Preparation workflow
text = text.replace(r'\subsection{Preparação dos dados}', 
r'''\subsection{Preparação dos dados}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.9\textwidth]{Data Preparation (Screenshot).png}
    \caption{Workflow da fase de Preparação dos Dados}
    \label{fig:workflow_data_prep}
\end{figure}''')

# 3. Clustering workflow
text = text.replace(r'\subsubsection{\textit{Clustering} de perfis acústicos}', 
r'''\subsubsection{\textit{Clustering} de perfis acústicos}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.9\textwidth]{Clustering (Screenshot).png}
    \caption{Workflow da fase de Clustering}
    \label{fig:workflow_clustering}
\end{figure}''')

# 4. Classification workflow
text = text.replace(r'\subsubsection{Classificação de género musical}', 
r'''\subsubsection{Classificação de género musical}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.9\textwidth]{Classification (Screenshot).png}
    \caption{Workflow da fase de Classificação}
    \label{fig:workflow_classification}
\end{figure}''')

# 5. Regression workflow
text = text.replace(r'\subsubsection{Regressão da popularidade}', 
r'''\subsubsection{Regressão da popularidade}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.9\textwidth]{Regression (Screenshot).png}
    \caption{Workflow da fase de Regressão}
    \label{fig:workflow_regression}
\end{figure}''')

# Annex refs replacements

# Elbow Plot
text = re.sub(
    r'\(Figura~\\ref\{fig:elbow\}(.*?)\)',
    r'(Figura~\\ref{fig:elbow})\n\n\\begin{figure}[ht]\n    \\centering\n    \\includegraphics[width=0.8\\textwidth]{Line Plot - Elbow_Technique.png}\n    \\caption{Soma das distâncias quadradas intra-cluster (WCSS) para diferentes valores de k}\n    \\label{fig:elbow}\n\\end{figure}',
    text
)

# Centroides Plot
text = re.sub(
    r'\(Figura~\\ref\{fig:centroides\}(.*?)\)',
    r'(Figura~\\ref{fig:centroides})\n\n\\begin{figure}[ht]\n    \\centering\n    \\includegraphics[width=0.8\\textwidth]{5clusters.png}\n    \\caption{Perfil acústico médio de cada cluster}\n    \\label{fig:centroides}\n\\end{figure}',
    text
)

# Cluster x Genre
text = re.sub(
    r'\(Figura~\\ref\{fig:genero_cluster\}(.*?)\)',
    r'(Figura~\\ref{fig:genero_cluster})\n\n\\begin{figure}[ht]\n    \\centering\n    \\includegraphics[width=0.9\\textwidth]{Bar Chart - Clusters_Genres.png}\n    \\caption{Distribuição dos géneros musicais por cluster}\n    \\label{fig:genero_cluster}\n\\end{figure}',
    text
)

# Confusion Matrix Heatmap
text = re.sub(
    r'\(Figura~\\ref\{fig:class_heatmap\}(.*?)\)',
    r'(Figura~\\ref{fig:class_heatmap})\n\n\\begin{figure}[ht]\n    \\centering\n    \\includegraphics[width=0.9\\textwidth]{Heatmap - Genres_Prediction_Random_Forest.png}\n    \\caption{Matriz de confusão para o modelo Random Forest}\n    \\label{fig:class_heatmap}\n\\end{figure}',
    text
)

# PCA Scatter Plot
text = re.sub(
    r'\(Figura~\\ref\{fig:class_pca\}(.*?)\)',
    r'(Figura~\\ref{fig:class_pca})\n\n\\begin{figure}[ht]\n    \\centering\n    \\includegraphics[width=0.8\\textwidth]{Scatter Plot - Genres_PCA_Rand_Forest.png}\n    \\caption{Projeção PCA das previsões da Random Forest}\n    \\label{fig:class_pca}\n\\end{figure}',
    text
)

# Linear Regression Plots
text = re.sub(
    r'\(Figuras~\\ref\{fig:linear_scatter\} e~\\ref\{fig:linear_res\}(.*?)\)',
    r'(Figuras~\\ref{fig:linear_scatter} e~\\ref{fig:linear_res})\n\n\\begin{figure}[ht]\n    \\centering\n    \\includegraphics[width=0.45\\textwidth]{regressionlinear.png}\n    \\hfill\n    \\includegraphics[width=0.45\\textwidth]{regressionlinearh.png}\n    \\caption{Diagnóstico do modelo linear: dispersão vs previsão e histograma de resíduos}\n    \\label{fig:linear_scatter}\n    \\label{fig:linear_res}\n\\end{figure}',
    text
)

# Tree Scatter Plots
text = re.sub(
    r'Figuras~\\ref\{fig:tree_scatter\} e~\\ref\{fig:tree_res\}([^)]*)\)',
    r'Figuras~\\ref{fig:tree_scatter} e~\\ref{fig:tree_res}\n\n\\begin{figure}[ht]\n    \\centering\n    \\includegraphics[width=0.45\\textwidth]{regressiontreeplot.png}\n    \\hfill\n    \\includegraphics[width=0.45\\textwidth]{regressiontreeh.png}\n    \\caption{Diagnóstico da árvore de regressão}\n    \\label{fig:tree_scatter}\n    \\label{fig:tree_res}\n\\end{figure}',
    text
)

# Random Forest Scatter Plots
text = re.sub(
    r'Figuras~\\ref\{fig:rf_scatter\} e~\\ref\{fig:rf_res\}(.*?)anx(.*?)\}',
    r'Figuras~\\ref{fig:rf_scatter} e~\\ref{fig:rf_res}\n\n\\begin{figure}[ht]\n    \\centering\n    \\includegraphics[width=0.45\\textwidth]{randonforestplot.png}\n    \\hfill\n    \\includegraphics[width=0.45\\textwidth]{randomforesth.png}\n    \\caption{Diagnóstico da Random Forest: dispersão vs previsão e histograma de resíduos}\n    \\label{fig:rf_scatter}\n    \\label{fig:rf_res}\n\\end{figure}',
    text
)

# Histograms and Boxplots
text = re.sub(
    r'\(Figuras~\\ref\{fig:spotify_histogramas\} e~\\ref\{fig:spotify_boxplots\}(.*?)\)',
    r'(Figuras~\\ref{fig:spotify_histogramas} e~\\ref{fig:spotify_boxplots})\n\n\\begin{figure}[ht]\n    \\centering\n    \\includegraphics[width=0.45\\textwidth]{Histogram - Popularity.png}\n    \\hfill\n    \\includegraphics[width=0.45\\textwidth]{Histogram - Danceability.png}\n    \\caption{Distribuições de Popularity e Danceability (exemplos)}\n    \\label{fig:spotify_histogramas}\n    \\label{fig:spotify_boxplots}\n\\end{figure}',
    text
)

# Correlation Matrix Heatmap
text = re.sub(
    r'\(Figura~\\ref\{fig:spotify_corr\}(.*?)\)',
    r'(Figura~\\ref{fig:spotify_corr})\n\n\\begin{figure}[ht]\n    \\centering\n    \\includegraphics[width=0.8\\textwidth]{Heatmap - Features_Correlation.png}\n    \\caption{Matriz de correlação linear sobre 15 colunas}\n    \\label{fig:spotify_corr}\n\\end{figure}',
    text
)

with open('docs/relatorio/relatorio.tex', 'w', encoding='utf-8') as f:
    f.write(text)
