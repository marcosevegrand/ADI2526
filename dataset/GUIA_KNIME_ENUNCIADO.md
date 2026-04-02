# Guia KNIME Passo a Passo para Responder ao Enunciado

## Objetivo deste guia
Este documento descreve um workflow completo em KNIME para:
- explorar, preparar e limpar dados;
- treinar e comparar modelos de classificacao e regressao;
- otimizar hiperparametros;
- apresentar analise critica de resultados;
- produzir evidencias para o relatorio final.

O guia esta orientado ao dataset Spotify do projeto, mas a estrutura e reutilizavel para o dataset atribuido no enunciado.

## Resultado esperado no fim
No fim deves ter:
- 1 workflow KNIME organizado por blocos (Data Quality, EDA, Classificacao, Regressao, Otimizacao, Relatorio);
- tabelas com metricas finais por modelo;
- graficos de EDA e comparacao de desempenho;
- conclusoes tecnicas para o relatorio.

---

## 1. Estrutura do workflow (visao geral)
Cria 7 metanodes (ou componentes), por esta ordem:
1. Ingestao e Qualidade de Dados
2. EDA
3. Preparacao para Modelacao
4. Classificacao (genre)
5. Regressao (popularity)
6. Otimizacao de Hiperparametros
7. Comparacao Final e Export

Sugestao de ligacoes:
- O bloco 1 alimenta o bloco 2 e o bloco 3.
- O bloco 3 alimenta os blocos 4 e 5.
- O bloco 6 recebe tabelas de treino do bloco 3 (separadamente para classificacao/regressao).
- O bloco 7 recebe os resultados dos blocos 4, 5 e 6.

---

## 2. Ingestao e Qualidade de Dados

### 2.1 Ler o CSV
Node: CSV Reader

Configuracao:
- Delimiter: `,`
- Quote character: `"`
- Header row: ativo
- Encoding: `UTF-8`
- Missing value patterns: vazio, `NA`, `N/A`, `null`, `None`, `?`

Validar:
- 21 colunas lidas;
- nomes corretos das colunas;
- sem deslocamento de colunas.

### 2.2 Limpar espacos em colunas texto
Node: String Manipulation (Multi Column)

Configuracao:
- Include apenas colunas String: `track_id`, `track_name`, `artist_name`, `album_name`, `genre`
- Expression: `strip($$CURRENTCOLUMN$$)`
- Result columns: Replace selected input columns

Se o teu KNIME der erro com `CURRENTCOLUMN`:
- usar 5 nodes String Manipulation (um por coluna) com:
  - `strip($track_id$)`
  - `strip($track_name$)`
  - `strip($artist_name$)`
  - `strip($album_name$)`
  - `strip($genre$)`

### 2.3 Corrigir tipos de dados
Node: Column Type Converter

Configuracao recomendada:
- String: `track_id`, `track_name`, `artist_name`, `album_name`, `genre`
- Integer: `release_year`, `popularity`, `duration_ms`, `key`, `mode`, `time_signature`
- Double: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`
- Boolean: `explicit`

Se `explicit` vier como texto:
- Node anterior: Value Mapper
- Mapeamentos: `True/true/1 -> true`, `False/false/0 -> false`

### 2.4 Regras de qualidade (linhas invalidas)
Node: Rule Engine

Criar nova coluna: `quality_flag`

Regras (ordem importante):
1. `MISSING $track_id$ => "drop"`
2. `MISSING $genre$ => "drop"`
3. `MISSING $popularity$ => "drop"`
4. `$popularity$ < 0 OR $popularity$ > 100 => "drop"`
5. `$release_year$ < 2000 OR $release_year$ > 2024 => "drop"`
6. `$duration_ms$ <= 0 => "drop"`
7. `$tempo$ <= 0 => "drop"`
8. `$key$ < 0 OR $key$ > 11 => "drop"`
9. `$mode$ != 0 AND $mode$ != 1 => "drop"`
10. `$danceability$ < 0 OR $danceability$ > 1 => "drop"`
11. `$energy$ < 0 OR $energy$ > 1 => "drop"`
12. `$speechiness$ < 0 OR $speechiness$ > 1 => "drop"`
13. `$acousticness$ < 0 OR $acousticness$ > 1 => "drop"`
14. `$instrumentalness$ < 0 OR $instrumentalness$ > 1 => "drop"`
15. `$liveness$ < 0 OR $liveness$ > 1 => "drop"`
16. `$valence$ < 0 OR $valence$ > 1 => "drop"`
17. `TRUE => "keep"`

Node seguinte: Row Filter
- Filter column: `quality_flag`
- Keep rows with value `keep`

### 2.5 Tratar missing values
Node: Missing Value Row Filter
- remover linhas com missing em alvos: `genre` e `popularity`

Node: Missing Value

Configuracao:
- Continuas: mediana (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`)
- Inteiras: moda (`release_year`, `duration_ms`, `key`, `mode`, `time_signature`)
- Boolean: moda (`explicit`)
- Texto: constante `Unknown` (`track_name`, `artist_name`, `album_name`)
- `track_id`: nao imputar (se faltar, remover)

### 2.6 Remover duplicados
Node: Duplicate Row Filter

Configuracao:
- Key column: `track_id`
- Strategy: keep first
- Output: unique rows

Opcional (seguranca):
- segundo Duplicate Row Filter por chave composta:
  - `track_name`, `artist_name`, `album_name`, `release_year`

### 2.7 Verificacao final de qualidade
Node: Statistics

Confirmar:
- missing residual relevante = 0;
- dominios validos;
- numero final de linhas.

---

## 3. EDA (Exploracao e Analise)

### 3.1 Distribuicoes
Nodes sugeridos:
- Histogram
- Box Plot

Aplicar em:
- `danceability`, `energy`, `loudness`, `valence`, `tempo`, `duration_ms`, `popularity`

### 3.2 Correlacoes
Nodes sugeridos:
- Linear Correlation
- Correlation Matrix
- Heatmap (opcional)

Objetivo:
- identificar features mais associadas a `popularity`.

### 3.3 Analise por genero
Nodes:
- GroupBy
- Bar Chart
- Box Plot

Configuracao GroupBy:
- Group column: `genre`
- Aggregations: media/mediana para `popularity`, `energy`, `danceability`, `valence`

### 3.4 Outliers
Nodes:
- Numeric Outliers (IQR)
- Box Plot (inspecao visual)

Regra pratica:
- nao remover outliers automaticamente sem justificar no relatorio.

---

## 4. Preparacao para Modelacao

### 4.1 Separar colunas que entram no modelo
Node: Column Filter

Remover como features:
- `track_id`, `track_name`, `artist_name`, `album_name`

Manter:
- features numericas e categoricas relevantes
- target conforme o ramo (classificacao/regressao)

### 4.2 Codificar categoricas
Node: One to Many

Usar quando existe feature categorica nominal.

Regra importante:
- se `genre` for target de classificacao, nao codificar `genre` como feature.

### 4.3 Escalar variaveis
Node: Normalizer ou Standardizer

Usar em ramos de modelos sensiveis a escala:
- SVM
- kNN

Nao obrigatorio para:
- Decision Tree
- Random Forest
- Gradient Boosted Trees

### 4.4 Dividir treino/teste
Node: Partitioning

Configuracao:
- proporcao: 80/20
- random seed fixa (ex: 42)
- classificacao: ativar stratified sampling por `genre`
- regressao: amostragem aleatoria simples

Opcional para maior robustez:
- X-Partitioner + X-Aggregator (cross-validation)

---

## 5. Modelacao - Classificacao de Genero

Target: `genre`

Modelos base (baseline + fortes):
1. Decision Tree Learner
2. Random Forest Learner
3. Gradient Boosted Trees Learner
4. SVM Learner (com escalonamento)
5. k-NN Learner (com escalonamento)

Pipeline por modelo:
- Learner -> Predictor -> Scorer

Node: Scorer
Métricas para recolher:
- Accuracy
- Precision
- Recall
- F1
- Matriz de confusao

Boas praticas:
- comparar treino vs teste;
- evitar escolher modelo apenas por accuracy.

---

## 6. Modelacao - Regressao de Popularidade

Target: `popularity`

Modelos recomendados:
1. Linear Regression Learner
2. Random Forest Learner (modo regressao)
3. Gradient Boosted Trees Learner (modo regressao)

Pipeline por modelo:
- Learner -> Predictor -> Numeric Scorer

Node: Numeric Scorer
Métricas:
- RMSE
- MAE
- R2

Boas praticas:
- comparar treino vs teste;
- analisar residuos (opcional com Scatter Plot: real vs previsto).

---

## 7. Otimizacao de Hiperparametros

### 7.1 Estrutura de tuning
Nodes:
- Parameter Optimization Loop Start
- Learner/Predictor/Scorer (ou Numeric Scorer)
- Parameter Optimization Loop End

### 7.2 Parametros a testar (sugestao)
Decision Tree:
- max depth: 5, 10, 15
- min records per node: 2, 5, 10

Random Forest:
- num trees: 100, 300, 500
- max depth: 10, 20, sem limite
- features por split: sqrt, log2, valor fixo

Gradient Boosted Trees:
- learning rate: 0.01, 0.05, 0.1
- num trees: 100, 300, 500
- max depth: 3, 5, 8

SVM:
- C: 0.1, 1, 10
- kernel: RBF
- gamma: 0.001, 0.01, 0.1

kNN:
- k: 3, 5, 9, 15
- distancia: Euclideana/Manhattan

### 7.3 Criterio de selecao
Classificacao:
- maximizar F1 (ou Accuracy se classes equilibradas)

Regressao:
- minimizar RMSE/MAE e maximizar R2

---

## 8. Comparacao Final e Export de Resultados

### 8.1 Tabela comparativa
Nodes:
- Table Creator (nomes dos modelos)
- Concatenate (juntar metricas)
- Sorter (ordenar por metrica principal)

Criar 2 tabelas finais:
- comparacao classificacao
- comparacao regressao

### 8.2 Exportar artefactos
Nodes:
- CSV Writer (metricas finais)
- Image Writer (graficos, se necessario)

Guardar:
- metricas treino e teste;
- hiperparametros otimos;
- numero de linhas removidas por limpeza.

---

## 9. O que escrever no relatorio (mapeado ao enunciado)

### 9.1 Metodologia e dados
- origem do dataset e objetivo;
- etapas de limpeza e justificacoes;
- EDA com principais achados.

### 9.2 Modelos
- modelos testados e parametros;
- estrategia de divisao e validacao;
- tuning realizado.

### 9.3 Resultados e analise critica
- melhor modelo em teste (nao apenas treino);
- trade-off desempenho vs interpretabilidade;
- erros comuns observados.

### 9.4 Limitacoes e melhorias
- dataset sintetico (menos sinal real de comportamento);
- ausencia de variaveis sociais/contextuais;
- propostas: feature engineering, CV mais robusta, calibracao e novos dados.

---

## 10. Checklist final antes de entregar
1. Workflow executa de ponta a ponta sem erro.
2. Seeds fixas para reproducibilidade.
3. Metricas de treino e teste guardadas.
4. Graficos EDA exportados.
5. Tabelas comparativas prontas para o relatorio.
6. Justificacao tecnica das decisoes de limpeza e modelacao.
7. Conclusao com recomendacoes objetivas.

---

## 11. Versao minima viavel (se houver pouco tempo)
1. Limpeza: CSV Reader -> Type Converter -> Missing Value -> Duplicate Row Filter
2. EDA minima: Histogram + Correlation Matrix + GroupBy por genre
3. Classificacao: Random Forest + Scorer
4. Regressao: Random Forest Regressor + Numeric Scorer
5. Tuning curto: 2 ou 3 parametros principais por modelo
6. Tabela final com metricas e conclusao critica

Este caminho nao substitui a versao completa, mas garante resposta valida ao enunciado com qualidade aceitavel.