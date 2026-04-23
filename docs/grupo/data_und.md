# CRISP-DM — Data Understanding

## Objetivo da fase

No workflow atual, a fase de Data Understanding ja existe como um bloco exploratorio executado em KNIME. O objetivo pratico e confirmar que o CSV foi lido corretamente, perceber a distribuicao das variaveis mais importantes e demonstrar, com evidencia, os problemas de qualidade que justificam a fase seguinte de preparacao.

Esta descricao foi alinhada com os nos realmente presentes em [grupo/ADI2526/workflow.knime](../../grupo/ADI2526/workflow.knime) e com os respetivos `settings.xml` do workflow.

## Caracterizacao do dataset a entrada

| Propriedade | Valor observado ou esperado |
|---|---|
| Ficheiro lido | `dataset/grupo/spotify_tracks.csv` |
| No de entrada | `CSV Reader (#2)` |
| Numero de registos | `100.500` |
| Numero de colunas | `21` |
| Numero de generos | `20` |
| Intervalo temporal | `2000-2024` |
| Log de apoio | `dataset/grupo/spotify_tracks_errors_log.csv` |

### Colunas presentes no CSV

| Grupo | Colunas |
|---|---|
| Identificacao | `track_id`, `track_name`, `artist_name`, `album_name` |
| Contexto | `release_year`, `genre`, `explicit`, `key`, `mode`, `time_signature` |
| Target | `popularity` |
| Audio | `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms` |

### Problemas de qualidade que o workflow procura evidenciar

Com base no gerador e no log de erros, o workflow foi montado para tornar visiveis estes problemas:

| Problema | Escala esperada | Colunas afetadas |
|---|---|---|
| Valores em falta | cerca de `27.000` celulas | 9 features de audio centrais |
| Outliers ou violacoes de dominio | varios milhares de ocorrencias | `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `loudness`, `popularity` |
| Duplicados | `500` copias completas | linhas repetidas do dataset |

Observacao importante: no estado atual do ficheiro, `energy`, `tempo` e `valence` continuam a ser lidas como colunas numericas. O workflow nao esta montado para resolver erros de tipo nestas colunas, mas sim missing values, outliers e duplicados.

## Leitura inicial no KNIME

### No 1: `CSV Reader (#2)`

Este no e a entrada do bloco de Data Understanding.

Verificacoes implicitas no workflow:

- `100.500` linhas e `21` colunas no output;
- `energy`, `tempo` e `valence` lidas como colunas numericas;
- presenca de valores em falta em varias audio features;
- leitura completa das colunas contextuais e identificadoras.

### No 2: `Statistics (#3)`

Ao contrario da versao anterior do documento, o no `Statistics` do workflow atual nao cobre 15 colunas numericas. As 12 colunas configuradas sao:

- `release_year`
- `popularity`
- `duration_ms`
- `danceability`
- `energy`
- `loudness`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`

E neste no que se confirmam os sinais mais relevantes para a preparacao:

- missing values nas audio features;
- valores fora de dominio em variaveis limitadas a `[0,1]`;
- extremos em `tempo`, `loudness` e `duration_ms`;
- distribuicao geral de `popularity` e `release_year`.

## Colunas categoricas e contextuais realmente exploradas

O workflow implementa cinco nos `Value Counter`, nao tres. Cada um verifica uma coluna diferente.

### Nos 4-8: `Value Counter`

| No | Coluna analisada | Leitura principal |
|---|---|---|
| `Value Counter (#4)` | `genre` | confirma os `20` generos |
| `Value Counter (#5)` | `explicit` | mostra a distribuicao entre `True` e `False` |
| `Value Counter (#6)` | `mode` | valida o dominio binario `0/1` |
| `Value Counter (#7)` | `key` | valida o dominio discreto `0-11` |
| `Value Counter (#8)` | `time_signature` | confirma predominancia de `4` e presenca de valores residuais |

Isto significa que a leitura categorica do workflow atual e mais ampla do que a descrita na documentacao antiga: `mode` e `key` tambem sao observados explicitamente nesta fase.

## Distribuicoes univariadas implementadas

### Nos 9-20: `Histogram`

O bloco exploratorio usa 12 histogramas. As colunas configuradas sao estas:

| No | Coluna |
|---|---|
| `Histogram (#9)` | `popularity` |
| `Histogram (#10)` | `duration_ms` |
| `Histogram (#11)` | `danceability` |
| `Histogram (#12)` | `energy` |
| `Histogram (#13)` | `loudness` |
| `Histogram (#14)` | `speechiness` |
| `Histogram (#15)` | `tempo` |
| `Histogram (#16)` | `release_year` |
| `Histogram (#17)` | `acousticness` |
| `Histogram (#18)` | `instrumentalness` |
| `Histogram (#19)` | `liveness` |
| `Histogram (#20)` | `valence` |

Comparando com a versao anterior do texto, duas diferencas sao importantes:

- `release_year` tem histograma proprio;
- `popularity` surge logo no inicio do bloco, nao apenas mais tarde.

## Deteccao visual de outliers

### Nos 21-24 e 35-37: `Box Plot`

O workflow atual usa 7 box plots para estas colunas:

- `duration_ms`
- `danceability`
- `energy`
- `loudness`
- `tempo`
- `popularity`
- `valence`

Estas vistas suportam diretamente a leitura dos casos que mais tarde sao tratados em preparacao:

- violacoes de dominio em `danceability`, `energy` e `valence`;
- extremos fisicos em `tempo` e `loudness`;
- amplitude de `duration_ms` e `popularity`.

## Relacoes bivariadas

### Nos 25-28: `Scatter Plot`

Os pares configurados no workflow sao:

| No | Eixo X | Eixo Y |
|---|---|---|
| `Scatter Plot (#25)` | `energy` | `loudness` |
| `Scatter Plot (#26)` | `energy` | `acousticness` |
| `Scatter Plot (#27)` | `danceability` | `valence` |
| `Scatter Plot (#28)` | `release_year` | `popularity` |

Estes graficos servem para separar padroes reais do dataset de problemas artificiais de qualidade e tambem para verificar relacoes esperadas entre variaveis sonoras e contexto temporal.

### No 29: `Linear Correlation`

O no `Linear Correlation (#29)` esta configurado para 15 colunas:

- `release_year`
- `popularity`
- `duration_ms`
- `danceability`
- `energy`
- `loudness`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `key`
- `mode`
- `time_signature`

No estado atual do workflow, este no corre apesar de existirem missings e avisa explicitamente essa situacao. Por isso, o seu papel aqui e exploratorio: perceber tendencias globais antes da limpeza, nao produzir a matriz final usada em modelacao.

## Duplicados no workflow atual

### No 30: `GroupBy (#40)`

O diagnostico de duplicados foi implementado por `track_id`.

Configuracao observada:

- coluna de agrupamento: `track_id`;
- agregacao: `Count` sobre `release_year`.

O output executado contem `100.000` linhas e duas colunas, o que mostra que o agrupamento consolidou as repeticoes presentes no CSV.

### No 31: `Row Filter (#39)`

Depois do agrupamento, o `Row Filter` mantem apenas os casos com `Count*(release_year) > 1`.

Resultado observado no workflow:

- `500` linhas no output.

Isto confirma que o problema de duplicacao existe efetivamente e e quantificado antes da fase de preparacao.

## O que esta fase prova no estado atual do workflow

| Problema | Evidencia no workflow | Consequencia para a fase seguinte |
|---|---|---|
| Valores em falta | `Statistics` e histogramas mostram lacunas nas audio features | imputacao em `Missing Value` |
| Valores fora de dominio | `Statistics`, box plots e scatter plots revelam limites violados | sanitizacao no `Expression` |
| Extremos artificiais | `tempo`, `loudness` e `duration_ms` aparecem com comportamento extremo | tratamento em `Numeric Outliers` |
| Duplicados | `GroupBy (#40)` e `Row Filter (#39)` isolam `500` casos | remocao em `Duplicate Row Filter (#41)` |
| Colunas pouco uteis para modelacao | identificadores e campos editoriais mantem alta cardinalidade | filtragem na preparacao |

## Nota sobre o que nao existe neste bloco

Ao contrario da versao anterior da documentacao, o workflow atual nao inclui um `Table Viewer` dedicado nesta fase. A inspecao visual foi distribuida por:

- `Statistics` para leitura descritiva;
- `Value Counter` para colunas categoricas;
- `Histogram` e `Box Plot` para distribuicoes e extremos;
- `Scatter Plot` e `Linear Correlation` para relacoes bivariadas.

## Conclusao da fase

No estado atual do workflow, Data Understanding ja nao e apenas um plano metodologico: e um bloco executado que demonstra que o CSV foi importado, que o dataset contem missing values, outliers e duplicados reais, e que a preparacao tem de resolver esses problemas antes de qualquer modelacao comparavel.

Isto liga diretamente a fase descrita em [Data Preparation](./data_prep.md), onde esses problemas deixam de ser apenas observados e passam a ser tratados no pipeline.

*** Add File: /home/marco/Projects/ADI2526/docs/grupo/data_prep.md
# CRISP-DM — Data Preparation

## Objetivo da fase

No workflow atual, a preparacao dos dados nao segue o plano idealizado da versao anterior da documentacao. O que existe em KNIME e um pipeline concreto que remove duplicados, sanitiza valores invalidos, imputa as colunas continuas relevantes, controla outliers e so depois separa os ramos de clustering e regressao.

Esta descricao foi reescrita para refletir o que esta realmente implementado em [grupo/ADI2526/workflow.knime](../../grupo/ADI2526/workflow.knime).

## Sequencia efetivamente implementada

O ramo principal de preparacao, antes da divisao em tarefas, e este:

1. `Duplicate Row Filter (#41)`
2. `Expression (#42)`
3. `Missing Value (#43)`
4. `Numeric Outliers (#45)`
5. `Column Filter (#47)`
6. separacao para clustering e regressao

Depois disso, o workflow abre dois ramos distintos:

- clustering: `Column Filter (#48)` -> `Normalizer (#50)`;
- regressao: `Table Partitioner (#51)` -> `Normalizer (#52)` e `Normalizer (Apply) (#53)` -> `Column Filter (#54)` e `Column Filter (#55)`.

## Etapa 2.1 — Remocao de duplicados

### No: `Duplicate Row Filter (#41)`

O workflow nao comeca por remover colunas de identificacao. Primeiro elimina duplicados.

Configuracao observada:

- comparacao sobre todas as colunas exceto `track_id`;
- modo `Remove duplicate rows`;
- retencao da primeira ocorrencia;
- preservacao da ordem.

Resultado observado:

- input implicito: `100.500` linhas;
- output executado: `100.000` linhas e `21` colunas.

Isto quer dizer que os `500` duplicados foram efetivamente removidos logo no inicio do pipeline.

## Etapa 2.2 — Sanitizacao, codificacao e features derivadas

### No: `Expression (#42)`

O workflow atual concentrou, num unico no, varias operacoes que antes estavam espalhadas por `Rule Engine`, `Math Formula` e filtros separados.

As transformacoes observadas sao estas:

- `popularity` fora de `[0,100]` passa a `missing`;
- `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness` e `valence` fora de `[0,1]` passam a `missing`;
- `key` fora de `0-11` passa a `missing`;
- `mode` diferente de `0` ou `1` passa a `missing`;
- `time_signature` fora do conjunto `{3,4,5,6,7}` passa a `missing`;
- `explicit` deixa de ficar booleana ou textual e e convertida in-place para `1` ou `0`;
- e criada a feature derivada `duration_min = duration_ms / 60000.0`;
- e criada a classe auxiliar `popularity_class` com os niveis `Baixa`, `Media` e `Alta`.

Resultado observado:

- output executado com `100.000` linhas e `23` colunas.

Isto altera o racional da preparacao face ao texto anterior: o workflow nao remove logo as linhas com violacoes de dominio nas colunas `[0,1]`; primeiro transforma esses valores em missing e deixa a decisao seguinte para o `Missing Value`.

## Etapa 2.3 — Imputacao e remocao de linhas com target invalido

### No: `Missing Value (#43)`

O no `Missing Value` e mais importante no workflow atual do que a versao anterior da documentacao sugeria, porque recebe tanto os missings originais do CSV como os missings produzidos pelo `Expression`.

### Estrategias observadas por coluna

| Coluna | Estrategia |
|---|---|
| `danceability` | Mean |
| `energy` | Median |
| `valence` | Mean |
| `speechiness` | Median |
| `acousticness` | Median |
| `instrumentalness` | Median |
| `liveness` | Median |
| `loudness` | Median |
| `tempo` | Median |
| `duration_ms` | Median |
| `popularity` | Remove Row |

Duas consequencias metodologicas relevantes:

- o workflow atual tambem imputa `duration_ms`;
- qualquer linha em que `popularity` fique missing e removida neste ponto.

Resultado observado:

- output executado com `99.353` linhas e `23` colunas.

Logo, nesta fase o workflow perde `647` linhas face as `100.000` que saiam do passo anterior, precisamente porque o target invalido nao e imputado.

## Etapa 2.4 — Tratamento de outliers continuos

### No: `Numeric Outliers (#45)`

O no `Numeric Outliers` nao esta limitado a `tempo` e `loudness`. No workflow atual ele trata tres colunas:

- `duration_ms`
- `loudness`
- `tempo`

Configuracao observada:

- detecao por IQR;
- `iqr-scalar = 3.0`;
- tratamento `Replace outlier values`;
- estrategia `Closest permitted value`.

Isto significa que os extremos sao truncados para o limite admissivel calculado pelo no, em vez de serem removidos.

## Etapa 2.5 — Selecao da base limpa comum as duas tarefas

### No: `Column Filter (#47)`

Depois da limpeza principal, o workflow cria a base comum a clustering e regressao com 15 colunas.

### Colunas mantidas

- `release_year`
- `popularity`
- `explicit`
- `danceability`
- `energy`
- `loudness`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `mode`
- `duration_min`
- `popularity_class`

### Colunas removidas

- `track_name`
- `artist_name`
- `album_name`
- `genre`
- `duration_ms`
- `key`
- `time_signature`
- `track_id`

Resultado observado:

- output executado com `99.353` linhas e `15` colunas.

Isto introduz tres diferencas centrais face ao documento anterior:

- `explicit_int` nao existe; o workflow usa a coluna `explicit` ja convertida para `0/1`;
- a modelacao passa a usar `duration_min`, nao `duration_ms`;
- `genre` sai desta base comum e so volta a ser trazido mais tarde nos ramos interpretativos de clustering.

## Etapa 2.6 — Ramo de clustering

### No: `Column Filter (#48)`

O ramo de clustering reduz a base comum as 10 variaveis efetivamente usadas no espaco metrico:

- `danceability`
- `energy`
- `loudness`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `duration_min`

### No: `Normalizer (#50)`

Configuracao observada:

- normalizacao `Min-Max`;
- intervalo novo `[0,1]`;
- aplicada a estas 10 colunas do ramo de clustering.

Resultado observado:

- output executado com `99.353` linhas e `10` colunas.

## Etapa 2.7 — Ramo de regressao

### No: `Table Partitioner (#51)`

O workflow usa `Table Partitioner`, nao `Partitioning` com seed `42`.

Configuracao observada:

- split relativo `80% / 20%`;
- amostragem `STRATIFIED`;
- coluna de estratificacao: `popularity_class`;
- seed fixa: `1`.

Saidas executadas:

- treino: `79.482` linhas;
- teste: `19.871` linhas.

### No: `Normalizer (#52)`

No treino, a normalizacao `Min-Max` nao cobre apenas as audio features. As 11 colunas normalizadas sao:

- `release_year`
- `danceability`
- `energy`
- `loudness`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `duration_min`

As colunas `explicit`, `mode` e `popularity` ficam fora desta transformacao.

### No: `Normalizer (Apply) (#53)`

O teste recebe exatamente o modelo aprendido no treino.

Ha um detalhe importante no estado atual do workflow: o no mostra um aviso de que alguns valores transformados ficam acima de `1.0` quando o teste contem valores fora do intervalo visto no treino. Isto nao significa leakage; significa apenas extrapolacao do `Min-Max` aprendido.

### Nos: `Column Filter (#54)` e `Column Filter (#55)`

Depois da normalizacao, treino e teste removem apenas `popularity_class`.

Resultado observado em ambos os ramos:

- `14` colunas por tabela;
- `13` preditores e `1` target.

## Features finais realmente usadas em regressao

| Papel | Colunas |
|---|---|
| Features `X` | `release_year`, `explicit`, `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `mode`, `duration_min` |
| Target `y` | `popularity` |

Ficam fora da regressao atual:

- `genre`
- `popularity_class`
- `duration_ms`
- `key`
- `time_signature`

## Leitura metodologica da preparacao atual

O workflow implementado difere do plano antigo em pontos relevantes:

| Tema | Plano antigo | Workflow atual |
|---|---|---|
| Conversao de `explicit` | `Rule Engine` para `explicit_int` | `Expression` converte `explicit` in-place para `0/1` |
| Violacoes em `[0,1]` | filtros de linha separados | `Expression` transforma em missing e `Missing Value` trata o caso |
| Outliers continuos | `tempo` e `loudness` | `tempo`, `loudness` e `duration_ms` |
| Feature de duracao | manter `duration_ms` na regressao | usar `duration_min` |
| Split treino ou teste | seed `42` | seed `1` |
| Particionamento | `Partitioning` | `Table Partitioner` |

## Estado esperado da base apos esta fase

No estado executado do workflow, a fase de preparacao produz:

- uma base limpa comum com `99.353` linhas e `15` colunas;
- um ramo de clustering com `99.353` linhas e `10` features normalizadas;
- um ramo de regressao com:
  - treino: `79.482` linhas;
  - teste: `19.871` linhas;
  - ambos com `14` colunas apos remover `popularity_class`.

## Conclusao da fase

A preparacao implementada no workflow e mais compacta e mais centralizada do que a versao anterior do documento. Em vez de espalhar decisoes por `Rule Engine`, `Math Formula` e multiplos filtros, o pipeline atual concentra a sanitizacao em `Expression`, usa `Missing Value` como etapa decisiva de imputacao e remocao do target invalido, e so depois estabiliza as bases finais para clustering e regressao.

E sobre essas saidas que a fase seguinte de [Modeling](./modeling.md) foi construida.

*** Add File: /home/marco/Projects/ADI2526/docs/grupo/modeling.md
# CRISP-DM — Modeling

## Objetivo da fase

O bloco de Modeling do workflow atual ja materializa duas linhas de trabalho distintas:

- clustering para segmentar faixas por perfil sonoro;
- regressao para prever `popularity`.

Ao contrario da versao anterior desta documentacao, o texto abaixo descreve a estrutura realmente presente em [grupo/ADI2526/workflow.knime](../../grupo/ADI2526/workflow.knime), incluindo diferencas importantes entre o que esta executado e o que esta apenas configurado.

## Inputs recebidos da preparacao

### Input A — clustering

O ramo de clustering recebe o output de `Column Filter (#48)` seguido de `Normalizer (#50)`:

- `99.353` linhas;
- `10` colunas;
- todas normalizadas por `Min-Max` para `[0,1]`.

As features usadas neste ramo sao:

- `danceability`
- `energy`
- `loudness`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `duration_min`

### Input B — regressao: treino

O treino recebe o output de `Table Partitioner (#51)` e `Normalizer (#52)`:

- `79.482` linhas;
- `14` colunas apos `Column Filter (#54)`.

### Input C — regressao: teste

O teste recebe o output de `Table Partitioner (#51)` e `Normalizer (Apply) (#53)`:

- `19.871` linhas;
- `14` colunas apos `Column Filter (#55)`.

Nota importante: o `Normalizer (Apply)` mostra um aviso porque alguns valores do teste ficam acima do intervalo visto no treino. Isso nao invalida o pipeline; apenas mostra extrapolacao do `Min-Max` aprendido.

## 3A — Clustering

### 3A.1 — Espaco de features realmente usado

O workflow atual faz clustering apenas sobre as 10 colunas do ramo `Column Filter (#48)`. Isso significa que o espaco metrico ja chega sem `genre`, `popularity`, `release_year`, `explicit`, `mode`, `key`, `time_signature` ou `popularity_class`.

Este ponto esta alinhado com a logica metodologica original: o clustering tenta captar perfil sonoro, nao metadados editoriais.

### 3A.2 — Elbow Method como esta implementado

O Elbow Method nao foi montado para `k = 2..10`. No workflow executado existem sete experiencias para:

- `k = 3`
- `k = 4`
- `k = 5`
- `k = 6`
- `k = 7`
- `k = 8`
- `k = 9`

Os nos correspondentes sao `k_Means (#58)`, `k_Means (#59)`, `k_Means (#60)`, `k_Means (#61)`, `k_Means (#62)`, `k_Means (#63)` e `k_Means (#113)`.

### Configuracao comum observada nesses nos

- mesmas 10 colunas de input;
- `Random initialization`;
- `Use static random seed = true`;
- `Seed value = 1`;
- `Max iterations = 99`.

### Como o WCSS e calculado no workflow

O workflow nao usa uma tabela de perda pronta a sair do `k-Means`. Em cada ramo candidato, o calculo e reconstruido desta forma:

1. `Joiner` junta cada linha ao centroide do respetivo cluster;
2. `Math Formula` calcula `sq_dist` como soma das distancias quadraticas as 10 coordenadas do centroide;
3. `GroupBy` soma `sq_dist` para obter o WCSS global;
4. outro `Math Formula` escreve o valor constante de `k` nessa linha agregada.

Depois, `Concatenate (#70)` reune as sete linhas e `Line Plot (#72)` desenha a curva com:

- eixo X: `k`;
- eixo Y: `Sum(sq_dist)`.

### Decisao efetivamente usada no workflow

O modelo final adotado no workflow esta fixado em `k = 5`.

## 3A.3 — Modelo final de `k-Means`

### No: `k_Means (#141)`

Este no e o modelo final atualmente usado no ramo principal de clustering.

Configuracao observada:

- `Number of clusters = 5`;
- `Random initialization`;
- `Seed = 1`;
- `Max iterations = 99`.

Ao contrario da versao anterior do documento, nao existe `Cluster Assigner` separado. O proprio output de dados do `k-Means` ja contem a coluna `Cluster`.

## 3A.4 — Interpretacao dos clusters no workflow atual

### Centroides

O workflow usa diretamente a tabela de centroides produzida pelo `k-Means` final. Em vez de recalcular medias com `GroupBy`, a interpretacao e preparada assim:

1. `Unpivot (#153)` transforma a tabela de centroides num formato longo;
2. `Column Renamer (#154)` renomeia as colunas para `Cluster`, `Features` e `Values`;
3. `Bar Chart (#108)` mostra os perfis dos centroides em grafico agrupado.

Este e o mecanismo real de leitura musical dos clusters no workflow.

### Tamanho dos clusters

O workflow tambem calcula tamanho e percentagem de cada cluster:

1. `GroupBy (#143)` conta linhas por `Cluster`;
2. `Cross Joiner (#157)` junta o total global;
3. `Math Formula (#159)` calcula `Percent = (Count / Total) * 100`;
4. `Column Renamer (#161)` ajusta os nomes para `Count` e `Total`;
5. `Column Filter (#162)` mantem `Cluster`, `Count` e `Percent`.

Isto substitui, na pratica, a verificacao manual de clusters degenerados descrita no documento antigo.

## 3A.5 — Comparacao hierarquica

### No: `Row Sampler (#131)`

O metodo hierarquico trabalha sobre uma amostra absoluta de `5.000` linhas.

Configuracao observada:

- `ABSOLUTE`;
- `5000` linhas;
- amostragem `RANDOM`;
- `seed = 1`.

### No: `Hierarchical Clustering (#132)`

O metodo hierarquico do workflow atual usa:

- `numberClusters = 5`;
- distancia `Euclidean`;
- linkage `COMPLETE`.

Isto e uma diferenca importante face ao texto anterior, que falava em `Ward` e em inspecao por `Dendrogram`.

No workflow atual:

- nao existe no `Dendrogram`;
- nao existe no separado de corte ou assigner;
- o proprio no hierarquico ja produz a tabela com coluna `Cluster` usada na avaliacao seguinte.

### No: `Silhouette Coefficient (#133)`

O coeficiente de silhouette executado sobre esta amostra tem valor medio global de aproximadamente `0.0732`.

Este valor e bastante mais baixo do que o esperado para clusters muito bem separados, o que deve ser lido no relatorio como indicio de sobreposicao relevante entre perfis sonoros na amostra hierarquica.

## 3A.6 — Diagnosticos adicionais de silhouette

Além do ramo principal e da comparacao hierarquica, o workflow guarda tres ramos adicionais de amostragem estratificada por `Cluster`:

- `Row Sampler (#187)` -> `Silhouette Coefficient (#188)`
- `Row Sampler (#189)` -> `Silhouette Coefficient (#190)`
- `Row Sampler (#191)` -> `Silhouette Coefficient (#192)`

Cada `Row Sampler` usa:

- amostra relativa de `20%`;
- modo `STRATIFIED`;
- estratificacao por `Cluster`;
- `seed = 1`.

Os tres valores executados nestes ramos sao aproximadamente:

- `0.1609`
- `0.1782`
- `0.1740`

O workflow mantem estes ramos como comparacao adicional entre solucoes proximas de `k-Means`, mas os nomes dos nos nao identificam explicitamente qual destes scores corresponde a cada valor de `k`. Por rigor, o documento regista aqui os valores observados sem lhes atribuir um rotulo adicional nao confirmado.

## 3A.7 — PCA e validacao qualitativa com `genre`

### PCA

O workflow atual usa:

- `PCA Compute (#144)`;
- `PCA Apply (#145)` para a tabela completa com clusters;
- `PCA Apply (#146)` para a tabela reduzida dos centroides.

Depois disso:

- `Color Manager (#148)` e `Scatter Plot (#147)` mostram os pontos completos no plano PCA;
- `Color Manager (#150)` e `Scatter Plot (#149)` mostram os centroides no mesmo espaco, com pontos maiores.

### Reintroducao de `genre`

Como `genre` foi removido da base comum na preparacao, ele e trazido de volta apenas para interpretacao:

1. `Column Filter (#111)` isola a coluna `genre`;
2. `Joiner (#136)` junta `genre` a tabela clustered por `RowID`;
3. `Pivot (#138)` cria uma matriz `Cluster x Genre` de contagens;
4. `Bar Chart (#142)` mostra a composicao relativa de cada cluster em formato stacked.

Isto substitui a abordagem descrita no documento antigo com `GroupBy` generico. Aqui ha um `Pivot` explicito e um `Bar Chart` stacked ja configurado.

## 3A.8 — Saidas efetivamente materializadas neste ramo

No estado atual do workflow, o bloco de clustering ja produz:

- curva de elbow construida manualmente;
- modelo final `k = 5`;
- tabela de tamanho e percentagem dos clusters;
- visualizacao dos centroides por feature;
- projecoes PCA dos pontos e dos centroides;
- composicao dos clusters por genero;
- silhouette do metodo hierarquico e silhouettes adicionais em amostras estratificadas.

## 3B — Regressao

### 3B.1 — Features realmente usadas

O ramo de regressao trabalha sobre 13 preditores e um target.

### Preditores

- `release_year`
- `explicit`
- `danceability`
- `energy`
- `loudness`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `mode`
- `duration_min`

### Target

- `popularity`

Logo, a regressao atual ja nao usa `duration_ms` nem uma coluna `explicit_int` separada.

## 3B.2 — Regressao linear multipla

### Nos: `Linear Regression Learner (#163)` -> `Regression Predictor (#164)` -> `Numeric Scorer (#165)`

Este ramo esta executado.

Configuracao observada no learner:

- target: `popularity`;
- todos os restantes campos da tabela entram como features;
- constante incluida.

### Diagnostico visual do ramo linear

Tambem estao executados:

- `Scatter Plot (#175)` para `popularity` real vs prevista;
- `Math Formula (#177)` com `residual = popularity - Prediction (popularity)`;
- `Histogram (#178)` para a distribuicao dos residuos.

## 3B.3 — Arvore de regressao

### Nos: `Simple Regression Tree Learner (#166)` -> `Simple Regression Tree Predictor (#167)` -> `Numeric Scorer (#168)`

Este ramo tambem esta executado.

Ha aqui uma diferenca clara face ao documento antigo: a arvore nao esta limitada a profundidade `10` nem a `50` registos por no. Os parametros observados sao:

- `maxLevels = -1`;
- `minNodeSize = -1`;
- `minChildSize = -1`.

Ou seja, o workflow atual deixou a arvore com limites abertos ou default, em vez de impor uma poda explicita.

### Diagnostico visual do ramo arvore

Tambem estao executados:

- `Scatter Plot (#180)`;
- `Math Formula (#182)` para residuos;
- `Histogram (#181)`.

## 3B.4 — Random Forest Regression

### Nos: `Random Forest Learner (Regression) (#172)` -> `Random Forest Predictor (Regression) (#173)` -> `Numeric Scorer (#174)`

Este ramo existe, mas esta apenas configurado, nao executado.

Configuracao observada no learner:

- target `popularity`;
- `100` arvores;
- `seed = 1`;
- bootstrap ativo;
- amostragem de colunas em modo `SquareRoot`;
- mesmos 13 preditores dos outros modelos.

No estado atual do workflow, isto significa:

- o ramo RF esta preparado para comparacao;
- ainda nao ha metricas calculadas nesta versao executada do workflow.

### Diagnostico visual do ramo RF

Os nos seguintes tambem existem, mas permanecem por executar no fluxo final:

- `Scatter Plot (#185)`;
- `Math Formula (#186)`;
- `Histogram (#184)`.

## 3B.5 — O que a comparacao entre modelos ja permite e o que ainda falta

### Ja existe no workflow

- um ramo linear executado com `Numeric Scorer`;
- um ramo de arvore executado com `Numeric Scorer`;
- um ramo random forest configurado;
- um padrao repetido de diagnostico residual por modelo.

### Ainda nao existe como bloco fechado

- uma tabela consolidada que reuna `R2`, `RMSE` e `MAE` dos tres modelos num unico ponto do workflow;
- execucao do ramo random forest;
- identificacao automatica do melhor modelo num ramo final;
- analise do erro por `genre` no ramo de regressao.

Isto e importante: a versao anterior da documentacao descrevia estes elementos como ja integrantes do workflow, mas na estrutura atual eles continuam a ser extensoes desejaveis e nao passos materializados.

## 3B.6 — Leitura correta do estado atual da regressao

O workflow ja suporta uma comparacao seria entre baseline linear e arvore de regressao, com visualizacao `Predicted vs Actual` e histogramas de residuos para ambos. No entanto, a fase ainda nao esta totalmente fechada porque:

- a random forest nao foi executada;
- nao existe uma tabela comparativa unica;
- nao ha ramo de erro por genero neste bloco.

Por isso, a regressao esta metodologicamente bem encaminhada, mas menos consolidada do que o clustering.

## Estado atual do bloco de Modeling

### Clustering

- [x] features de clustering claramente definidas;
- [x] elbow implementado manualmente;
- [x] modelo final `k = 5` presente e executado;
- [x] comparacao hierarquica presente;
- [x] PCA e analise por genero presentes.

### Regressao

- [x] treino ou teste e normalizacao sem leakage;
- [x] regressao linear executada;
- [x] arvore de regressao executada;
- [ ] random forest executada;
- [ ] tabela comparativa unica de metricas;
- [ ] analise do erro por genero.

## Conclusao da fase

Na versao atual do workflow, o bloco de Modeling ja prova duas coisas com clareza:

- o clustering foi implementado de forma rica, com elbow manual, comparacao hierarquica, PCA, composicao por genero e leitura de centroides;
- a regressao ja tem dois modelos executados e um terceiro ramo preparado, mas ainda precisa de consolidacao final para fechar a comparacao entre modelos.

Esta leitura e mais fiel ao que realmente existe em KNIME do que a documentacao anterior, que misturava decisoes metodologicas desejadas com passos que o workflow ainda nao materializou.