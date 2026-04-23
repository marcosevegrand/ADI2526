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

Alemd do ramo principal e da comparacao hierarquica, o workflow guarda tres ramos adicionais de amostragem estratificada por `Cluster`:

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