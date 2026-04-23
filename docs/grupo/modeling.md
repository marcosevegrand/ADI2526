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