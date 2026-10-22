# Investigação científica do ranking temporal — 21/10/2026

## Decisão honesta

O resultado anterior de `18/80` não era suficiente para justificar a promessa
de que um criativo “vai viralizar”. A investigação corrigiu dois problemas de
integridade temporal antes de tentar melhorar qualquer número:

1. As taxas históricas por formato/duração reutilizavam `viral`, um rótulo
   calculado com toda a série do canal. Mesmo filtrando vídeos publicados no
   passado, o percentil que definia o rótulo conhecia vídeos futuros.
2. A curva de idade e o limiar top-10% haviam sido congelados na população
   inteira. No backtest temporal, o treino poderia assim depender do período
   que deveria representar o futuro.

O executor agora bloqueia taxas de resultado sem timestamp de observação
anterior ao criativo-alvo, e o corte temporal ajusta idade e limiar **somente
no treino**. Os  resultados abaixo são, portanto, mais exigentes e não usam
views, likes, comentários, título, descrição ou rótulos de desempenho como
entrada.

## Pergunta avaliada

Para cada canal, os 75% mais antigos formam o treino e os 25% mais recentes o
teste. Um vídeo de teste é positivo se seu desempenho ajustado por idade supera
o limiar que o canal tinha no treino. Há 1.890 itens de treino e 641 itens de
teste, em 27 canais; 98 itens no teste são positivos.

O orçamento operacional é selecionar aproximadamente o top 10% de cada canal
(73 vídeos no total). A métrica central é AUC intra-canal: em pares de um
positivo e um negativo do mesmo canal, qual fração recebe nota maior para o
positivo. Aleatório é 0,50.

## Comparação reproduzível

Artefatos: `data/modeling/runs/known-channel-asof-candidates-ci-20261021/`.
Os intervalos são bootstrap de canais, não de linhas individuais.

| Modelo | AUC intra-canal (IC95%) | Precisão@10 (IC95%) | Recall@10 (IC95%) | Lift@10 (IC95%) |
|---|---:|---:|---:|---:|
| Logística pooled, top-10 | 0,524 [0,426; 0,627] | 20,5% [10,4; 32,9] | 15,3% [8,1; 23,8] | 1,34 [0,72; 2,06] |
| HistGradientBoosting pooled, contraste | 0,665 [0,590; 0,735] | 26,0% [14,9; 39,2] | 19,4% [12,5; 27,6] | 1,70 [1,13; 2,37] |
| HistGradientBoosting por setor, contraste | 0,664 [0,580; 0,748] | 30,1% [17,9; 43,8] | 22,4% [15,5; 29,7] | 1,97 [1,35; 2,58] |
| **CatBoost Ranker por setor, pareado** | **0,666 [0,589; 0,743]** | **35,6% [22,1; 50,0]** | **26,5% [19,2; 34,9]** | **2,33 [1,71; 3,00]** |
| MLP pequena pooled, contraste | 0,564 [0,475; 0,655] | 20,5% [11,3; 31,9] | 15,3% [9,0; 23,3] | 1,34 [0,81; 1,98] |

O ranker encontra **26 dos 98** positivos ao revisar 73 itens. Uma escolha
aleatória encontraria cerca de 11; portanto ele é uma ferramenta de triagem,
não uma garantia individual. A MLP não supera as árvores e não é promovida.

## Por que ranking pareado faz sentido

- A tarefa do produto é escolher os melhores criativos de um canal, não
  estimar uma verdade universal de “viral”. O ranker aprende a ordenar itens
  comparáveis no mesmo contexto.
- [Understanding Image Virality](https://arxiv.org/abs/1503.02318) também
  trata virality relativa em pares e mostra que semântica de alto nível é mais
  útil que sinais visuais rasos. Nosso ranker segue a ideia de comparação,
  mas não reivindica comparabilidade numérica: o domínio e o alvo são outros.
- [MVP](https://arxiv.org/abs/2507.00950) combina representações multimodais,
  metadados e gradient boosting. Isso sustenta testar árvores e fusão; não
  autoriza copiar sua métrica, pois seus dados, split e alvo não são os do
  Galaxies.
- [Tan, Lee e Pang](https://aclanthology.org/P14-1017/) reforçam que controlar
  autor e contexto é essencial ao atribuir efeito ao conteúdo. Por isso os
  pares e as métricas são intra-canal e temporais.

## O que ainda não é válido afirmar

- Não é válido dizer que AUC 0,666 “supera os artigos”. Eles usam populações,
  horizontes e métricas diferentes.
- Não é válido prometer que uma alteração sugerida no vídeo causará mais views;
  SHAP e comparador de cenários descrevem associação no modelo.
- Este corte temporal virou conjunto de desenvolvimento depois desta
  investigação. O ranker é o candidato líder, não o campeão congelado.

## Próximo experimento que pode promovê-lo

1. Selecionar arquitetura e hiperparâmetros somente por dobras temporais
   internas dos 1.890 itens de treino.
2. Congelar o ranker por setor, sem mais mudanças no outer test.
3. Validar uma única vez em um período posterior, com snapshots prospectivos
   no mesmo horizonte (idealmente 30 dias), ou em canais reservados com essa
   mesma coleta.
4. Só promover se AUC, Lift@10 e precisão continuarem acima da referência com
   IC do delta que não cruze zero em cada setor aplicável.

Para elevar o teto além disso, a prioridade não é uma rede maior: é obter
rótulos de horizonte fixo e uma única extração VLM consistente, depois testar
ablação multimodal no mesmo protocolo. Redes profundas com estes milhares de
linhas e centenas de positivos tenderiam a memorizar canal/protocolo, como já
ocorreu com a MLP multimodal no cofre.
