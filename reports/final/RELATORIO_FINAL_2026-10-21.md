# Galaxies — relatório final de modelagem (21/10/2026)

## Resposta curta

**Nenhum modelo foi promovido.** Com 13.685 Shorts de 26 canais, avaliados num
corte de calendário e com seleção feita só dentro do treino, os modelos não
ordenam Shorts do mesmo canal melhor do que o acaso usando metadados e
cadência de publicação:

- **AUC média por canal: 0,491 [0,423; 0,560]**, com 0,5 sendo o acaso;
- **Lift@10: 1,15 [0,92; 1,44]**, com 1 sendo o acaso;
- a logística de referência também ficou perto do acaso (AUC média por canal
  0,540), e nenhum candidato a superou.

Os números melhores das versões anteriores (CatBoost Ranker com AUC de 0,666 e
Lift@10 de 2,33; regressões com "MAE vencedor") vinham de problemas de
protocolo que esta rodada corrigiu. O produto **não deve exibir previsão de
potencial ou prioridade** até que um modelo passe na validação prospectiva.

## 1. Pergunta do produto

Dentro de um canal e de um setor, quais **Shorts** vale priorizar antes de
publicar? A resposta tem de ser um ranking relativo ao canal, sem prometer
viralização e sem usar nenhuma informação posterior à publicação.

## 2. O que mudou em relação aos números anteriores

| Número anterior | Problema | Número honesto |
|---|---|---|
| CatBoost Ranker: AUC intra 0,666, Lift@10 2,33 | Misturava Shorts e vídeos longos; parte do acerto era "longo ganha de Short" | Com o mesmo split, só pares do mesmo formato: 0,608. Só Shorts, base nova: 0,455 em automobilismo e 0,491 nos três setores |
| AUC de 0,72 a 0,77 no corte de calendário (Fase 1) | A duração sozinha dava 0,734: o rótulo media o formato | Mesmo formato: de 0,51 a 0,60 |
| Regressão: MAE de 0,994, 0,912 e 0,930 | O vencedor era escolhido pelo MAE do próprio holdout | Escolha interna: 1,038, 0,926 e 0,930 (veja a seção 6) |
| Corte "temporal" por canal | Treinava com vídeos de outros canais publicados depois do teste (66% a 84% do treino) | Corte de calendário por setor |

Detalhes: `reports/final/modeling/auditoria-vazamentos-20261021.md`,
`fase1-protocolo-corrigido-20261021.md` e `fase2-ranking-shorts-20261021.md`.

## 3. Base final

- Histórico completo das playlists de uploads dos 30 canais selecionados
  (55 mil vídeos), coletado em 21/10/2026 pela YouTube Data API v3
  (`scripts/collect_channel_shorts_history.py`).
- **Short pela data de publicação:** nada antes de 12/07/2021, até 60 s até
  15/10/2024 e até 180 s depois disso. A API não marca Short; é uma regra de
  elegibilidade.
- 14.421 Shorts elegíveis; 13.685 com 30 dias ou mais e fora do cofre
  (automobilismo 7.419, beleza 3.451, moda 2.815).
- O cofre continua fechado: um canal por setor (Riachuelo, Niina Secrets e
  FlatOut Brasil), nunca lido.
- Vídeos publicados a partir de 21/10/2026 estão reservados para a validação
  final e bloqueados no código (`galaxies/prospective.py`).

## 4. Protocolo

1. **Rótulo:** um Short é positivo se o desempenho dele, ajustado pela idade,
   fica no top 10% dos Shorts do próprio canal. A curva de idade e o limiar são
   estimados só no treino e aplicados ao teste.
2. **Corte externo:** uma data por setor, no quantil 75% das publicações
   (automobilismo, beleza e moda). O treino termina antes dela. Teste com 3.413
   Shorts, 507 positivos e 20 canais.
3. **Seleção:** três dobras crescentes de calendário dentro do treino, com o
   rótulo recalculado em cada uma. O critério é a AUC média por canal, fixada
   antes de rodar. A escolha foi congelada (sha256 `27ab9b450728…`) antes de o
   teste ser lido.
4. **Candidatos:** logística (top-10 e contraste), HistGradientBoosting (pooled
   e por setor), CatBoost Ranker por setor e MLP pequena regularizada. Features:
   duração, metadados de publicação (categoria, hora, dia, contagens de título,
   descrição e tags) e cadência do canal.
5. **Proibido como feature:** views, likes, comentários, IDs, posição na série,
   inscritos e qualquer rótulo histórico sem data de observação.
6. **Intervalos:** bootstrap de 1.000 reamostras de canais reais, seed 42.

## 5. Resultado do ranking de Shorts

Execução: `data/modeling/runs/shorts-ranking-final-14k/`.

| Setor | Escolhido nas dobras internas | Canais | Positivos | AUC média por canal (IC95%) | Precision@10 | Recall@10 | Lift@10 (IC95%) | Δ AUC vs. referência (IC95%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Automobilismo | CatBoost Ranker | 8 | 269 | 0,455 [0,315; 0,582] | 17,6% | 12,3% | 1,21 [0,77; 1,68] | −0,091 [−0,308; +0,049] |
| Beleza | CatBoost Ranker | 7 | 86 | 0,534 [0,427; 0,630] | 10,1% | 10,5% | 1,00 [0,57; 1,62] | −0,027 [−0,146; +0,084] |
| Moda | HistGB pooled | 5 | 152 | 0,489 [0,393; 0,603] | 24,7% | 11,8% | 1,14 [0,78; 1,69] | −0,010 [−0,065; +0,045] |
| **Três setores** | — | 20 | 507 | **0,491 [0,423; 0,560]** | 17,1% | 11,8% | **1,15 [0,92; 1,44]** | [−0,150; +0,028] |

Diagnósticos, sem efeito na decisão:
- **AUC com pares somados:** 0,540 [0,472; 0,615].
- **Sem canais com menos de 100 pares:** a AUC média por canal fica em 0,511
  (16 canais). O canal com só 12 pares teve 0,17 e pesa igual na média principal.
- **Idade:** o rho entre score e idade dentro do canal ficou entre −0,16 e
  −0,02; o score não carrega a idade do vídeo.
- **Seleção interna instável:** os candidatos ficaram entre 0,45 e 0,64, e a
  mesma spec muda até 0,3 entre dobras.

Figuras em `reports/figures/shorts-ranking-final-14k/`:
- `topk_por_setor.png`: Precision, Recall e Lift no top K% de cada canal;
- `auc_por_canal.png`: AUC de cada canal no corte externo;
- `selecao_interna.png`: AUC de cada candidato, na média e em cada dobra.

## 6. Regressão contínua

A regressão de `residual_age_channel` foi corrigida na Fase 1, mas na base de
581 vídeos **com todos os formatos**
(`reports/final/modelagem/final-frozen-evaluation-phase1/`):
- só dois cortes (moda em canal novo e beleza no temporal) superaram o
  baseline com IC favorável, e nenhum setor confirmou nos dois cortes;
- os intervalos conformais de 90% vão de 5× a 23× em views;
- ela não foi refeita só com Shorts. **Não é utilizável no produto** e continua
  como referência metodológica. MAPE não se aplica a esse alvo residual; só
  faria sentido com `views_30d` positivo e de horizonte fixo.

## 7. Vazamentos e problemas corrigidos

1. O vencedor da regressão era escolhido pelo holdout. Agora vale a escolha
   interna, com dobras por tempo ou por canal conforme o corte.
2. O corte do ranking era por canal. Agora é de calendário por setor.
3. O rótulo do ranking misturava formatos. Agora a população é só de Shorts.
4. O alvo da regressão usava canais reservados na curva setorial. Agora esses
   canais ficam de fora.
5. O conformal vinha de outro modelo, e a calibração ignorava tempo e canal.
6. O OOD comparava cada ponto com ele mesmo.
7. A escolha das colunas e do alvo olhava o teste.
8. As taxas históricas de sucesso por formato usavam um rótulo calculado com a
   série completa. Já estavam bloqueadas, e o bloqueio foi verificado.
9. O código não tinha nenhum período intocado. Agora existe a reserva
   prospectiva com trava no código.

## 8. O que o produto pode e não pode dizer hoje

**Pode:**
- comparar criativos por atributos observáveis (duração, formato, horário,
  elementos da abertura) com dados descritivos do canal, marcados como
  descrição e não como previsão;
- mostrar o histórico do canal e a fila de acompanhamento dos Shorts novos.

**Não pode:**
- dar uma nota de "potencial" ou uma ordem de prioridade como previsão;
- dizer que um Short "vai viralizar" ou que uma mudança "vai aumentar views";
- comparar este resultado com artigos científicos: população, alvo e protocolo
  são diferentes.

## 9. Limitações

- **Rótulo:** cada vídeo tem uma única leitura de views (21/10), em idades
  diferentes. A curva de idade corrige a média, não o crescimento heterogêneo.
  O rótulo de 30 dias só existe para a fila prospectiva.
- **Shorts:** até 180 s é um critério operacional; vídeos horizontais curtos
  passam por ele.
- **Metadados:** título, descrição e tags vêm do snapshot da coleta e podem ter
  sido editados. A fila prospectiva grava esses campos no primeiro contato.
- **Canais:** são 20 no teste, só 5 em moda. Em moda, um canal tem 62% dos
  Shorts.
- **Conteúdo audiovisual:** VLM, thumbnail e transcrição não foram testados
  nesta base de Shorts. O resultado negativo vale para metadados e cadência.

## 10. Próximos passos, em ordem

1. **Manter a fila prospectiva rodando.** O agendamento diário às 09:00 já está
   criado, e a abertura única da validação final está prevista para perto do fim
   de dezembro (`docs/validacao_final_prospectiva.md`).
2. **Testar conteúdo nos Shorts, com a mesma seleção aninhada:**
   - primeiro, embedding de thumbnail dos 14 mil Shorts, que é barato e tem
     cobertura total;
   - depois, a abertura por VLM num protocolo único (Gemini 3.5 Flash-Lite,
     0–3 s). Para custo, uma amostra estratificada por canal antes de escalar.
   - Promover só com IC do delta contra a referência acima de zero.
3. **Mais canais**, sobretudo em moda e beleza. O IC fica largo porque há
   poucos canais, não poucos vídeos.
4. **Refazer a regressão só com Shorts**, se o produto ainda quiser uma métrica
   contínua, com `views_30d` da fila prospectiva como alvo.

## 11. Reprodução

```bash
python scripts/collect_channel_shorts_history.py --run-id shorts-history-2026-10-21
python scripts/build_shorts_feature_map.py --history shorts-history-2026-10-21
python scripts/evaluate_shorts_ranking.py --feature-map data/modeling/runs/feature-map-shorts-history-2026-10-21/feature_map.jsonl --run-id <novo-id>
python scripts/plot_shorts_ranking.py <novo-id>
```

Ambiente: Python 3.12, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.4 e
CatBoost. Seeds: 42 no ranking e 2026 no cofre e na regressão. 120 testes
automatizados passam, e o `ruff check` está limpo. Nada foi commitado.
