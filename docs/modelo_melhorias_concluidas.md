# Fechamento técnico — melhorias do modelo

Data: 19/10/2026. Protocolo: `viral-intra-canal-v1`.

## Checklist de execução do MD

| Tarefa do MD | Estado | Evidência |
|---|---|---|
| Embeddings, PCA intra-dobra e diagnóstico pareado | concluída | `embedding-ablation-20261018/report.md` |
| Contraste top 20% vs. base 50% | concluída | `train_label=clean20vs50`, com teste automatizado |
| kNN de outro canal, defasado 30 dias | concluída e não promovida | AUC 0,569; diferença -0,001 |
| Agente com resposta estruturada e catálogo fechado | concluída | `run_hypothesis_agent.py` e `agent/hypotheses.py` |
| Modo canal conhecido, corte 75/25 por canal e histórico defasado | concluída no código e testada | `agent/channel_history.py` |
| Snapshots 1/3/7/14/30 dias | concluída e em coleta | `collect_daily_snapshots.py` |
| Detector precoce | aguardando observações futuras | exige snapshots reais de 1 dia e 30 dias do mesmo vídeo |

### Status do Gemini

Em 19/10/2026 foi criado o lote retomável `gemini-case-control-20261019` da fila caso-controle. A API respondeu HTTP 503 (`UNAVAILABLE`, alta demanda) nos três primeiros vídeos e o processo pausou sem marcar nenhum como concluído. Os diagnósticos estão preservados em `data/features/runs/gemini-case-control-20261019/`; a retomada deve reutilizar o mesmo manifesto, sem reenviar rótulos ou repetir resultados válidos.

## Decisões sustentadas pelos testes

| Variante na mesma população (1.808 vídeos, 27 canais fora do cofre) | AUC intra-canal [IC95%] | Diferença pareada vs. referência |
|---|---:|---:|
| Referência de metadados + estatísticas de texto | 0,570 [0,521; 0,626] | — |
| + thumbnail CLIP, PCA 16 dentro da dobra | 0,585 [0,537; 0,637] | +0,014 [-0,024; +0,053] |
| + thumbnail CLIP, PCA 32 dentro da dobra | 0,573 [0,525; 0,621] | +0,003 [-0,061; +0,057] |
| + embedding do título, PCA 16 | 0,518 [0,469; 0,578] | -0,052 [-0,089; -0,013] |
| + thumbnail e título, PCA 16 | 0,544 [0,499; 0,598] | -0,027 [-0,069; +0,010] |
| Logística de diferenças de pares thumbnail+título | 0,540 [0,479; 0,605] | diagnóstico, não promovida |
| kNN temporal de thumbnail entre canais | 0,569 [0,520; 0,632] | -0,001 [-0,022; +0,018] |

O thumbnail 16D é o único sinal com ganho pontual, mas o intervalo da diferença inclui zero; por isso fica como candidato, não como feature promovida. Título, combinação multimodal e logística pareada não entram no modelo atual. O melhor resultado global que continua válido é contraste limpo + vizinhos de outros canais: **0,602 [0,546; 0,667]**.

## O que foi fechado no código

- Embeddings retomáveis de thumbnail, título e título+descrição; PCA só é ajustado no treino de cada dobra.
- Contraste de treino top 20% contra base 50%, sem alterar a avaliação top 10%.
- kNN apenas de outro canal e de vídeos ao menos 30 dias anteriores ao alvo.
- Divisão temporal de canal conhecido: últimos 25% de **cada** canal no teste.
- `known_channel_history_features`: taxa suavizada por formato/faixa de duração e margem de similaridade viral–não viral; nunca usa contagem anterior de vídeos, métricas pós-publicação nem futuro.
- Catálogo do agente registra `train_label` e o grupo `text_stats`; embeddings continuam no executor dedicado porque a ablação não justificou promovê-los ao agente genérico.
- Coletor diário rastreável em `scripts/collect_daily_snapshots.py`, com janelas 1/3/7/14/30 dias e respostas brutas. Ele não fabrica `views_30d` retroativas.

## Primeiro snapshot real

Em 19/10/2026 o coletor registrou 73 observações retornadas pela API: 12 na janela de 3 dias, 24 na de 7, 20 na de 14 e 17 na de 30. Cada resposta e seu horário de coleta estão em `data/youtube/daily_snapshots/2026-10-19/`; a janela de 1 dia não tinha vídeos elegíveis neste lote.

## Próxima medição que realmente pode melhorar a decisão

Rodar o coletor diariamente sobre uploads novos. Em 30 dias, gerar um alvo observado `views_30d`; só então é legítimo avaliar o detector precoce (sinais de 1 h/1 d) separadamente do modelo antes da publicação.
