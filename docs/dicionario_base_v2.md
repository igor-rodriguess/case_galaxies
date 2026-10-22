# Dicionário da base v2 de Shorts

Gerada por `scripts/build_viral_dataset.py` (regras em `src/galaxies/viral_dataset.py`).
Retrato e funil: `reports/final/dataset_v2/retrato_base_v2.md`. Os arquivos ficam em
`data/dataset_v2/`, fora do git, como a coleta de onde saem.

| Camada | Arquivo | Linhas |
|---|---|---|
| 1. Bruto | `data/youtube/shorts_history/shorts-history-2026-10-21/normalized_records.json` | 22.442 |
| 2. Limpa | `shorts_clean.parquet` | 14.118 |
| 3. Rótulo | `shorts_labels.parquet` (13.142 com rótulo) | 14.118 |

## `shorts_clean.parquet`: features possíveis

Tudo aqui é conhecido no momento da publicação.

| Coluna | Tipo | Papel | Descrição |
|---|---|---|---|
| `video_id` | texto | chave | ID do YouTube |
| `channel_id`, `channel_title` | texto | agrupamento | canal; nunca feature |
| `sector` | texto | feature | automotivo, beleza ou moda |
| `published_at` | data UTC | tempo | publicação; usada para cortes de calendário |
| `collected_at` | data UTC | controle | data da coleta das views |
| `title`, `description`, `tags` | texto | feature | texto escrito pelo criador |
| `title_key` | texto | controle | título normalizado, usado para achar reuploads |
| `category_id`, `definition` | texto | feature | categoria do YouTube e resolução |
| `duration_seconds` | número | feature | duração |
| `age_days` | número | controle | idade na coleta; nunca feature (atalho de idade) |

## `shorts_labels.parquet`: alvo, nunca feature

| Coluna | Tipo | Descrição |
|---|---|---|
| `views` | número | views na coleta |
| `prior_shorts` | inteiro | Shorts anteriores do canal na base limpa (até 30 contam) |
| `baseline_median_views` | número | mediana das views dos até 30 Shorts anteriores do canal |
| `outlier_score` | número | `views ÷ baseline_median_views` |
| `log_outlier_score` | número | log10 do score, só para rotulados (alvo contínuo) |
| `label_status` | texto | `rotulado`, `jovem demais` (< 60 dias) ou `histórico curto` (< 10 anteriores) |
| `viral` | 0/1 | `outlier_score ≥ 3`, só para rotulados |

## Base v3

`python scripts/build_viral_dataset.py --version v3` grava em `data/dataset_v3/`, com as mesmas colunas e
mais estas:

- **Fontes:** a coleta de 21/10 e `shorts-history-2026-10-21-criadores`, com 29 criadores de moda e
  beleza achados por `scripts/discover_creator_channels.py` e conferidos à mão.
- **`channel_type`** (marca, mídia ou criador): classificação manual em `config/selected_channels.json`.
  Serve para agrupar e separar resultados; nunca é alvo.
- **`label_status` ganha dois motivos**, ambos de `audit_boosted`, que usa likes só para auditar o
  rótulo:
  - `provável impulsionado`: likes/view abaixo de 10% da mediana dos 30 Shorts anteriores do canal;
  - `likes ocultos`: o canal tem likes ocultos em mais da metade dos vídeos.

## Hipóteses declaradas

- **60 dias de maturidade:** não foi medido nesta base. Com duas coletas do mesmo vídeo daria para medir
  a curva de crescimento.
- **k = 3:** convenção de outlier; a sensibilidade para k = 2, 5 e 10 está no retrato.
- **Viés de idade:** a taxa de virais sobe de 14,0% (60–180 dias) para 18,2% (> 2 anos). Todo
  modelo nesta base deve reportar a correlação da nota com a idade dentro do canal.
- **Canais pequenos:** Vult (6 rotulados), FlatOut Brasil (12) e FullpowerTV (28) têm pouca base;
  avaliações por canal devem exigir um mínimo de vídeos.
