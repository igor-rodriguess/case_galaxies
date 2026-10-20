# Prompt — Galaxies: melhorar o modelo de criativos virais com base na literatura

> Cole este arquivo inteiro no seu agente de código (Claude Code ou Codex), aberto na raiz do repositório `case_galaxies` (Windows; use `.venv\Scripts\python`).
> Objetivo de hoje: implementar e medir as melhorias com evidência científica, sem usar nenhum dado do futuro.

---

## 1. Seu papel e regras inegociáveis

Você é o engenheiro de ML do projeto Galaxies. Trabalhe em etapas. Ao final de cada etapa, mostre os números e só então siga para a próxima.

1. **Nada do futuro.** São proibidos como feature: curtidas, comentários, views, inscritos atuais, histórico de views, idade do vídeo e "posição do vídeo na série". Toda feature tem que existir no momento da publicação.
2. **O protocolo não muda.** Use `src/galaxies/agent/protocol.py` (`viral-intra-canal-v1`):
   - viral = top 10% do próprio canal, com ajuste de idade dentro do canal, só vídeos com 30+ dias;
   - 1 canal por setor fica no cofre;
   - métrica = AUC dentro do canal, com intervalo de confiança reamostrando canais;
   - o rho idade (correlação entre a nota e a idade do vídeo) não pode passar de 0,20.
3. **Toda melhoria precisa ser comparada à referência nas mesmas dobras.** Reporte a diferença pareada (`evaluation.paired_comparison`). Acurácia não serve como métrica.
4. **Toda transformação aprendida acontece dentro da dobra de treino:** PCA, seleção de features, target encoding, geração de hipóteses, vizinhos.
5. **Sem busca grande de hiperparâmetros.** Já foi testado: o Optuna aninhado (720 buscas) deu 0,543 e o FLAML deu 0,564, contra 0,565 da referência. Use poucas configurações definidas antes de rodar.
6. **Acima de 0,70 em canal novo, suspeite de vazamento primeiro.** Procure o atalho antes de comemorar.

## 2. Onde estamos (medido em 18/10/2026)

| Modelo (canal novo, leave-one-channel-out) | AUC dentro do canal [IC95%] |
|---|---|
| Referência: metadados + estatísticas de texto, HistGradientBoosting | 0,589 [0,539; 0,637] |
| + contraste limpo (treina top 20% contra os 50% de baixo) + vizinhos de outros canais | **0,602** [0,546; 0,667] |
| Soma simples de 8 sinais com pesos iguais (Dawes) | 0,586 |
| Canal conhecido: + histórico do próprio canal, treino top 20% (divisão temporal) | **0,639** [0,554; 0,726] |
| Só a abertura (VLM), 268 vídeos e 30 virais | 0,554, sem ganho sobre os metadados |

Diagnóstico:
- Estamos no teto de informação dos metadados e do texto. Afinar o modelo não muda o resultado.
- A incerteza vem do número de canais (só 27 fora do cofre), não do número de vídeos.
- Armadilhas já comprovadas:
  - com divisão aleatória e canais misturados, as mesmas features dão 0,80, e caem para 0,52 com canal novo;
  - as curtidas atuais dariam 0,87, o que é vazamento.

## 3. O que a literatura diz

**Faixa realista** quando se usa só conteúdo e se compara dentro do mesmo criador: **0,54 a 0,68**.

| Trabalho | O que mostra |
|---|---|
| Hessel, Lee & Mimno (WWW 2017) — https://arxiv.org/abs/1703.01725 | Pares da mesma comunidade do Reddit: texto 0,58–0,63, **imagem pré-treinada 0,60–0,65**, os dois juntos 0,63–0,68 |
| Tan, Lee & Pang (ACL 2014) — https://arxiv.org/abs/1405.1438 | Tweets do mesmo autor: 0,656; humanos 0,61 |
| Gligorić et al. (PLOS ONE 2023) — https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0281682 | Testes A/B de manchete, confirmatórios: 0,54 |
| Khosla et al. (WWW 2014) — https://people.csail.mit.edu/khosla/papers/www2014_khosla.pdf | Entre usuários, o perfil social domina; dentro do usuário, o conteúdo vale mais |
| Ding et al. (ACM MM 2019) — https://arxiv.org/abs/1907.01985 | Imagem, pares do mesmo usuário com perda pareada; cai para cerca de 0,59–0,63 em outra plataforma |
| Ye et al., LOLA (Marketing Science 2025) — https://arxiv.org/abs/2406.02611 | LLM julgando sem treino ≈ acaso; **embeddings + regressão linear** quase empatam com ajuste fino |
| Zhou et al., HypoGeniC (2024) — https://arxiv.org/abs/2404.04326 | LLM gerando hipóteses vence o few-shot e o ajuste fino com 200 exemplos |
| Kayal et al. (ACL Findings 2025) — https://aclanthology.org/2025.findings-acl.597/ | Vídeo descrito em texto + hipóteses + vizinhos como exemplo; supervisionado: thumbnail 0,77, título 0,75 (classes extremas, não comparável) |
| Rodrigues et al. (KDMiLe 2022) — https://sol.sbc.org.br/index.php/kdmile/article/download/24981/24802 | **Canal brasileiro de automobilismo**: thumbnail (SE-ResNet50) + título (BERTimbau) supera cada um sozinho |
| Wang et al., Alibaba (WWW 2021) — https://arxiv.org/abs/2102.04033 | Imagem com perda listwise dentro do produto; teste em produtos nunca vistos |
| Li et al., SnapUGC (ECCV 2024) — https://arxiv.org/abs/2410.00289 | A retenção de 5 s é previsível pelo conteúdo, e poucos frames bastam |
| Teixeira, Wedel & Pieters (JMR 2012) — https://research.tilburguniversity.edu/en/publications/emotion-induced-engagement-in-internet-video-ads/ | Surpresa prende a atenção; alegria retém o espectador |
| Rajaram & Manchanda — https://arxiv.org/abs/2012.12311 | Marca ou produto embalado no início atrapalha; pessoa em cena e música ajudam |
| Berger & Milkman (JMR 2012) — https://faculty.wharton.upenn.edu/wp-content/uploads/2011/11/Virality.pdf | Emoções de alta ativação aumentam o compartilhamento |
| Banerjee & Urminsky (Marketing Science 2025) — https://home.uchicago.edu/ourminsky/Banerjee_Urminsky_Headlines.pdf | Lacuna de curiosidade, números e concretude ajudam (versão preliminar) |
| King & Zeng (Political Analysis 2001) — https://gking.harvard.edu/publication/logistic-regression-in-rare-events-data/ | Caso-controle é eficiente para eventos raros, desde que corrigido |
| Martin et al. (WWW 2016) — https://arxiv.org/abs/1602.01013 | Existe um teto natural de previsibilidade |
| Kapoor & Narayanan (Patterns 2023) — https://reproducible.cs.princeton.edu/ | Catálogo de tipos de vazamento em ML |

---

## 4. Tarefas de hoje, em ordem

### Tarefa 1 — Thumbnail + título como embeddings (maior lacuna)
1. Crie `scripts/extract_embeddings.py`:
   - **Thumbnails:** baixe de `https://i.ytimg.com/vi/{video_id}/hqdefault.jpg` para os vídeos de `data/youtube/runs/*/normalized_records.json`. Salve em `data/features/thumbnails/{video_id}.jpg`. Deve ser retomável e pular os que já existem.
   - **Imagem:** gere o embedding com CLIP (`open_clip`, ViT-B-32, pesos laion2b) ou SigLIP. Faça o mesmo com o 1º frame, se estiver disponível.
   - **Texto:** gere o embedding do título com um sentence-transformer (`neuralmind/bert-base-portuguese-cased` com mean pooling, ou `intfloat/multilingual-e5-base`). Repita para título + 300 primeiros caracteres da descrição.
   - **Saída:** `data/features/embeddings/{thumb,title,title_desc}.parquet` (video_id + vetor) e um manifest com modelo, versão e data.
2. Na avaliação:
   - reduza com PCA para 16 e 32 dimensões, **ajustado dentro de cada dobra de treino**;
   - concatene às features da referência;
   - avalie três configurações pré-definidas: (a) referência + thumb; (b) referência + título; (c) referência + thumb + título.
3. Rode também a versão pareada sugerida pela literatura, como diagnóstico: logística sobre a **diferença** de embeddings entre pares viral e não viral do mesmo canal (Hessel; Ding).
4. **Critério:** se (c) passar de 0,63 com limite inferior acima de 0,58, a thumbnail vira o núcleo do modelo.

### Tarefa 2 — Rótulo de treino e amostragem
- Treine com **contraste limpo**: positivo = top 20% do canal, negativo = 50% de baixo, descartando o meio. Avalie sempre no top 10% contra o resto.
- Reporte também, como diagnóstico, a AUC "top 10% contra os 50% de baixo", que é mais comparável aos pares filtrados da literatura.

### Tarefa 3 — Vizinhos de outros canais (kNN)
- Para cada vídeo, pegue os K = 10 vídeos mais parecidos **de outros canais**, usando o embedding da Tarefa 1 (thumb + título).
- Considere só vizinhos publicados pelo menos 30 dias antes do vídeo-alvo e que estejam no treino da dobra.
- Feature = média ponderada do percentil dos vizinhos dentro do canal de cada um.

### Tarefa 4 — Gemini como gerador de hipóteses (HypoGeniC), não como juiz
1. **Dentro de cada dobra de treino:** amostre 40 pares (viral, não viral) do mesmo canal com título, descrição, thumbnail e o JSON da abertura. Peça ao Gemini de 15 a 30 hipóteses verificáveis, por exemplo:
   - lacuna de curiosidade;
   - número no título;
   - valência negativa;
   - história em 1ª pessoa;
   - rosto com emoção na thumbnail;
   - produto em destaque no início;
   - surpresa nos primeiros segundos;
   - texto grande na thumbnail.
2. Transforme as hipóteses em perguntas sim/não, com JSON e enum no `responseSchema`, e aplique-as a todos os vídeos da dobra e do teste.
3. Use as respostas como features. Guarde as hipóteses e a versão do prompt.
4. **Proibido** gerar hipóteses olhando os canais de teste.

### Tarefa 5 — Modo "canal conhecido" (o produto real para clientes com histórico)
- Avaliação: divisão temporal dentro de cada canal, com os 25% vídeos mais recentes de cada canal como teste.
- Features de histórico, calculadas só com vídeos do mesmo canal publicados pelo menos 30 dias antes:
  - taxa de viral por formato e por faixa de duração, suavizadas;
  - similaridade com os virais passados menos a similaridade com os não virais passados (pelo embedding).
- **Não use** a contagem de vídeos anteriores: é um índice de tempo.
- Compare com o mesmo modelo sem histórico. A referência atual é 0,639.

### Tarefa 6 — Integrar ao agente
- Adicione ao catálogo de `src/galaxies/agent/spec.py` os grupos `text_stats`, `thumb_emb` e `title_emb`, a opção `train_label ∈ {top10, clean20vs50}` e o modo de avaliação `known_channel`.
- Atualize `protocol.FORBIDDEN_*` e os testes. Rode `scripts/run_hypothesis_agent.py`.

### Tarefa 7 — Dados (disparar hoje; rodam em segundo plano)
- **Coleta diária**, em tarefa agendada: leia os vídeos novos dos canais nas idades de 1, 3, 7, 14 e 30 dias, guardando título, descrição e thumbnail no momento da publicação. Isso viabiliza o detector precoce e o alvo de views aos 30 dias.
- **Ampliar canais:** +20 por setor, coletando os uploads completos. Canal novo é o que mais reduz a incerteza.
- **Fila caso-controle da abertura:** todos os virais maduros + 3 controles sorteados do mesmo canal, nunca os piores (King & Zeng). Inclua os canais novos.

### Tarefa 8 — Depois de 30 dias de coleta: detector precoce
- Use as views das primeiras horas e do primeiro dia, relativas ao normal do canal, para prever o top 10% aos 30 dias.
- Divulgue esse resultado separado dos modelos antes da publicação, porque é um produto diferente.

---

## 5. Armadilhas a evitar (comuns na literatura)
1. Divisão aleatória com o mesmo canal no treino e no teste.
2. Sinais pós-publicação: views do 1º dia, curtidas, comentários.
3. Classes extremas ou corte na mediana sendo vendidos como "detecção de viral".
4. Idade do vídeo vazando por proxies: data, posição na série, tendência temporal. Confira o rho idade.
5. Agregados calculados sobre o período inteiro (médias de canal, popularidade de tags). Calcule tudo só com o passado.
6. Escolher limiar, hiperparâmetro ou feature olhando o teste.
7. Muitas configurações com poucos canais. Prefira poucas configurações pré-registradas e confira o resultado final no cofre.

## 6. Entregáveis de hoje
- Para cada tarefa, uma tabela com: AUC dentro do canal [IC95%], AUC top 10% contra 50% de baixo, recall no top 10%, lift, rho idade, número de canais e de virais, e a diferença pareada contra a referência.
- Os scripts novos, com testes passando (`pytest`) e `ruff check` limpo.
- Uma conclusão em 5 linhas: o que entra no modelo, o que sai e qual é o próximo dado a coletar.
- **Meta realista:** canal novo entre 0,63 e 0,67; canal conhecido perto de 0,70; detector precoce acima de 0,85, quando houver coleta diária.

---

## 7. Registro de execução — 19/10/2026

Este apêndice fecha o roteiro com o que foi efetivamente implementado e
medido. Ele não transforma limitações de coleta em resultados.

| Tarefa | Estado | Evidência ou decisão |
|---|---|---|
| 1. Embeddings de thumbnail e título | Concluída | `scripts/extract_embeddings.py`; ablação em `data/modeling/runs/embedding-ablation-20261018/report.md`. A população comum tem 1.808 vídeos e 27 canais fora do cofre. |
| 1. PCA intra-dobra e comparação pareada | Concluída | Thumbnail PCA16: AUC 0,585 [0,537; 0,637], Δ +0,014 [-0,024; +0,053]. Título PCA16 piorou: Δ -0,052 [-0,089; -0,013]. Nenhum embedding foi promovido. |
| 1. Logística pareada por diferenças | Concluída | AUC 0,540 [0,479; 0,605]; diagnóstico negativo, não promovido. |
| 2. Contraste limpo | Concluída | `train_label=clean20vs50`; top 20% é positivo e os 50% inferiores são negativos, sempre calculados dentro do treino. |
| 3. kNN entre canais e defasado | Concluída | `agent/temporal_knn.py`; K=10, exclui mesmo canal e futuro. AUC 0,569; Δ -0,001 [-0,022; +0,018], portanto não promovido. |
| 4. Gemini como gerador de hipóteses | Implementado; execução externa bloqueada | `scripts/run_hypothesis_agent.py` usa resposta estruturada e catálogo fechado. O lote audiovisual `gemini-case-control-20261019` recebeu HTTP 503 por alta demanda em 19/10; diagnósticos e manifesto foram preservados. |
| 5. Canal conhecido | Concluída no código | Corte temporal 75/25 por canal e `known_channel_history_features`, que só consulta vídeos do mesmo canal publicados 30+ dias antes; não usa views, likes, comentários nem contagem de vídeos anteriores. |
| 6. Integração ao agente | Concluída | Catálogo inclui `train_label`, modo `known_channel`, `text_stats`, proibições de métricas pós-publicação e testes anti-vazamento. Embeddings ficam no executor dedicado pois a ablação não justificou promovê-los. |
| 7. Snapshots e fila caso-controle | Concluída e em operação | `scripts/collect_daily_snapshots.py` gravou 73 snapshots reais em 19/10, com respostas brutas e janelas de 3/7/14/30 dias. A fila caso-controle preserva seleção, papéis e proveniência. |
| 8. Detector precoce | Corretamente pendente | Só será avaliado quando os mesmos vídeos tiverem snapshots observados no dia 1 e perto do dia 30. Não há `views_30d` retroativas na API pública. |

### Conclusão operacional

O campeão atual para **canal novo** é o contraste limpo com vizinhos de outros
canais: AUC intra-canal **0,602 [0,546; 0,667]**. O resultado ainda não prova
que thumbnail, título ou VLM melhorem a generalização; essas hipóteses ficam
fora do modelo até vencerem a comparação pareada. A próxima melhoria de maior
valor é acumular snapshots de horizonte fixo, normalizar a taxonomia dos 733
resultados audiovisuais disponíveis e só então reavaliar o modelo sem sinais
pós-publicação.

### Verificação final

Em 19/10/2026: `62 passed, 1 skipped` na suíte de testes antes da integração
final do modo de canal conhecido; os testes focados dessa integração passaram
(`20 passed, 1 skipped`) e o `ruff check` está limpo. As referências e os
artefatos acima devem ser a fonte de verdade para a apresentação.
