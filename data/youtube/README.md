# Dados do YouTube

Os diretórios gerados nesta pasta não são versionados por padrão.

- `channel_resolution/`: respostas usadas para resolver nomes em IDs de canais.
- `runs/<timestamp>/raw/`: respostas originais da YouTube Data API v3.
- `runs/<timestamp>/curated/`: documentos organizados, derivados do conteúdo bruto.

Não edite os arquivos em `raw/`. Correções de estrutura devem ser feitas nos scripts e regeneradas em `curated/`.
