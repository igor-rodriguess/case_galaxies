# Creative Battle no MCP

`creative_battle` compara duas versões pré-publicação do **mesmo canal** e
**setor**. A entrada contém `baseline` (A), `candidate` (B), `channel_id` e
`sector`; cada versão é um `CreativePayload` com seu mapa de features.

```text
Versão A + Versão B
        │
        ▼
validação de setor/canal → baseline congelada → delta + confiança + trace MCP
```

A tool só retornará escore, probabilidade e vencedor quando houver um modelo
**Shorts-only** promovido. No estado atual do projeto, ela devolve
`model_not_ready`: a interface deve exibir o diff das versões e explicar que a
comparação preditiva aguarda validação prospectiva. Não deve escolher vencedor.

## Limite científico

O resultado é uma comparação **associativa** do modelo congelado para o rótulo
histórico `top-10% intra-canal`. Ele não estima views absolutas, não prova que
uma edição causará uma melhora e não garante viralização. Uma alegação causal
exige teste controlado (por exemplo, A/B real ou experimento de publicação).

Além disso, a baseline antiga foi retirada da tool porque misturava vídeos
longos e Shorts. No corte temporal Shorts-only ela não se sustentou; voltar a
exibi-la como recomendação seria um viés de formato, não evidência sobre o
criativo.

## Exemplo de chamada

```json
{
  "sector": "moda",
  "channel_id": "UC...",
  "baseline": {"features": {"duration_seconds": 19, "title": "Versão A"}},
  "candidate": {"features": {"duration_seconds": 12, "title": "Versão B"}}
}
```

O trace MCP registra só setor, canal, contagem de features, versão do artefato,
quantos inputs mudaram, vencedor e confiança; valores criativos não são escritos
no log de observabilidade.
