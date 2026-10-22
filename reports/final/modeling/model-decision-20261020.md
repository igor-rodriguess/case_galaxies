# Decisão de modelo — freeze de 20/10/2026

## Decisão

A baseline logística com metadados pré-publicação e cadência é a única referência
implantável neste freeze. A MLP com sinais de VLM e trecho transcrito fica como
experimento de pesquisa; ela não foi promovida porque falhou no cofre de canais
isolados.

O produto deve descrever a saída como **ranking experimental de potencial
relativo**, não como previsão garantida de viralização nem como previsão de views
absolutas.

## População e protocolo

- Treino da baseline: 2.531 vídeos maduros, 262 positivos e 27 canais, fora do
  cofre.
- Rótulo: top 10% de desempenho ajustado por idade dentro do próprio canal.
- Entradas da baseline: duração/formato, tamanho de título/descrição, tags,
  categoria, hora/dia e cadência. Nenhuma métrica do próprio vídeo depois de
  publicado entra como feature.
- Avaliação: leave-one-channel-out, corte temporal e cofre trancado por canal.
- VLM: 1.000 respostas globais; o subconjunto comparável e maduro fora do cofre
  tinha 523 vídeos, 110 positivos e 27 canais. Os protocolos de 3 s e 15 s foram
  tratados separadamente ou harmonizados; não foram misturados como se fossem a
  mesma feature.

## Resultado da referência implantável

| Corte | AUC intra-canal | IC 95% | Leitura |
|---|---:|---:|---|
| LOCO | 0,534 | [0,468; 0,599] | ganho pequeno e incerto sobre aleatório |
| Temporal | 0,497 | [0,404; 0,602] | sem evidência de generalização temporal |
| Cofre | 0,648 | [0,573; 0,789] | melhor sinal, mas apenas 3 canais |

A calibração cruzada por canal reduziu o Brier OOF de 0,246 para 0,093. Isso
melhora a interpretação probabilística, mas não aumenta a capacidade de ranking.

![SHAP global da baseline](C:/Users/Inteli/Documents/case_galaxies/reports/figures/baseline-freeze-calibrated-20261020/shap-global.png)

![Calibração LOCO](C:/Users/Inteli/Documents/case_galaxies/reports/figures/baseline-freeze-calibrated-20261020/calibration-loco.png)

## Resultado multimodal e motivo da não promoção

No subconjunto harmonizado, a MLP com transcrição derivada obteve AUC LOCO 0,615
e Lift@10 de 1,56; contudo, a diferença pareada para a referência teve IC 95%
[-0,019; 0,219], cruzando zero, e a AUC temporal foi 0,415.

No cofre, a mesma MLP caiu para AUC 0,431 contra 0,637 da referência. A diferença
pareada foi -0,206, IC 95% [-0,412; -0,107]. Como o intervalo inteiro é negativo,
ela não pode ser tratada como campeã. Esse resultado é mais valioso do que um
score interno alto: ele evita promover um modelo que não transfere para canais
novos.

## Explicabilidade e limites

SHAP global e contribuições locais descrevem associações no modelo. Eles não
provam que alterar duração, descrição, categoria ou horário causará viralização.
O comparador de cenários deve manter esse aviso e mostrar baixa confiança se uma
entrada estiver fora da distribuição de treino ou se a cadência estiver ausente.

## Próxima evidência necessária

1. Aumentar canais independentes por setor, especialmente automobilismo.
2. Repetir VLM em um único protocolo temporal para remover o desvio entre janelas
   de 3 s e 15 s.
3. Coletar snapshots prospectivos na mesma idade de observação, se houver acesso
   autorizado aos canais, para validar um alvo de views em horizonte fixo.
4. Só reabrir a competição multimodal com uma nova base e uma única avaliação
   final em canais nunca usados durante busca.
