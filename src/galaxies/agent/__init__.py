"""Agente de hipóteses para o modelo de criativos virais.

Peças:
- ``protocol``: rótulo viral congelado, cofre de canais e grupos de features;
- ``spec``: catálogo fechado de experimentos que o agente pode propor;
- ``evaluation``: executor determinístico (LOCO + corte temporal, AUC intra-canal);
- ``hypotheses``: gerador Gemini e modo offline;
- ``ledger``: histórico cumulativo, campeões e relatório;
- ``graph``: orquestração com LangGraph.

Ponto de entrada: ``scripts/run_hypothesis_agent.py``.
"""
