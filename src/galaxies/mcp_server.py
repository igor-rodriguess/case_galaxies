"""Servidor MCP fino: valida entradas e delega para a função Python testável."""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from .modeling.inference import FrozenBaselinePredictor
from .observability import McpTraceStore
from .prediction import FeatureMapRepository, PredictionService
from .schemas import (
    CreativeBattleResponse,
    CreativePayload,
    FeatureValue,
    PredictionRequest,
    Sector,
)

# A baseline anterior misturava vídeos longos e Shorts. A auditoria de 21/10
# mostrou que ela aprendia sobretudo o formato; portanto não pode alimentar
# uma recomendação de produto. Um preditor pode ser injetado explicitamente em
# testes ou depois de uma promoção formal de modelo Shorts-only.
CREATIVE_BATTLE_PROMOTION_MESSAGE = (
    "Nenhum modelo Shorts-only foi promovido para comparação preditiva. "
    "A avaliação temporal externa atual não superou a referência; use a comparação "
    "somente como diff descritivo até a validação prospectiva."
)


def _scenario_features(
    payload: CreativePayload,
    *,
    sector: Sector,
    channel_id: str,
) -> dict[str, FeatureValue]:
    """Completa as chaves de contexto e rejeita cenários inconsistentes."""
    features = dict(payload.features)
    supplied_sector = features.get("sector")
    if supplied_sector is not None and str(supplied_sector) != sector.value:
        raise ValueError("O setor dentro do cenário deve corresponder ao setor da comparação.")
    supplied_channel = features.get("channel_id")
    if supplied_channel is not None and str(supplied_channel) != channel_id:
        raise ValueError("O canal dentro do cenário deve corresponder ao canal da comparação.")
    features["sector"] = sector.value
    # ``channel_id`` não entra na baseline logística atual, mas preservá-lo aqui
    # deixa explícito que as duas versões representam a mesma origem.
    features["channel_id"] = channel_id
    return features


def create_mcp_server(
    prediction_service: PredictionService | None = None,
    feature_repository: FeatureMapRepository | None = None,
    trace_store: McpTraceStore | None = None,
    creative_battle_predictor: FrozenBaselinePredictor | None = None,
) -> MCPServer:
    """Cria um servidor com dependências substituíveis para testes e implantação."""
    server = MCPServer(
        "galaxies-performance",
        instructions=(
            "Use as tools para consultar modelos, validar features, prever views e "
            "ordenar criativos. Nunca invente uma previsão quando model_status indicar "
            "que o setor ainda não possui modelo promovido."
        ),
    )
    service = prediction_service or PredictionService()
    feature_store = feature_repository or FeatureMapRepository()
    traces = trace_store or McpTraceStore()
    battle_predictor = creative_battle_predictor

    def load_battle_predictor() -> FrozenBaselinePredictor:
        nonlocal battle_predictor
        if battle_predictor is None:
            raise FileNotFoundError(CREATIVE_BATTLE_PROMOTION_MESSAGE)
        return battle_predictor

    @server.tool()
    def modeling_contract() -> dict[str, Any]:
        """Descreve o contrato e as limitações atuais da camada preditiva."""
        result = {
            "input": {
                "sector": [item.value for item in Sector],
                "features": "Mapa achatado produzido por build_modeling_dataset.py.",
                "video_id": "Opcional; usado para rastreabilidade.",
            },
            "output": [
                "predicted_metric",
                "interval_low",
                "interval_high",
                "confidence_flag",
                "relative_to_channel",
                "top_drivers",
                "explanation",
            ],
            "rule": "Sem modelo promovido, a tool retorna model_not_ready e não chuta.",
            "creative_battle": {
                "tool": "creative_battle",
                "requires": ["setor comum", "channel_id comum", "versão A", "versão B"],
                "returns": [
                    "delta de escore de ranking",
                    "delta de probabilidade calibrada",
                    "alertas de confiança",
                    "inputs do modelo que mudaram",
                ],
                "limitation": (
                    "Só fica preditiva após a promoção de um modelo Shorts-only. No estado "
                    "atual retorna model_not_ready; não demonstra o efeito causal de editar "
                    "um criativo nem garante viralização."
                ),
            },
        }
        traces.record(
            tool_name="modeling_contract",
            input_summary={},
            output_summary={"status": "ok"},
        )
        return result

    @server.tool()
    def model_status(sector: Sector) -> dict[str, Any]:
        """Informa se o modelo de um setor está treinado e promovido."""
        result = service.model_status(sector.value).model_dump(mode="json")
        traces.record(
            tool_name="model_status",
            input_summary={"sector": sector.value},
            output_summary={"ready": result["ready"], "model_name": result.get("model_name")},
        )
        return result

    @server.tool()
    def validate_feature_payload(
        sector: Sector, features: dict[str, FeatureValue]
    ) -> dict[str, Any]:
        """Compara um mapa de features com o contrato do modelo promovido."""
        result = service.validate_features(sector.value, features)
        traces.record(
            tool_name="validate_feature_payload",
            input_summary={"sector": sector.value, "feature_count": len(features)},
            output_summary={
                "valid": result["valid"],
                "model_ready": result["model_ready"],
                "warning_count": len(result["warnings"]),
            },
        )
        return result

    @server.tool()
    def get_video_feature_map(video_id: str) -> dict[str, Any]:
        """Consulta o último mapa materializado para um video_id."""
        try:
            result = feature_store.get(video_id)
        except FileNotFoundError as exc:
            result = {
                "status": "data_not_ready",
                "video_id": video_id,
                "message": str(exc),
            }
            traces.record(
                tool_name="get_video_feature_map",
                input_summary={"video_id": video_id},
                output_summary={"status": result["status"]},
            )
            return result
        response = result or {"status": "not_found", "video_id": video_id}
        traces.record(
            tool_name="get_video_feature_map",
            input_summary={"video_id": video_id},
            output_summary={"status": response.get("status", "ok")},
        )
        return response

    @server.tool()
    def predict_performance(
        sector: Sector,
        features: dict[str, FeatureValue],
        video_id: str | None = None,
    ) -> dict[str, Any]:
        """Prevê visualizações com intervalo e evidências quando há modelo promovido."""
        request = PredictionRequest(
            sector=sector,
            features=features,
            video_id=video_id,
        )
        result = service.predict(request).model_dump(mode="json")
        traces.record(
            tool_name="predict_performance",
            input_summary={
                "sector": sector.value,
                "video_id": video_id,
                "feature_count": len(features),
            },
            output_summary={
                "status": result["status"],
                "confidence_flag": result["confidence_flag"],
                "model_run_id": result.get("model_run_id"),
            },
        )
        return result

    @server.tool()
    def rank_creatives(sector: Sector, creatives: list[CreativePayload]) -> dict[str, Any]:
        """Ordena criativos do mesmo setor pela previsão do modelo."""
        result = service.rank(sector, creatives).model_dump(mode="json")
        traces.record(
            tool_name="rank_creatives",
            input_summary={"sector": sector.value, "creative_count": len(creatives)},
            output_summary={"status": result["status"], "ranked_count": len(result["items"])},
        )
        return result

    @server.tool()
    def creative_battle(
        sector: Sector,
        channel_id: str,
        baseline: CreativePayload,
        candidate: CreativePayload,
    ) -> dict[str, Any]:
        """Compara duas versões do mesmo criativo no modelo congelado de ranking.

        O resultado só responde como o *modelo* muda diante das entradas A e B.
        Não use o delta como prova de causalidade, nem como promessa de viralização.
        """
        try:
            predictor = load_battle_predictor()
            comparison = predictor.compare(
                _scenario_features(baseline, sector=sector, channel_id=channel_id),
                _scenario_features(candidate, sector=sector, channel_id=channel_id),
            )
        except (FileNotFoundError, OSError) as exc:
            result = CreativeBattleResponse(
                status="model_not_ready",
                sector=sector,
                channel_id=channel_id,
                comparison_scope=(
                    "Comparação preditiva indisponível até a promoção de um modelo Shorts-only "
                    "validado prospectivamente."
                ),
                warnings=[str(exc) or CREATIVE_BATTLE_PROMOTION_MESSAGE],
            ).model_dump(mode="json")
            traces.record(
                tool_name="creative_battle",
                input_summary={"sector": sector.value, "channel_id": channel_id},
                output_summary={"status": result["status"]},
            )
            return result
        except ValueError as exc:
            result = CreativeBattleResponse(
                status="invalid_input",
                sector=sector,
                channel_id=channel_id,
                comparison_scope="Os dois cenários devem representar o mesmo setor e canal.",
                warnings=[str(exc)],
            ).model_dump(mode="json")
            traces.record(
                tool_name="creative_battle",
                input_summary={"sector": sector.value, "channel_id": channel_id},
                output_summary={"status": result["status"]},
            )
            return result

        delta = comparison["delta"]
        probability_delta = float(delta["viral_probability_calibrated"])
        winner = "candidate" if probability_delta > 0 else "baseline" if probability_delta < 0 else "tie"
        confidence_levels = {
            comparison["baseline"]["confidence"]["level"],
            comparison["candidate"]["confidence"]["level"],
        }
        confidence_flag = "low" if "low" in confidence_levels else "normal"
        warnings = [
            "A diferença é associativa ao modelo congelado; valide a versão vencedora em experimento "
            "controlado antes de atribuir causalidade."
        ]
        if not comparison["changed_model_inputs"]:
            warnings.append("Nenhuma feature usada pelo modelo mudou entre as versões A e B.")
        result = CreativeBattleResponse(
            status="ok",
            sector=sector,
            channel_id=channel_id,
            model_run_id=predictor.artifact_dir.name,
            model_name="logistic_frozen_ranking_baseline",
            comparison_scope=comparison["comparison_scope"],
            changed_model_inputs=comparison["changed_model_inputs"],
            baseline=comparison["baseline"],
            candidate=comparison["candidate"],
            delta={key: float(value) for key, value in delta.items()},
            winner=winner,
            confidence_flag=confidence_flag,
            warnings=warnings,
        ).model_dump(mode="json")
        traces.record(
            tool_name="creative_battle",
            input_summary={
                "sector": sector.value,
                "channel_id": channel_id,
                "baseline_feature_count": len(baseline.features),
                "candidate_feature_count": len(candidate.features),
            },
            output_summary={
                "status": result["status"],
                "winner": result["winner"],
                "confidence_flag": result["confidence_flag"],
                "changed_model_input_count": len(result["changed_model_inputs"]),
                "model_run_id": result["model_run_id"],
            },
        )
        return result

    @server.tool()
    def recent_mcp_calls(limit: int = 20) -> dict[str, Any]:
        """Retorna chamadas recentes, com entradas sanitizadas, para auditoria."""
        result = {"events": traces.recent(limit), "event_count": 0}
        result["event_count"] = len(result["events"])
        return result

    return server


mcp = create_mcp_server()


if __name__ == "__main__":
    mcp.run()
