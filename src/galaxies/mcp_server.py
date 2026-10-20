"""Servidor MCP fino: valida entradas e delega para a função Python testável."""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from .observability import McpTraceStore
from .prediction import FeatureMapRepository, PredictionService
from .schemas import CreativePayload, FeatureValue, PredictionRequest, Sector


def create_mcp_server(
    prediction_service: PredictionService | None = None,
    feature_repository: FeatureMapRepository | None = None,
    trace_store: McpTraceStore | None = None,
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
    def recent_mcp_calls(limit: int = 20) -> dict[str, Any]:
        """Retorna chamadas recentes, com entradas sanitizadas, para auditoria."""
        result = {"events": traces.recent(limit), "event_count": 0}
        result["event_count"] = len(result["events"])
        return result

    return server


mcp = create_mcp_server()


if __name__ == "__main__":
    mcp.run()
