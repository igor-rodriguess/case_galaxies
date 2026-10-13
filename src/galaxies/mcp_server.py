"""Servidor MCP fino: valida entradas e delega para a função Python testável."""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from .prediction import FeatureMapRepository, PredictionService
from .schemas import CreativePayload, FeatureValue, PredictionRequest, Sector


def create_mcp_server(
    prediction_service: PredictionService | None = None,
    feature_repository: FeatureMapRepository | None = None,
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

    @server.tool()
    def modeling_contract() -> dict[str, Any]:
        """Descreve o contrato e as limitações atuais da camada preditiva."""
        return {
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

    @server.tool()
    def model_status(sector: Sector) -> dict[str, Any]:
        """Informa se o modelo de um setor está treinado e promovido."""
        return service.model_status(sector.value).model_dump(mode="json")

    @server.tool()
    def validate_feature_payload(
        sector: Sector, features: dict[str, FeatureValue]
    ) -> dict[str, Any]:
        """Compara um mapa de features com o contrato do modelo promovido."""
        return service.validate_features(sector.value, features)

    @server.tool()
    def get_video_feature_map(video_id: str) -> dict[str, Any]:
        """Consulta o último mapa materializado para um video_id."""
        try:
            result = feature_store.get(video_id)
        except FileNotFoundError as exc:
            return {
                "status": "data_not_ready",
                "video_id": video_id,
                "message": str(exc),
            }
        return result or {"status": "not_found", "video_id": video_id}

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
        return service.predict(request).model_dump(mode="json")

    @server.tool()
    def rank_creatives(sector: Sector, creatives: list[CreativePayload]) -> dict[str, Any]:
        """Ordena criativos do mesmo setor pela previsão do modelo."""
        return service.rank(sector, creatives).model_dump(mode="json")

    return server


mcp = create_mcp_server()


if __name__ == "__main__":
    mcp.run()
