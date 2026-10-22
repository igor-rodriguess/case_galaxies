"""Inicia a interface do Galaxies e a ponte para o servidor MCP real.

Uso:
    python scripts/run_frontend.py                      # MCP em memória
    python scripts/run_frontend.py --mcp-url http://127.0.0.1:8000/mcp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxies.frontend_bridge import serve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mcp-url", help="Servidor MCP streamable-http já em execução.")
    args = parser.parse_args()
    serve(args.host, args.port, args.mcp_url)


if __name__ == "__main__":
    main()
