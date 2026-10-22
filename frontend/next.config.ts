import type { NextConfig } from "next";

// A interface só lê e chama tools pela ponte Python (src/galaxies/frontend_bridge.py).
const bridge = process.env.GALAXIES_BRIDGE_URL ?? "http://127.0.0.1:8765";

const nextConfig: NextConfig = {
  // Dados ao vivo do MCP: os componentes clientes buscam na ponte, sem cache de componente.
  cacheComponents: false,
  // A ponte e o painel abrem em 127.0.0.1; sem isto o HMR é recusado e a página não hidrata.
  allowedDevOrigins: ["127.0.0.1"],
  devIndicators: false,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${bridge}/api/:path*` }];
  },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
