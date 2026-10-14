import type { NextConfig } from "next";

// The web demo is a client of the FastAPI app (`uv run uvicorn app.main:app --port 8000`).
// Requests to /api/* are proxied there, so the browser never needs CORS.
const API_URL = process.env.API_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${API_URL}/api/:path*` },
      { source: "/docs", destination: `${API_URL}/docs` },
      { source: "/openapi.json", destination: `${API_URL}/openapi.json` },
    ];
  },
};

export default nextConfig;
