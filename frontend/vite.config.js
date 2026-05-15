import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite proxies /api and /ws to FastAPI during development so the React
// dev server (5173) and FastAPI (8000) can coexist without CORS pain.
// In production, build with `npm run build` and serve the dist/ folder
// behind the same origin as the API.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
        changeOrigin: true,
      },
    },
  },
  build: {

    outDir: "dist",
    sourcemap: true,
  },
});
