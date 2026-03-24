import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      "/api": {
        target: process.env.VITE_API_URL ?? "http://localhost:8000",
        changeOrigin: true,
        // Configure for SSE streaming - don't buffer responses
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            // Ensure SSE headers are preserved
            proxyRes.headers['Cache-Control'] = 'no-cache';
            proxyRes.headers['Connection'] = 'keep-alive';
            proxyRes.headers['X-Accel-Buffering'] = 'no';
          });
        },
      },
    },
  },
})
