import { resolve } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@smear/web-core": resolve(__dirname, "../../packages/web-core/src"),
    },
  },
  server: {
    port: 5174,
  },
  preview: {
    allowedHosts: [
      ".up.railway.app",
      "play-smear.com",
      "www.play-smear.com",
      ".play-smear.com",
    ],
  },
});
