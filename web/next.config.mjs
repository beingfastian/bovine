/** @type {import('next').NextConfig} */
const nextConfig = {
  // Fully static. No server, no serverless functions, no cold starts -- the
  // whole point of moving to the web was a link anyone can open instantly.
  output: "export",
  images: { unoptimized: true },
  reactStrictMode: true,

  webpack: (config) => {
    // onnxruntime-web references its binary as `new URL("...wasm",
    // import.meta.url)`, so webpack resolves it and emits a 14 MB copy into
    // _next/static/media/. Nothing ever fetches that copy: lib/infer.ts sets
    // ort.env.wasm.wasmPaths to the versioned /ort/ directory that
    // scripts/copy-ort-wasm.mjs populates.
    //
    // `emit: false` keeps the module graph intact -- the URL is still
    // generated, it just is not written to disk -- and halves what has to be
    // uploaded on every deploy.
    config.module.rules.push({
      test: /\.wasm$/,
      type: "asset/resource",
      generator: { emit: false },
    });
    return config;
  },
};

export default nextConfig;
