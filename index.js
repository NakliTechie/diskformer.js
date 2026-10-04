// diskformer — a disk tier for in-browser inference. Keep model weights in OPFS and page into WebGPU only
// what each step needs. Plain ESM, no dependencies, no build step.
export { OpfsReaderPool, OpfsWriter, canInline, opfsDir, readOpfsText, writeOpfsText, removeOpfs } from './src/opfs-reader.js';
export { ExpertStreamer, ExpertStreamer as RecordPool } from './src/expert-stream.js';
export { RowFile, RowCache, fingerprint } from './src/rows.js';
export { parseGguf, tensorBytes, splitQ8, splitQ4, GGML, Q8_BLOCK, Q4_BLOCK, Q6K_BLOCK } from './src/gguf.js';
export { ingestGguf, ingestProgress, readHeader, storeKey, removeStore } from './src/ingest.js';
export { measure } from './src/measure.js';
