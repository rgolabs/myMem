import { MemError } from '../core/errors.js';
import { NgramEmbedder } from './ngram.js';
import { DEFAULT_MODEL, MODELS, TransformersEmbedder } from './transformers.js';
import type { EmbeddingProvider } from './provider.js';

export type { EmbeddingProvider, EmbedRole } from './provider.js';
export { NgramEmbedder } from './ngram.js';
export { TransformersEmbedder, MODELS, DEFAULT_MODEL, defaultModelCacheDir } from './transformers.js';

export interface ResolvedEmbedder {
  provider: EmbeddingProvider;
  /** True when the semantic model failed and the lexical fallback was used (opt-in only). */
  fallbackActivated: boolean;
  requested: string;
  warning?: string;
}

/**
 * Parse an embedder spec: "onnx:<model>[@dtype]" | "ngram[:dim]". Default from MEM_EMBEDDER or MiniLM.
 */
export function createEmbedder(spec?: string): EmbeddingProvider {
  const s = spec || process.env.MEM_EMBEDDER || `onnx:${DEFAULT_MODEL}`;
  if (s.startsWith('ngram')) {
    const dim = s.includes(':') ? parseInt(s.split(':')[1], 10) : 256;
    if (!Number.isInteger(dim) || dim < 16) throw new MemError('INVALID_ARGUMENT', `bad ngram dimension in ${s}`);
    return new NgramEmbedder(dim);
  }
  if (s.startsWith('onnx:')) {
    const rest = s.slice(5);
    const [model, dtype] = rest.split('@');
    return new TransformersEmbedder(model || DEFAULT_MODEL, dtype ? { dtype } : {});
  }
  if (MODELS[s]) return new TransformersEmbedder(s);
  throw new MemError('INVALID_ARGUMENT', `unknown embedder spec ${s}; use onnx:<model> or ngram[:dim]`);
}

/**
 * Initialise the requested embedder. If it fails and `allowFallback` (or MEM_ALLOW_FALLBACK=1) is set,
 * degrade to the n-gram embedder and say so; otherwise throw EMBEDDER_UNAVAILABLE.
 */
export async function resolveEmbedder(opts: { spec?: string; allowFallback?: boolean; log?: (msg: string) => void } = {}): Promise<ResolvedEmbedder> {
  const requested = opts.spec || process.env.MEM_EMBEDDER || `onnx:${DEFAULT_MODEL}`;
  const allowFallback = opts.allowFallback ?? process.env.MEM_ALLOW_FALLBACK === '1';
  const provider = createEmbedder(requested);
  try {
    await provider.init();
    return { provider, fallbackActivated: false, requested };
  } catch (e: any) {
    if (!allowFallback || !provider.semantic) throw e;
    const warning = `semantic embedder ${requested} unavailable (${e?.message ?? e}); falling back to lexical n-gram embedder. Results are NOT semantic.`;
    opts.log?.(warning);
    const fb = new NgramEmbedder(256);
    await fb.init();
    return { provider: fb, fallbackActivated: true, requested, warning };
  }
}
