/**
 * Local sentence-embedding models through transformers.js (ONNX Runtime). The default is a
 * MiniLM-class 384-d model (~23 MB quantised), cached once per user. Models are pinned by id;
 * the engine never fetches a model during a query (only in `init()`), and `MEM_OFFLINE=1`
 * disables downloads entirely.
 */
import os from 'node:os';
import path from 'node:path';
import { MemError } from '../core/errors.js';
import { sha256 } from '../core/witness.js';
import type { EmbeddingSpaceIdentity } from '../core/types.js';
import type { EmbeddingProvider, EmbedRole } from './provider.js';

export interface ModelSpec {
  hf: string;
  dimensions: number;
  pooling: 'mean' | 'cls';
  queryPrefix: string;
  passagePrefix: string;
  prefixPolicy: string;
}

export const MODELS: Record<string, ModelSpec> = {
  'all-MiniLM-L6-v2': { hf: 'Xenova/all-MiniLM-L6-v2', dimensions: 384, pooling: 'mean', queryPrefix: '', passagePrefix: '', prefixPolicy: 'none' },
  'all-MiniLM-L12-v2': { hf: 'Xenova/all-MiniLM-L12-v2', dimensions: 384, pooling: 'mean', queryPrefix: '', passagePrefix: '', prefixPolicy: 'none' },
  'bge-small-en-v1.5': {
    hf: 'Xenova/bge-small-en-v1.5',
    dimensions: 384,
    pooling: 'cls',
    queryPrefix: 'Represent this sentence for searching relevant passages: ',
    passagePrefix: '',
    prefixPolicy: 'bge-query',
  },
  'e5-small-v2': { hf: 'Xenova/e5-small-v2', dimensions: 384, pooling: 'mean', queryPrefix: 'query: ', passagePrefix: 'passage: ', prefixPolicy: 'e5' },
  'multilingual-e5-small': { hf: 'Xenova/multilingual-e5-small', dimensions: 384, pooling: 'mean', queryPrefix: 'query: ', passagePrefix: 'passage: ', prefixPolicy: 'e5' },
  'gte-small': { hf: 'Xenova/gte-small', dimensions: 384, pooling: 'mean', queryPrefix: '', passagePrefix: '', prefixPolicy: 'none' },
  'bge-base-en-v1.5': {
    hf: 'Xenova/bge-base-en-v1.5',
    dimensions: 768,
    pooling: 'cls',
    queryPrefix: 'Represent this sentence for searching relevant passages: ',
    passagePrefix: '',
    prefixPolicy: 'bge-query',
  },
};

export const DEFAULT_MODEL = 'all-MiniLM-L6-v2';

export function defaultModelCacheDir(): string {
  return process.env.MEM_MODEL_CACHE || path.join(os.homedir(), '.cache', 'agent-memory', 'models');
}

type Pipe = (texts: string[], opts: { pooling: 'mean' | 'cls'; normalize: boolean }) => Promise<{ data: Float32Array; dims: number[] }>;

export class TransformersEmbedder implements EmbeddingProvider {
  readonly name: string;
  readonly dimensions: number;
  readonly semantic = true;
  readonly spec: ModelSpec;
  private pipe: Pipe | null = null;
  private initPromise: Promise<void> | null = null;
  readonly dtype: string;
  readonly offline: boolean;

  constructor(readonly modelId: string = DEFAULT_MODEL, opts: { dtype?: string; offline?: boolean } = {}) {
    const spec = MODELS[modelId];
    if (!spec) throw new MemError('INVALID_ARGUMENT', `unknown model ${modelId}; known: ${Object.keys(MODELS).join(', ')}`);
    this.spec = spec;
    this.dimensions = spec.dimensions;
    this.dtype = opts.dtype ?? process.env.MEM_EMBEDDER_DTYPE ?? 'q8';
    this.offline = opts.offline ?? process.env.MEM_OFFLINE === '1';
    this.name = `onnx:${modelId}@${this.dtype}`;
  }

  isReady() {
    return this.pipe !== null;
  }

  init(): Promise<void> {
    if (this.pipe) return Promise.resolve();
    if (!this.initPromise) {
      this.initPromise = (async () => {
        let tf: typeof import('@huggingface/transformers');
        try {
          tf = await import('@huggingface/transformers');
        } catch (e: any) {
          throw new MemError('EMBEDDER_UNAVAILABLE', `transformers.js could not be loaded: ${e?.message ?? e}`);
        }
        tf.env.cacheDir = defaultModelCacheDir();
        tf.env.allowLocalModels = true;
        tf.env.allowRemoteModels = !this.offline;
        try {
          const p = await tf.pipeline('feature-extraction', this.spec.hf, { dtype: this.dtype as any });
          this.pipe = p as unknown as Pipe;
        } catch (e: any) {
          this.initPromise = null;
          const hint = this.offline ? 'MEM_OFFLINE=1 is set; prepopulate the cache with `mem init` on a connected machine' : 'check network access or set MEM_ALLOW_FALLBACK=1 to use the lexical fallback';
          throw new MemError('EMBEDDER_UNAVAILABLE', `could not load ${this.spec.hf} (${this.dtype}): ${e?.message ?? e}. ${hint}`);
        }
      })();
    }
    return this.initPromise;
  }

  private async run(texts: string[], role: EmbedRole): Promise<Float32Array[]> {
    if (!this.pipe) throw new MemError('EMBEDDER_UNAVAILABLE', `embedder ${this.name} is not initialised; call init() first`);
    const prefix = role === 'query' ? this.spec.queryPrefix : this.spec.passagePrefix;
    const out: Float32Array[] = [];
    const BATCH = 32;
    for (let i = 0; i < texts.length; i += BATCH) {
      const chunk = texts.slice(i, i + BATCH).map((t) => prefix + (t || ' '));
      const res = await this.pipe(chunk, { pooling: this.spec.pooling, normalize: true });
      const d = res.dims[res.dims.length - 1];
      for (let j = 0; j < chunk.length; j++) out.push(Float32Array.from(res.data.subarray(j * d, (j + 1) * d)));
    }
    return out;
  }

  async embedQuery(text: string) {
    return (await this.run([text], 'query'))[0];
  }
  async embedPassage(text: string) {
    return (await this.run([text], 'passage'))[0];
  }
  async embedBatch(texts: string[], role: EmbedRole) {
    if (!texts.length) return [];
    return this.run(texts, role);
  }

  identity(): EmbeddingSpaceIdentity {
    return {
      embedderKind: 'onnx',
      modelId: `${this.modelId}@${this.dtype}`,
      dimension: this.dimensions,
      normalize: true,
      prefixPolicy: this.spec.prefixPolicy,
      promptTemplateHash: sha256(`${this.spec.queryPrefix}\u0000${this.spec.passagePrefix}\u0000${this.spec.pooling}`).slice(0, 16),
    };
  }
}
