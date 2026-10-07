import type { EmbeddingSpaceIdentity } from '../core/types.js';

export type EmbedRole = 'query' | 'passage';

export interface EmbeddingProvider {
  /** e.g. "onnx:all-MiniLM-L6-v2", "ngram-256" */
  readonly name: string;
  readonly dimensions: number;
  /** false for hash / n-gram fallbacks */
  readonly semantic: boolean;
  /** Loads (and, if allowed, downloads) the model. Idempotent. Never called implicitly during a query. */
  init(): Promise<void>;
  isReady(): boolean;
  embedQuery(text: string): Promise<Float32Array>;
  embedPassage(text: string): Promise<Float32Array>;
  embedBatch(texts: string[], role: EmbedRole): Promise<Float32Array[]>;
  identity(): EmbeddingSpaceIdentity;
}
