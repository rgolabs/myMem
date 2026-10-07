export type Metadata = Record<string, unknown>;

export type DistanceMetric = 'cosine' | 'euclidean' | 'dotProduct' | 'manhattan';

export interface MemRecord {
  id: string;
  vector: Float32Array;
  metadata: Metadata;
}

/** Identity of the embedding space a collection was built in. Vectors from a different space are refused. */
export interface EmbeddingSpaceIdentity {
  embedderKind: string; // 'onnx' | 'api' | 'ngram' | 'external'
  modelId: string;
  dimension: number;
  normalize: boolean;
  prefixPolicy: string; // e.g. 'none', 'e5', 'bge-query'
  promptTemplateHash: string;
}

export function sameSpace(a: EmbeddingSpaceIdentity, b: EmbeddingSpaceIdentity): boolean {
  return (
    a.embedderKind === b.embedderKind &&
    a.modelId === b.modelId &&
    a.dimension === b.dimension &&
    a.normalize === b.normalize &&
    a.prefixPolicy === b.prefixPolicy &&
    a.promptTemplateHash === b.promptTemplateHash
  );
}

export interface IndexOptions {
  kind: 'flat' | 'hnsw';
  m: number;
  efConstruction: number;
  efSearch: number;
  maxElements: number;
}

export type CompactionPolicyName = 'lru' | 'lfu' | 'coherence';

export interface CollectionOptions {
  dimensions: number;
  distanceMetric: DistanceMetric;
  index: IndexOptions;
  /** Reserved: 'none' is the only accepted value; anything else is rejected at creation (spec §10). */
  quantization: 'none';
  /** Maximum live records before compaction runs; 0 disables. */
  capacity: number;
  compactionPolicy: CompactionPolicyName;
  /** Metadata key holding source text for lexical (BM25) search and re-embedding. */
  textKey: string;
  /** Metadata key holding a millisecond timestamp for temporal decay. */
  timestampKey: string;
  embedding?: EmbeddingSpaceIdentity;
}

export const DEFAULT_INDEX: IndexOptions = {
  kind: 'hnsw',
  m: 16,
  efConstruction: 200,
  efSearch: 100,
  maxElements: 1_000_000,
};

export const LIMITS = {
  maxDimension: 4096,
  maxIdBytes: 512,
  maxMetadataBytes: 64 * 1024,
  maxBatch: 10_000,
  maxK: 1000,
  maxQueryChars: 32_768,
  overfetchCeiling: 4096,
};

/** Structured filter: plain value means equality; object form supports operators (spec §16 enhancement 8). */
export type FilterOp = {
  $eq?: unknown;
  $ne?: unknown;
  $in?: unknown[];
  $nin?: unknown[];
  $gt?: number | string;
  $gte?: number | string;
  $lt?: number | string;
  $lte?: number | string;
  $prefix?: string;
  $exists?: boolean;
  /** For array-valued metadata: true when the array contains the value. */
  $contains?: unknown;
  /** For array-valued metadata: true when the array contains every value. */
  $all?: unknown[];
};
export type Filter = Record<string, unknown | FilterOp>;

export interface DecayOptions {
  /** Per-millisecond decay constant; similarity is multiplied by exp(-lambda * ageMs). */
  lambda?: number;
  /** Convenience: half-life in days (overrides lambda). */
  halfLifeDays?: number;
  timestampKey?: string;
  now?: number;
}

export interface FusionOptions {
  /** Lexical query text. */
  lexical: string;
  method?: 'rrf' | 'rsf';
  /** Weight of the dense ranking in [0,1] for rsf; rrf uses it as a weight too. Default 0.5. */
  alpha?: number;
  /** RRF rank constant. Default 60. */
  rrfK?: number;
}

export interface SearchQuery {
  vector: Float32Array | number[];
  k: number;
  filter?: Filter;
  efSearch?: number;
  includeVectors?: boolean;
  includeMetadata?: boolean;
  /** Held capability mask. A record whose `_caps` metadata has bits not in this mask is skipped inside traversal. */
  capabilities?: bigint | number;
  decay?: DecayOptions;
  fusion?: FusionOptions;
  /** Maximal marginal relevance diversity in [0,1]; 0 disables. 0.3 is a sensible default when enabled. */
  diversity?: number;
  /** Boost records with many close neighbours in the candidate set (coherence gating). */
  coherence?: { weight: number; threshold?: number };
  explain?: boolean;
  /** Optional embedding identity of the query; refused if it differs from the collection's. */
  identity?: EmbeddingSpaceIdentity;
}

export interface Explain {
  distance: number;
  similarity: number;
  decay?: number;
  fusion?: { dense: number; lexical: number; method: string; denseRank?: number; lexicalRank?: number };
  coherence?: number;
  mmr?: number;
}

export interface SearchResult {
  id: string;
  /** Distance under the collection metric; lower is closer. */
  score: number;
  /** Derived similarity; higher is better. For cosine this is 1 - score. */
  similarity: number;
  /** Present when re-ranking (decay, fusion, coherence, diversity) was applied: the value results were ordered by. */
  finalScore?: number;
  vector?: Float32Array;
  metadata?: Metadata;
  explain?: Explain;
}

export interface SearchResponse {
  results: SearchResult[];
  /** Candidates the index produced before filtering/re-ranking. */
  fetched: number;
  /** Candidates that matched the filter. */
  matched: number;
  /** True when `matched >= k` or the collection was exhausted. */
  complete: boolean;
  strategy: 'exact' | 'exact-filtered' | 'hnsw' | 'hnsw-predicate';
  tookMs: number;
}
