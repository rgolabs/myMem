/**
 * Collection: a durable, file-backed set of records with one dimension, one metric and one
 * embedding space (spec §4). Layout of a collection directory:
 *
 *   manifest.json     options, embedding identity, schema version, feature matrix
 *   checkpoint.json   ids, metadata, serialized HNSW, access stats, log generation at checkpoint
 *   vectors.bin       Float32 vectors by slot, written with the checkpoint
 *   log.jsonl         write-ahead log; one line per transaction (a line is all-or-nothing)
 *   .lock/            advisory cross-process lock
 *
 * Open = load checkpoint (with its persisted index) + replay the log tail. The index is only
 * rebuilt when the checkpoint is missing, corrupt, or has too many tombstones.
 *
 * Several processes may open the same directory: writers serialise through the lock and every
 * operation first applies any log lines written by other processes.
 */
import fs from 'node:fs';
import path from 'node:path';
import { MemError, invalid } from './errors.js';
import {
  type CollectionOptions,
  type DistanceMetric,
  type EmbeddingSpaceIdentity,
  type Filter,
  type MemRecord,
  type Metadata,
  type SearchQuery,
  type SearchResponse,
  type SearchResult,
  type Explain,
  DEFAULT_INDEX,
  LIMITS,
  sameSpace,
} from './types.js';
import { base64ToF32, distanceFor, f32ToBase64, normalize, normalizedCosineDistance, similarityFor, toFloat32, type DistanceFn } from './distance.js';
import { HNSW, type HnswSerialized } from './hnsw.js';
import { Bm25Index, rrf, rsf } from './lexical.js';
import { equalityTerms, indexKey, matchesFilter, validateFilter } from './filter.js';
import { appendLineSync, fileSize, fileStamp, readFrom, readJsonIfExists, withLock, writeFileAtomic } from './fsutil.js';
import type { WitnessLog } from './witness.js';

export const SCHEMA_VERSION = 1;

export interface Manifest {
  schemaVersion: number;
  createdAt: string;
  options: CollectionOptions;
  fallbackEmbedder?: boolean;
  features: string[];
}

interface AccessStat {
  l: number; // last access ms
  c: number; // access count
  f: number; // feedback score (explicit outcomes only)
}

interface Checkpoint {
  schemaVersion: number;
  generation: number;
  dimensions: number;
  ids: (string | null)[];
  metas: (Metadata | null)[];
  modGen: number[];
  index: HnswSerialized | null;
  access: Record<string, AccessStat>;
  recentQueries: string[];
}

type LogEntry =
  | { g: number; op: 'insert' | 'upsert'; recs: Array<{ id: string; v: string; m: Metadata }> }
  | { g: number; op: 'delete'; ids: string[]; reason?: string };

export interface CompactionContext {
  now: number;
  recentQueries: Float32Array[];
  access: (id: string) => AccessStat | undefined;
  similarity: (a: Float32Array, b: Float32Array) => number;
}
export type CompactionPolicy = (record: MemRecord, ctx: CompactionContext) => number;

export interface CompactResult {
  before: number;
  after: number;
  evicted: string[];
  policy: string;
  tookMs: number;
}

export interface CollectionStats {
  path: string;
  count: number;
  slots: number;
  tombstones: number;
  generation: number;
  checkpointGeneration: number;
  logBytes: number;
  options: CollectionOptions;
  embedding?: EmbeddingSpaceIdentity;
  fallbackEmbedder: boolean;
}

export const FEATURES = [
  'index:flat',
  'index:hnsw',
  'index:persisted',
  'filter:structured',
  'filter:predicate-traversal',
  'filter:overfetch',
  'capabilities:mask64',
  'retrieval:hybrid-bm25',
  'retrieval:temporal-decay',
  'retrieval:coherence',
  'retrieval:mmr',
  'retrieval:explain',
  'lifecycle:compaction',
  'lifecycle:snapshot',
  'lifecycle:branch',
  'audit:witness',
  'concurrency:multi-process-log',
];

const CHECKPOINT_EVERY_OPS = 500;
const CHECKPOINT_LOG_BYTES = 64 * 1024 * 1024;
const REBUILD_TOMBSTONE_RATIO = 0.3;
const RECENT_QUERIES = 32;
const EXACT_THRESHOLD = 1000;

export class Collection {
  readonly dir: string;
  readonly options: CollectionOptions;
  readonly dim: number;
  readonly metric: DistanceMetric;
  readonly distance: DistanceFn;
  manifest: Manifest;

  private ids: (string | null)[] = [];
  private metas: (Metadata | null)[] = [];
  private vectors: Float32Array[] = [];
  private modGen: number[] = [];
  private idToSlot = new Map<string, number>();
  private metaIndex = new Map<string, Map<string, Set<number>>>();
  private index: HNSW | null = null;
  private lexical = new Bm25Index();
  private access = new Map<string, AccessStat>();
  private recentQueries: Float32Array[] = [];
  private generation = 0;
  private checkpointGeneration = 0;
  private logOffset = 0;
  private liveCount = 0;
  private closed = false;
  private readOnly = false;
  private witness: WitnessLog | null;
  private actor: string;
  private customPolicy: CompactionPolicy | null = null;
  private logFile: string;
  private cpFile: string;
  private cpStamp = 'none';

  private constructor(dir: string, manifest: Manifest, witness: WitnessLog | null, actor: string) {
    this.dir = dir;
    this.manifest = manifest;
    this.options = manifest.options;
    this.dim = manifest.options.dimensions;
    this.metric = manifest.options.distanceMetric;
    // cosine collections store unit vectors, so the distance is a plain dot product
    this.distance = this.metric === 'cosine' ? normalizedCosineDistance : distanceFor(this.metric);
    this.witness = witness;
    this.actor = actor;
    this.logFile = path.join(dir, 'log.jsonl');
    this.cpFile = path.join(dir, 'checkpoint.json');
  }

  // ------------------------------------------------------------------ open / create

  static exists(dir: string): boolean {
    return fs.existsSync(path.join(dir, 'manifest.json'));
  }

  static create(
    dir: string,
    opts: Partial<Omit<CollectionOptions, 'index'>> & { dimensions: number; index?: Partial<CollectionOptions['index']> },
    extra: { witness?: WitnessLog; actor?: string; fallbackEmbedder?: boolean } = {},
  ): Collection {
    if (Collection.exists(dir)) throw new MemError('ALREADY_EXISTS', `collection already exists at ${dir}`);
    if (!Number.isInteger(opts.dimensions) || opts.dimensions <= 0 || opts.dimensions > LIMITS.maxDimension) {
      invalid(`dimensions must be an integer in [1, ${LIMITS.maxDimension}]`, { dimensions: opts.dimensions });
    }
    if (opts.quantization && opts.quantization !== 'none') {
      throw new MemError('UNSUPPORTED', `quantization '${opts.quantization}' is not implemented; only 'none' is accepted (it is rejected, never stored silently)`);
    }
    if (opts.embedding && opts.embedding.dimension !== opts.dimensions) {
      throw new MemError('DIMENSION_MISMATCH', `embedding identity dimension ${opts.embedding.dimension} != collection dimensions ${opts.dimensions}`);
    }
    const options: CollectionOptions = {
      dimensions: opts.dimensions,
      distanceMetric: opts.distanceMetric ?? 'cosine',
      index: { ...DEFAULT_INDEX, ...(opts.index ?? {}) },
      quantization: 'none',
      capacity: opts.capacity ?? 0,
      compactionPolicy: opts.compactionPolicy ?? 'coherence',
      textKey: opts.textKey ?? 'text',
      timestampKey: opts.timestampKey ?? 'createdAtMs',
      embedding: opts.embedding,
    };
    fs.mkdirSync(dir, { recursive: true });
    const manifest: Manifest = {
      schemaVersion: SCHEMA_VERSION,
      createdAt: new Date().toISOString(),
      options,
      fallbackEmbedder: extra.fallbackEmbedder ?? false,
      features: FEATURES,
    };
    writeFileAtomic(path.join(dir, 'manifest.json'), JSON.stringify(manifest, null, 2));
    const c = new Collection(dir, manifest, extra.witness ?? null, extra.actor ?? 'local');
    c.initIndex();
    return c;
  }

  static open(dir: string, extra: { witness?: WitnessLog; actor?: string; readOnly?: boolean } = {}): Collection {
    const manifest = readJsonIfExists<Manifest>(path.join(dir, 'manifest.json'));
    if (!manifest) throw new MemError('NOT_FOUND', `no collection at ${dir}`);
    if (manifest.schemaVersion > SCHEMA_VERSION) {
      throw new MemError('VERSION', `collection schema ${manifest.schemaVersion} is newer than this engine (${SCHEMA_VERSION}); upgrade the engine`);
    }
    const c = new Collection(dir, manifest, extra.witness ?? null, extra.actor ?? 'local');
    c.readOnly = !!extra.readOnly;
    c.load();
    return c;
  }

  static openOrCreate(
    dir: string,
    opts: Partial<Omit<CollectionOptions, 'index'>> & { dimensions: number; index?: Partial<CollectionOptions['index']> },
    extra: { witness?: WitnessLog; actor?: string; fallbackEmbedder?: boolean } = {},
  ): Collection {
    return Collection.exists(dir) ? Collection.open(dir, extra) : Collection.create(dir, opts, extra);
  }

  private initIndex() {
    const io = this.options.index;
    this.index = io.kind === 'hnsw' ? new HNSW({ m: io.m, efConstruction: io.efConstruction }, (i) => this.vectors[i], this.distance) : null;
  }

  private load() {
    this.cpStamp = fileStamp(this.cpFile);
    const cp = readJsonIfExists<Checkpoint>(this.cpFile);
    this.ids = [];
    this.metas = [];
    this.vectors = [];
    this.modGen = [];
    this.idToSlot.clear();
    this.metaIndex.clear();
    this.lexical.clear();
    this.access.clear();
    this.liveCount = 0;
    this.generation = 0;
    this.checkpointGeneration = 0;
    this.logOffset = 0;
    let indexOk = false;
    if (cp) {
      if (cp.dimensions !== this.dim) throw new MemError('CORRUPT', `checkpoint dimension ${cp.dimensions} != manifest ${this.dim}`);
      const vecFile = path.join(this.dir, 'vectors.bin');
      const buf = fs.existsSync(vecFile) ? fs.readFileSync(vecFile) : Buffer.alloc(0);
      const expected = cp.ids.length * this.dim * 4;
      if (buf.byteLength < expected) throw new MemError('CORRUPT', `vectors.bin is ${buf.byteLength} bytes, expected at least ${expected}`);
      const all = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + expected));
      for (let slot = 0; slot < cp.ids.length; slot++) {
        const id = cp.ids[slot];
        this.ids.push(id);
        this.metas.push(cp.metas[slot]);
        this.vectors.push(all.subarray(slot * this.dim, (slot + 1) * this.dim));
        this.modGen.push(cp.modGen?.[slot] ?? 0);
        if (id !== null) {
          this.idToSlot.set(id, slot);
          this.liveCount++;
          this.indexMeta(slot, cp.metas[slot]!);
        }
      }
      this.generation = cp.generation;
      this.checkpointGeneration = cp.generation;
      this.access = new Map(Object.entries(cp.access ?? {}));
      this.recentQueries = (cp.recentQueries ?? []).map(base64ToF32);
      if (this.options.index.kind === 'hnsw' && cp.index && cp.index.levels.length === cp.ids.length) {
        this.index = HNSW.deserialize(cp.index, (i) => this.vectors[i], this.distance);
        indexOk = true;
      }
    }
    if (!indexOk) {
      this.initIndex();
      if (this.index) for (let slot = 0; slot < this.ids.length; slot++) if (this.ids[slot] !== null) this.index.insert(slot);
    }
    // skip log lines already covered by the checkpoint, apply the rest
    this.applyLogTail();
  }

  /** Under the write lock: drop a torn trailing line left by a crashed writer so new lines start clean. */
  private truncateTornTail(): void {
    this.applyLogTail();
    const size = fileSize(this.logFile);
    if (size > this.logOffset) fs.truncateSync(this.logFile, this.logOffset);
  }

  /** Apply log lines written after `logOffset` (by this or any other process). */
  private applyLogTail(): void {
    const size = fileSize(this.logFile);
    if (size < this.logOffset || fileStamp(this.cpFile) !== this.cpStamp) {
      // another process wrote a checkpoint (and truncated the log): reload everything
      this.load();
      return;
    }
    if (size === this.logOffset) return;
    const buf = readFrom(this.logFile, this.logOffset);
    let start = 0;
    for (;;) {
      const nl = buf.indexOf(10, start);
      if (nl < 0) break;
      const line = buf.subarray(start, nl).toString('utf8');
      start = nl + 1;
      if (!line.trim()) continue;
      let entry: LogEntry;
      try {
        entry = JSON.parse(line);
      } catch {
        throw new MemError('CORRUPT', `unparseable log line at byte ${this.logOffset + start} of ${this.logFile}`);
      }
      if (entry.g <= this.generation) continue; // already in checkpoint
      if (entry.g !== this.generation + 1) {
        throw new MemError('CORRUPT', `log generation gap: have ${this.generation}, next line is ${entry.g}`);
      }
      this.applyEntry(entry);
    }
    this.logOffset += start; // a trailing partial line (torn write) stays unconsumed
  }

  private applyEntry(e: LogEntry) {
    if (e.op === 'delete') {
      for (const id of e.ids) this.applyDelete(id, e.g);
    } else {
      for (const r of e.recs) this.applyPut(r.id, base64ToF32(r.v), r.m, e.g);
    }
    this.generation = e.g;
  }

  private applyPut(id: string, vec: Float32Array, meta: Metadata, g: number) {
    const old = this.idToSlot.get(id);
    if (old !== undefined) this.applyDelete(id, g);
    const slot = this.ids.length;
    this.ids.push(id);
    this.metas.push(meta);
    this.vectors.push(vec);
    this.modGen.push(g);
    this.idToSlot.set(id, slot);
    this.liveCount++;
    this.indexMeta(slot, meta);
    if (this.index) this.index.insert(slot);
  }

  private applyDelete(id: string, g: number) {
    const slot = this.idToSlot.get(id);
    if (slot === undefined) return;
    this.unindexMeta(slot, this.metas[slot]!);
    this.ids[slot] = null;
    this.metas[slot] = null;
    this.modGen[slot] = g;
    this.idToSlot.delete(id);
    this.liveCount--;
    if (this.index) this.index.markDeleted(slot);
    this.access.delete(id);
  }

  private indexMeta(slot: number, meta: Metadata) {
    for (const [k, v] of Object.entries(meta)) {
      const vals = Array.isArray(v) ? v : [v];
      for (const x of vals) {
        const ik = indexKey(x);
        if (ik === null) continue;
        let m = this.metaIndex.get(k);
        if (!m) this.metaIndex.set(k, (m = new Map()));
        let s = m.get(ik);
        if (!s) m.set(ik, (s = new Set()));
        s.add(slot);
      }
    }
    const text = meta[this.options.textKey];
    if (typeof text === 'string' && text) this.lexical.add(slot, text);
  }

  private unindexMeta(slot: number, meta: Metadata) {
    for (const [k, v] of Object.entries(meta)) {
      const m = this.metaIndex.get(k);
      if (!m) continue;
      const vals = Array.isArray(v) ? v : [v];
      for (const x of vals) {
        const ik = indexKey(x);
        if (ik === null) continue;
        const s = m.get(ik);
        if (s) {
          s.delete(slot);
          if (!s.size) m.delete(ik);
        }
      }
    }
    this.lexical.remove(slot);
  }

  // ------------------------------------------------------------------ validation

  private assertOpen() {
    if (this.closed) throw new MemError('INVALID_ARGUMENT', 'collection is closed');
  }
  private assertWritable() {
    this.assertOpen();
    if (this.readOnly) throw new MemError('READ_ONLY', `collection at ${this.dir} is open read-only`);
  }

  assertSpace(identity?: EmbeddingSpaceIdentity) {
    if (!identity) return;
    const mine = this.options.embedding;
    if (!mine) {
      throw new MemError('EMBEDDING_SPACE_MISMATCH', `collection at ${this.dir} has no recorded embedding space (legacy); re-embed it before writing with ${identity.modelId}`);
    }
    if (!sameSpace(mine, identity)) {
      throw new MemError('EMBEDDING_SPACE_MISMATCH', `collection embedding space ${describe(mine)} differs from caller ${describe(identity)}`, {
        collection: mine,
        caller: identity,
      });
    }
  }

  private validateRecord(r: { id: string; vector: Float32Array | number[]; metadata?: Metadata }): MemRecord {
    if (typeof r.id !== 'string' || !r.id) invalid('record id must be a non-empty string');
    if (Buffer.byteLength(r.id) > LIMITS.maxIdBytes) throw new MemError('LIMIT_EXCEEDED', `id exceeds ${LIMITS.maxIdBytes} bytes`);
    let vector = toFloat32(r.vector);
    if (vector.length !== this.dim) throw new MemError('DIMENSION_MISMATCH', `vector has ${vector.length} dimensions, collection has ${this.dim}`);
    for (let i = 0; i < vector.length; i++) if (!Number.isFinite(vector[i])) invalid('vector contains a non-finite value', { id: r.id, index: i });
    if (this.metric === 'cosine') vector = normalize(vector);
    const metadata = r.metadata ?? {};
    if (typeof metadata !== 'object' || Array.isArray(metadata)) invalid('metadata must be an object');
    const bytes = Buffer.byteLength(JSON.stringify(metadata));
    if (bytes > LIMITS.maxMetadataBytes) throw new MemError('LIMIT_EXCEEDED', `metadata is ${bytes} bytes; limit ${LIMITS.maxMetadataBytes}`, { id: r.id });
    return { id: r.id, vector, metadata };
  }

  // ------------------------------------------------------------------ writes

  insert(r: { id: string; vector: Float32Array | number[]; metadata?: Metadata }, identity?: EmbeddingSpaceIdentity): string {
    return this.insertBatch([r], identity)[0];
  }

  insertBatch(records: Array<{ id: string; vector: Float32Array | number[]; metadata?: Metadata }>, identity?: EmbeddingSpaceIdentity): string[] {
    return this.put(records, 'insert', identity);
  }

  upsert(r: { id: string; vector: Float32Array | number[]; metadata?: Metadata }, identity?: EmbeddingSpaceIdentity): string {
    return this.put([r], 'upsert', identity)[0];
  }

  upsertBatch(records: Array<{ id: string; vector: Float32Array | number[]; metadata?: Metadata }>, identity?: EmbeddingSpaceIdentity): string[] {
    return this.put(records, 'upsert', identity);
  }

  /** Replace vector and metadata of an existing record. */
  update(id: string, r: { vector: Float32Array | number[]; metadata?: Metadata }, identity?: EmbeddingSpaceIdentity): void {
    this.assertWritable();
    this.sync();
    if (!this.idToSlot.has(id)) throw new MemError('NOT_FOUND', `record ${id} not found`);
    this.put([{ id, ...r }], 'upsert', identity);
  }

  /** Patch metadata only (vector unchanged). */
  updateMetadata(id: string, patch: Metadata): Metadata {
    this.assertWritable();
    this.sync();
    const slot = this.idToSlot.get(id);
    if (slot === undefined) throw new MemError('NOT_FOUND', `record ${id} not found`);
    const merged = { ...this.metas[slot]!, ...patch };
    for (const k of Object.keys(patch)) if (patch[k] === undefined || patch[k] === null) delete merged[k];
    this.put([{ id, vector: this.vectors[slot], metadata: merged }], 'upsert');
    return merged;
  }

  private put(records: Array<{ id: string; vector: Float32Array | number[]; metadata?: Metadata }>, op: 'insert' | 'upsert', identity?: EmbeddingSpaceIdentity): string[] {
    this.assertWritable();
    this.assertSpace(identity);
    if (!Array.isArray(records) || records.length === 0) invalid('records must be a non-empty array');
    if (records.length > LIMITS.maxBatch) throw new MemError('LIMIT_EXCEEDED', `batch of ${records.length} exceeds ${LIMITS.maxBatch}`);
    const recs = records.map((r) => this.validateRecord(r));
    const seen = new Set<string>();
    for (const r of recs) {
      if (seen.has(r.id)) invalid(`duplicate id ${r.id} in batch`);
      seen.add(r.id);
    }
    const ids = withLock(this.dir, () => {
      this.applyLogTail();
      if (op === 'insert') {
        for (const r of recs) if (this.idToSlot.has(r.id)) throw new MemError('ALREADY_EXISTS', `record ${r.id} already exists (use update/upsert)`);
      }
      const entry: LogEntry = {
        g: this.generation + 1,
        op,
        recs: recs.map((r) => ({ id: r.id, v: f32ToBase64(r.vector), m: r.metadata })),
      };
      this.appendLog(entry);
      return recs.map((r) => r.id);
    });
    if (this.witness) this.witness.appendMany(recs.map((r) => ({ operation: op, recordId: r.id, payload: { metadata: r.metadata } })), this.actor);
    this.maybeCheckpoint();
    this.maybeCompact();
    return ids;
  }

  private appendLog(entry: LogEntry) {
    this.truncateTornTail();
    const line = JSON.stringify(entry) + '\n';
    appendLineSync(this.logFile, line);
    this.applyEntry(entry);
    this.logOffset += Buffer.byteLength(line);
  }

  delete(id: string, reason = 'delete'): boolean {
    return this.deleteBatch([id], reason) === 1;
  }

  deleteBatch(ids: string[], reason = 'delete'): number {
    this.assertWritable();
    const removed = withLock(this.dir, () => {
      this.applyLogTail();
      const present = ids.filter((id) => this.idToSlot.has(id));
      if (!present.length) return [];
      this.appendLog({ g: this.generation + 1, op: 'delete', ids: present, reason });
      return present;
    });
    if (this.witness) this.witness.appendMany(removed.map((id) => ({ operation: reason, recordId: id })), this.actor);
    this.maybeCheckpoint();
    return removed.length;
  }

  // ------------------------------------------------------------------ reads

  /** Pick up writes from other processes. Cheap when nothing changed (one stat). */
  sync(): void {
    this.assertOpen();
    this.applyLogTail();
  }

  get(id: string, opts: { touch?: boolean } = {}): MemRecord | null {
    this.sync();
    const slot = this.idToSlot.get(id);
    if (slot === undefined) return null;
    if (opts.touch) this.touch(id);
    return { id, vector: this.vectors[slot], metadata: this.metas[slot]! };
  }

  has(id: string): boolean {
    this.sync();
    return this.idToSlot.has(id);
  }

  count(): number {
    this.sync();
    return this.liveCount;
  }

  /** Modification generation of a record (0 if unknown); used by branches to detect conflicts. */
  modificationGeneration(id: string): number {
    const slot = this.idToSlot.get(id);
    return slot === undefined ? 0 : this.modGen[slot];
  }

  get currentGeneration(): number {
    return this.generation;
  }

  *iterate(filter?: Filter): Generator<MemRecord> {
    validateFilter(filter);
    this.sync();
    for (let slot = 0; slot < this.ids.length; slot++) {
      const id = this.ids[slot];
      if (id === null) continue;
      const meta = this.metas[slot]!;
      if (filter && !matchesFilter(meta, filter)) continue;
      yield { id, vector: this.vectors[slot], metadata: meta };
    }
  }

  list(opts: { filter?: Filter; limit?: number; offset?: number; includeVectors?: boolean } = {}): { records: MemRecord[]; total: number } {
    validateFilter(opts.filter);
    const limit = Math.max(0, Math.min(opts.limit ?? 100, LIMITS.maxBatch));
    const offset = Math.max(0, opts.offset ?? 0);
    const out: MemRecord[] = [];
    let total = 0;
    for (const r of this.iterate(opts.filter)) {
      if (total >= offset && out.length < limit) out.push(opts.includeVectors ? r : { ...r, vector: undefined as unknown as Float32Array });
      total++;
    }
    return { records: out, total };
  }

  private touch(id: string) {
    const now = Date.now();
    const a = this.access.get(id);
    if (a) {
      a.l = now;
      a.c++;
    } else this.access.set(id, { l: now, c: 1, f: 0 });
  }

  /** Explicit feedback write (the only path that changes `f`). Caller logs it to the witness chain. */
  recordFeedback(id: string, reward: number): void {
    const a = this.access.get(id) ?? { l: Date.now(), c: 0, f: 0 };
    a.f += reward;
    this.access.set(id, a);
  }

  accessStats(id: string): AccessStat | undefined {
    return this.access.get(id);
  }

  // ------------------------------------------------------------------ search

  /** Resolve equality terms to a candidate slot set via the metadata index; null when not resolvable. */
  private candidateSet(filter: Filter): Set<number> | null {
    const terms = equalityTerms(filter);
    if (!terms.length) return null;
    let best: Set<number> | null = null;
    for (const t of terms) {
      const m = this.metaIndex.get(t.key);
      const union = new Set<number>();
      if (m) {
        for (const v of t.values) {
          const ik = indexKey(v);
          if (ik === null) return null; // unindexable value type; fall back
          const s = m.get(ik);
          if (s) for (const x of s) union.add(x);
        }
      }
      if (best === null || union.size < best.size) best = union;
      if (best.size === 0) return best;
    }
    return best;
  }

  search(q: SearchQuery): SearchResponse {
    const t0 = performance.now();
    this.sync();
    if (!Number.isInteger(q.k) || q.k < 1) invalid('k must be a positive integer', { k: q.k });
    if (q.k > LIMITS.maxK) throw new MemError('LIMIT_EXCEEDED', `k ${q.k} exceeds ${LIMITS.maxK}`);
    this.assertSpace(q.identity);
    let qv = toFloat32(q.vector);
    if (qv.length !== this.dim) throw new MemError('DIMENSION_MISMATCH', `query has ${qv.length} dimensions, collection has ${this.dim}`);
    if (this.metric === 'cosine') qv = normalize(qv);
    validateFilter(q.filter);
    const caps = q.capabilities === undefined ? null : BigInt(q.capabilities);
    const filter = q.filter && Object.keys(q.filter).length ? q.filter : undefined;

    const allow = (slot: number): boolean => {
      if (this.ids[slot] === null) return false;
      const meta = this.metas[slot]!;
      if (caps !== null) {
        const req = meta._caps;
        if (req !== undefined && req !== null) {
          const reqBits = BigInt(req as number | string | bigint);
          if ((reqBits & caps) !== reqBits) return false;
        }
      } else if (meta._caps !== undefined && meta._caps !== null && BigInt(meta._caps as number | string) !== 0n) {
        return false; // gated record, no capabilities supplied
      }
      return filter ? matchesFilter(meta, filter) : true;
    };
    const hasPredicate = filter !== undefined || caps !== null || this.metaIndex.has('_caps');

    const rerank = !!(q.decay || q.fusion || (q.diversity && q.diversity > 0) || q.coherence);
    const kFetch = rerank ? Math.min(Math.max(q.k * 4, 20), LIMITS.overfetchCeiling) : q.k;

    let candidates: Array<{ slot: number; distance: number }> = [];
    let strategy: SearchResponse['strategy'];
    let fetched = 0;
    let exhausted = false;

    const exactOver = (slots: Iterable<number>) => {
      const out: Array<{ slot: number; distance: number }> = [];
      for (const slot of slots) {
        if (!allow(slot)) continue;
        out.push({ slot, distance: this.distance(qv, this.vectors[slot]) });
      }
      out.sort((a, b) => a.distance - b.distance);
      return out;
    };

    const liveSlots = function* (ids: (string | null)[]) {
      for (let s = 0; s < ids.length; s++) if (ids[s] !== null) yield s;
    };

    if (!this.index || this.liveCount <= EXACT_THRESHOLD) {
      strategy = 'exact';
      fetched = this.liveCount;
      candidates = exactOver(liveSlots(this.ids));
      exhausted = true;
    } else {
      const cand = filter ? this.candidateSet(filter) : null;
      if (cand !== null && cand.size <= Math.max(2000, Math.floor(this.liveCount * 0.1))) {
        strategy = 'exact-filtered';
        fetched = cand.size;
        candidates = exactOver(cand);
        exhausted = true;
      } else if (hasPredicate) {
        strategy = 'hnsw-predicate';
        let kk = kFetch;
        let ef = Math.max(q.efSearch ?? this.options.index.efSearch, kk * 4);
        for (;;) {
          const res = this.index.search(qv, kk, ef, allow);
          candidates = res.map((r) => ({ slot: r.id, distance: r.distance }));
          fetched = ef;
          if (candidates.length >= kFetch || ef >= LIMITS.overfetchCeiling || ef >= this.liveCount) {
            exhausted = ef >= this.liveCount;
            break;
          }
          kk *= 2;
          ef *= 2;
        }
      } else {
        strategy = 'hnsw';
        const ef = Math.max(q.efSearch ?? this.options.index.efSearch, kFetch);
        const res = this.index.search(qv, kFetch, ef);
        candidates = res.map((r) => ({ slot: r.id, distance: r.distance }));
        fetched = ef;
      }
    }
    const matched = candidates.length;
    if (rerank) candidates = candidates.slice(0, kFetch);
    else candidates = candidates.slice(0, q.k);

    // ---- post-processing
    const items: Array<{ slot: number; distance: number; similarity: number; final: number; explain: Explain }> = candidates.map((c) => {
      const sim = similarityFor(this.metric, c.distance);
      return { slot: c.slot, distance: c.distance, similarity: sim, final: sim, explain: { distance: c.distance, similarity: sim } };
    });

    if (q.decay) {
      const now = q.decay.now ?? Date.now();
      const lambda = q.decay.halfLifeDays ? Math.LN2 / (q.decay.halfLifeDays * 86_400_000) : (q.decay.lambda ?? 0);
      const key = q.decay.timestampKey ?? this.options.timestampKey;
      for (const it of items) {
        const ts = toMs(this.metas[it.slot]![key]);
        const factor = ts === null ? 1 : Math.exp(-lambda * Math.max(0, now - ts));
        it.final *= factor;
        it.explain.decay = factor;
      }
    }

    if (q.coherence && items.length > 1) {
      const thr = q.coherence.threshold ?? 0.75;
      const n = Math.min(items.length, 200);
      for (let i = 0; i < n; i++) {
        let close = 0;
        for (let j = 0; j < n; j++) {
          if (i === j) continue;
          if (similarityFor(this.metric, this.distance(this.vectors[items[i].slot], this.vectors[items[j].slot])) >= thr) close++;
        }
        const coh = close / (n - 1);
        items[i].final *= 1 + q.coherence.weight * coh;
        items[i].explain.coherence = coh;
      }
    }

    if (q.fusion) {
      const alpha = q.fusion.alpha ?? 0.5;
      const lex = this.lexical.search(q.fusion.lexical, kFetch, allow);
      const bySlot = new Map(items.map((it) => [it.slot, it]));
      for (const l of lex) {
        if (!bySlot.has(l.doc)) {
          const d = this.distance(qv, this.vectors[l.doc]);
          const sim = similarityFor(this.metric, d);
          const it = { slot: l.doc, distance: d, similarity: sim, final: sim, explain: { distance: d, similarity: sim } as Explain };
          items.push(it);
          bySlot.set(l.doc, it);
        }
      }
      const dense = { items: [...items].sort((a, b) => b.final - a.final).map((it) => ({ key: it.slot, score: it.final })), weight: alpha };
      const lexical = { items: lex.map((l) => ({ key: l.doc, score: l.score })), weight: 1 - alpha };
      const fused = (q.fusion.method ?? 'rrf') === 'rrf' ? rrf([dense, lexical], q.fusion.rrfK ?? 60) : rsf([dense, lexical]);
      const fusedMap = new Map(fused.map((f) => [f.key, f]));
      for (const it of items) {
        const f = fusedMap.get(it.slot);
        if (!f) continue;
        it.final = f.score;
        it.explain.fusion = { dense: f.parts[0], lexical: f.parts[1], method: q.fusion.method ?? 'rrf', denseRank: f.ranks[0], lexicalRank: f.ranks[1] };
      }
    }

    items.sort((a, b) => b.final - a.final);

    let chosen = items;
    if (q.diversity && q.diversity > 0 && items.length > 1) {
      const lambda = 1 - Math.min(1, q.diversity);
      const selected: typeof items = [];
      const pool = [...items];
      // normalise finals to [0,1] for mixing with pairwise similarity
      const maxF = pool[0].final, minF = pool[pool.length - 1].final, range = maxF - minF || 1;
      while (pool.length && selected.length < q.k) {
        let bestI = 0, best = -Infinity;
        for (let i = 0; i < pool.length; i++) {
          const rel = (pool[i].final - minF) / range;
          let red = 0;
          for (const s of selected) red = Math.max(red, similarityFor(this.metric, this.distance(this.vectors[pool[i].slot], this.vectors[s.slot])));
          const mmr = lambda * rel - (1 - lambda) * red;
          if (mmr > best) { best = mmr; bestI = i; }
        }
        const pick = pool.splice(bestI, 1)[0];
        pick.explain.mmr = best;
        selected.push(pick);
      }
      chosen = selected;
    }

    const results: SearchResult[] = chosen.slice(0, q.k).map((it) => {
      const id = this.ids[it.slot]!;
      const r: SearchResult = { id, score: it.distance, similarity: it.similarity };
      if (rerank) r.finalScore = it.final;
      if (q.includeVectors) r.vector = this.vectors[it.slot];
      if (q.includeMetadata !== false) r.metadata = this.metas[it.slot]!;
      if (q.explain) r.explain = it.explain;
      return r;
    });
    for (const r of results) this.touch(r.id);
    this.rememberQuery(qv);
    return {
      results,
      fetched,
      matched,
      complete: results.length >= q.k || exhausted,
      strategy,
      tookMs: Math.round((performance.now() - t0) * 1000) / 1000,
    };
  }

  private rememberQuery(qv: Float32Array) {
    this.recentQueries.push(Float32Array.from(qv));
    if (this.recentQueries.length > RECENT_QUERIES) this.recentQueries.shift();
  }

  // ------------------------------------------------------------------ compaction

  setCompactionPolicy(policy: CompactionPolicy | null) {
    this.customPolicy = policy;
  }

  private maybeCompact() {
    const cap = this.options.capacity;
    if (cap > 0 && this.liveCount > cap) this.compact({ target: Math.floor(cap * 0.8) });
  }

  private builtinPolicy(name: string): CompactionPolicy {
    switch (name) {
      case 'lru':
        return (r, ctx) => {
          const a = ctx.access(r.id);
          const last = a?.l ?? toMs(r.metadata[this.options.timestampKey]) ?? 0;
          return last / ctx.now;
        };
      case 'lfu':
        return (r, ctx) => ctx.access(r.id)?.c ?? 0;
      default:
        return (r, ctx) => {
          const a = ctx.access(r.id);
          const last = a?.l ?? toMs(r.metadata[this.options.timestampKey]) ?? 0;
          const ageDays = Math.max(0, ctx.now - last) / 86_400_000;
          const recency = Math.exp(-ageDays / 30);
          const frequency = 1 - 1 / (1 + (a?.c ?? 0));
          let coherence = 0.5;
          if (ctx.recentQueries.length) {
            coherence = -Infinity;
            for (const qv of ctx.recentQueries) coherence = Math.max(coherence, ctx.similarity(r.vector, qv));
            coherence = Math.max(0, Math.min(1, coherence));
          }
          const importance = typeof r.metadata.importance === 'number' ? r.metadata.importance : 0.5;
          const feedback = Math.tanh((a?.f ?? 0) / 4);
          return 0.25 * recency + 0.35 * frequency + 0.4 * coherence + 0.2 * importance + 0.2 * feedback;
        };
    }
  }

  /**
   * Evict lowest-scoring records until `target` live records remain. Pinned records (`pinned: true`)
   * are never evicted. A cluster-diversity constraint keeps at least `minPerCluster` survivors per
   * cluster key (metadata `kind` + first `tags` entry) so one dominant topic cannot evict everything else.
   */
  compact(opts: { target?: number; policy?: string; minPerCluster?: number } = {}): CompactResult {
    this.assertWritable();
    this.sync();
    const t0 = performance.now();
    const policyName = opts.policy ?? (this.customPolicy ? 'custom' : this.options.compactionPolicy);
    const policy = this.customPolicy && policyName === 'custom' ? this.customPolicy : this.builtinPolicy(policyName);
    const target = opts.target ?? (this.options.capacity > 0 ? Math.floor(this.options.capacity * 0.8) : this.liveCount);
    const minPerCluster = opts.minPerCluster ?? 2;
    const before = this.liveCount;
    if (before <= target) return { before, after: before, evicted: [], policy: policyName, tookMs: 0 };
    const ctx: CompactionContext = {
      now: Date.now(),
      recentQueries: this.recentQueries,
      access: (id) => this.access.get(id),
      similarity: (a, b) => similarityFor(this.metric, this.distance(a, b)),
    };
    const scored: Array<{ id: string; score: number; cluster: string }> = [];
    const clusterSize = new Map<string, number>();
    for (const r of this.iterate()) {
      if (r.metadata.pinned === true) continue;
      const tags = r.metadata.tags;
      const cluster = `${r.metadata.kind ?? ''}|${Array.isArray(tags) && tags.length ? tags[0] : ''}`;
      clusterSize.set(cluster, (clusterSize.get(cluster) ?? 0) + 1);
      scored.push({ id: r.id, score: policy(r, ctx), cluster });
    }
    scored.sort((a, b) => a.score - b.score);
    const evict: string[] = [];
    let remaining = before;
    for (const s of scored) {
      if (remaining <= target) break;
      const sz = clusterSize.get(s.cluster)!;
      if (sz <= minPerCluster) continue;
      clusterSize.set(s.cluster, sz - 1);
      evict.push(s.id);
      remaining--;
    }
    if (evict.length) {
      for (let i = 0; i < evict.length; i += LIMITS.maxBatch) this.deleteBatch(evict.slice(i, i + LIMITS.maxBatch), 'compact');
    }
    return { before, after: this.liveCount, evicted: evict, policy: policyName, tookMs: Math.round(performance.now() - t0) };
  }

  // ------------------------------------------------------------------ persistence

  private maybeCheckpoint() {
    if (this.generation - this.checkpointGeneration >= CHECKPOINT_EVERY_OPS || fileSize(this.logFile) > CHECKPOINT_LOG_BYTES) this.flush();
  }

  /** Write a checkpoint (records, persisted index, access stats) and truncate the log. */
  flush(opts: { rebuildIndex?: boolean } = {}): void {
    this.assertOpen();
    if (this.readOnly) return;
    withLock(this.dir, () => {
      this.applyLogTail();
      const tomb = this.ids.length - this.liveCount;
      const rebuild = opts.rebuildIndex || (this.ids.length > 0 && tomb / this.ids.length > REBUILD_TOMBSTONE_RATIO);
      if (rebuild) this.renumber();
      const vecBuf = Buffer.alloc(this.ids.length * this.dim * 4);
      for (let slot = 0; slot < this.ids.length; slot++) {
        const v = this.vectors[slot];
        Buffer.from(v.buffer, v.byteOffset, v.byteLength).copy(vecBuf, slot * this.dim * 4);
      }
      const cp: Checkpoint = {
        schemaVersion: SCHEMA_VERSION,
        generation: this.generation,
        dimensions: this.dim,
        ids: this.ids,
        metas: this.metas,
        modGen: this.modGen,
        index: this.index ? this.index.serialize() : null,
        access: Object.fromEntries(this.access),
        recentQueries: this.recentQueries.map(f32ToBase64),
      };
      writeFileAtomic(path.join(this.dir, 'vectors.bin'), vecBuf);
      writeFileAtomic(this.cpFile, JSON.stringify(cp));
      // truncate the log: other processes detect the new checkpoint (or size < their offset) and reload
      fs.writeFileSync(this.logFile, '');
      this.logOffset = 0;
      this.checkpointGeneration = this.generation;
      this.cpStamp = fileStamp(this.cpFile);
    });
  }

  /** Drop tombstoned slots and rebuild the index with dense numbering. */
  private renumber() {
    const ids: (string | null)[] = [];
    const metas: (Metadata | null)[] = [];
    const vectors: Float32Array[] = [];
    const modGen: number[] = [];
    this.idToSlot.clear();
    this.metaIndex.clear();
    this.lexical.clear();
    for (let slot = 0; slot < this.ids.length; slot++) {
      const id = this.ids[slot];
      if (id === null) continue;
      const ns = ids.length;
      ids.push(id);
      metas.push(this.metas[slot]);
      vectors.push(this.vectors[slot]);
      modGen.push(this.modGen[slot]);
      this.idToSlot.set(id, ns);
    }
    this.ids = ids;
    this.metas = metas;
    this.vectors = vectors;
    this.modGen = modGen;
    for (let slot = 0; slot < ids.length; slot++) this.indexMeta(slot, metas[slot]!);
    this.initIndex();
    if (this.index) for (let slot = 0; slot < ids.length; slot++) this.index.insert(slot);
  }

  close(): void {
    if (this.closed) return;
    if (!this.readOnly) this.flush();
    this.closed = true;
  }

  stats(): CollectionStats {
    this.sync();
    return {
      path: this.dir,
      count: this.liveCount,
      slots: this.ids.length,
      tombstones: this.ids.length - this.liveCount,
      generation: this.generation,
      checkpointGeneration: this.checkpointGeneration,
      logBytes: fileSize(this.logFile),
      options: this.options,
      embedding: this.options.embedding,
      fallbackEmbedder: !!this.manifest.fallbackEmbedder,
    };
  }

  /** Learned/behavioural state bundled with snapshots. */
  exportState() {
    return { access: Object.fromEntries(this.access), recentQueries: this.recentQueries.map(f32ToBase64) };
  }
  importState(s: { access?: Record<string, AccessStat>; recentQueries?: string[] }) {
    if (s.access) this.access = new Map(Object.entries(s.access));
    if (s.recentQueries) this.recentQueries = s.recentQueries.map(base64ToF32);
  }
}

export function toMs(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string') {
    const t = Date.parse(v);
    return Number.isNaN(t) ? null : t;
  }
  return null;
}

function describe(i: EmbeddingSpaceIdentity): string {
  return `${i.embedderKind}:${i.modelId}/${i.dimension}d/${i.normalize ? 'norm' : 'raw'}/${i.prefixPolicy}`;
}
