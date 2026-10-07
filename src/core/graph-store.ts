/**
 * Property graph with hyperedges (spec §7). Nodes, edges and hyperedges carry typed properties
 * (string, number, boolean, null, arrays) and optional embeddings in the same space as the vector
 * collections, so "what is connected to this" and "what is similar to this" compose.
 *
 * Persistence: `graph.log.jsonl` (one transaction per line) + `graph.checkpoint.json`, hydrated
 * fully on open. Reopen-reads-empty is tested as an invariant.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { MemError, invalid } from './errors.js';
import { base64ToF32, distanceFor, f32ToBase64, normalize, normalizedCosineDistance, similarityFor, toFloat32, type DistanceFn } from './distance.js';
import { appendLineSync, fileSize, fileStamp, readFrom, readJsonIfExists, withLock, writeFileAtomic } from './fsutil.js';
import type { DistanceMetric } from './types.js';
import { LIMITS } from './types.js';
import type { WitnessLog } from './witness.js';
import { evalAggregate, evalExpr, hasAggregate, idSeeds, parseCypher, type Bindings, type GraphElement, type NodePat, type Query, type RelPat } from './cypher.js';

export type PropValue = string | number | boolean | null | string[] | number[];
export type Properties = Record<string, PropValue>;

export interface GNode {
  id: string;
  labels: string[];
  properties: Properties;
  embedding?: Float32Array;
}
export interface GEdge {
  id: string;
  from: string;
  to: string;
  type: string;
  confidence: number;
  properties: Properties;
  embedding?: Float32Array;
}
export interface GHyperedge {
  id: string;
  nodes: string[];
  type: string;
  confidence: number;
  properties: Properties;
  embedding?: Float32Array;
}

type Wire<T> = Omit<T, 'embedding'> & { e?: string };
type Op =
  | { op: 'node'; n: Wire<GNode> }
  | { op: 'edge'; e: Wire<GEdge> }
  | { op: 'hyper'; h: Wire<GHyperedge> }
  | { op: 'delNode'; id: string }
  | { op: 'delEdge'; id: string }
  | { op: 'delHyper'; id: string };
interface LogEntry {
  g: number;
  ops: Op[];
}
interface Checkpoint {
  generation: number;
  nodes: Wire<GNode>[];
  edges: Wire<GEdge>[];
  hyperedges: Wire<GHyperedge>[];
}
export interface GraphManifest {
  schemaVersion: number;
  dimensions: number;
  distanceMetric: DistanceMetric;
  createdAt: string;
}

export interface GraphChange {
  generation: number;
  ops: Op[];
}

export interface QueryResult {
  columns: string[];
  rows: unknown[][];
  nodes: GNode[];
  edges: GEdge[];
  stats: { bindings: number; rows: number; tookMs: number };
}

const CHECKPOINT_EVERY = 200;

function wireNode(n: GNode): Wire<GNode> {
  const { embedding, ...rest } = n;
  return embedding ? { ...rest, e: f32ToBase64(embedding) } : rest;
}
function unwire<T extends { embedding?: Float32Array }>(w: Wire<T>): T {
  const { e, ...rest } = w as Wire<T> & { e?: string };
  return (e ? { ...rest, embedding: base64ToF32(e) } : rest) as unknown as T;
}

export class GraphStore {
  readonly dir: string;
  readonly manifest: GraphManifest;
  readonly dim: number;
  private readonly distance: DistanceFn;
  private nodes = new Map<string, GNode>();
  private edges = new Map<string, GEdge>();
  private hyperedges = new Map<string, GHyperedge>();
  private out = new Map<string, Set<string>>(); // node -> edge ids
  private inc = new Map<string, Set<string>>();
  private nodeHyper = new Map<string, Set<string>>();
  private labelIndex = new Map<string, Set<string>>();
  private generation = 0;
  private checkpointGeneration = 0;
  private logOffset = 0;
  private logFile: string;
  private cpFile: string;
  private cpStamp = 'none';
  private subscribers = new Set<(c: GraphChange) => void>();
  private tx: Op[] | null = null;
  private witness: WitnessLog | null;
  private actor: string;

  private constructor(dir: string, manifest: GraphManifest, witness: WitnessLog | null, actor: string) {
    this.dir = dir;
    this.manifest = manifest;
    this.dim = manifest.dimensions;
    this.distance = manifest.distanceMetric === 'cosine' ? normalizedCosineDistance : distanceFor(manifest.distanceMetric);
    this.logFile = path.join(dir, 'graph.log.jsonl');
    this.cpFile = path.join(dir, 'graph.checkpoint.json');
    this.witness = witness;
    this.actor = actor;
  }

  static exists(dir: string) {
    return fs.existsSync(path.join(dir, 'graph.manifest.json'));
  }

  static openOrCreate(dir: string, opts: { dimensions: number; distanceMetric?: DistanceMetric }, extra: { witness?: WitnessLog; actor?: string } = {}): GraphStore {
    fs.mkdirSync(dir, { recursive: true });
    const mf = path.join(dir, 'graph.manifest.json');
    let manifest = readJsonIfExists<GraphManifest>(mf);
    if (!manifest) {
      if (!Number.isInteger(opts.dimensions) || opts.dimensions < 1 || opts.dimensions > LIMITS.maxDimension) invalid('graph dimensions out of range');
      manifest = { schemaVersion: 1, dimensions: opts.dimensions, distanceMetric: opts.distanceMetric ?? 'cosine', createdAt: new Date().toISOString() };
      writeFileAtomic(mf, JSON.stringify(manifest, null, 2));
    } else if (manifest.dimensions !== opts.dimensions) {
      throw new MemError('DIMENSION_MISMATCH', `graph at ${dir} has ${manifest.dimensions} dimensions, caller expects ${opts.dimensions}`);
    }
    const g = new GraphStore(dir, manifest, extra.witness ?? null, extra.actor ?? 'local');
    g.load();
    return g;
  }

  // ------------------------------------------------------------------ persistence

  private load() {
    this.nodes.clear();
    this.edges.clear();
    this.hyperedges.clear();
    this.out.clear();
    this.inc.clear();
    this.nodeHyper.clear();
    this.labelIndex.clear();
    this.generation = 0;
    this.checkpointGeneration = 0;
    this.logOffset = 0;
    this.cpStamp = fileStamp(this.cpFile);
    const cp = readJsonIfExists<Checkpoint>(this.cpFile);
    if (cp) {
      for (const n of cp.nodes) this.applyNode(unwire<GNode>(n));
      for (const e of cp.edges) this.applyEdge(unwire<GEdge>(e));
      for (const h of cp.hyperedges) this.applyHyper(unwire<GHyperedge>(h));
      this.generation = cp.generation;
      this.checkpointGeneration = cp.generation;
    }
    this.applyLogTail();
  }

  private applyLogTail() {
    const size = fileSize(this.logFile);
    if (size < this.logOffset || fileStamp(this.cpFile) !== this.cpStamp) {
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
        throw new MemError('CORRUPT', `unparseable graph log line in ${this.logFile}`);
      }
      if (entry.g <= this.generation) continue;
      if (entry.g !== this.generation + 1) throw new MemError('CORRUPT', `graph log generation gap at ${entry.g}`);
      this.applyEntry(entry);
    }
    this.logOffset += start;
  }

  private applyEntry(e: LogEntry) {
    for (const op of e.ops) this.applyOp(op);
    this.generation = e.g;
  }

  private applyOp(op: Op) {
    switch (op.op) {
      case 'node':
        this.applyNode(unwire<GNode>(op.n));
        break;
      case 'edge':
        this.applyEdge(unwire<GEdge>(op.e));
        break;
      case 'hyper':
        this.applyHyper(unwire<GHyperedge>(op.h));
        break;
      case 'delNode':
        this.applyDelNode(op.id);
        break;
      case 'delEdge':
        this.applyDelEdge(op.id);
        break;
      case 'delHyper':
        this.applyDelHyper(op.id);
        break;
    }
  }

  private applyNode(n: GNode) {
    const old = this.nodes.get(n.id);
    if (old) for (const l of old.labels) this.labelIndex.get(l)?.delete(n.id);
    this.nodes.set(n.id, n);
    for (const l of n.labels) {
      let s = this.labelIndex.get(l);
      if (!s) this.labelIndex.set(l, (s = new Set()));
      s.add(n.id);
    }
    if (!this.out.has(n.id)) this.out.set(n.id, new Set());
    if (!this.inc.has(n.id)) this.inc.set(n.id, new Set());
    if (!this.nodeHyper.has(n.id)) this.nodeHyper.set(n.id, new Set());
  }
  private applyEdge(e: GEdge) {
    const old = this.edges.get(e.id);
    if (old) {
      this.out.get(old.from)?.delete(e.id);
      this.inc.get(old.to)?.delete(e.id);
    }
    this.edges.set(e.id, e);
    this.out.get(e.from)?.add(e.id);
    this.inc.get(e.to)?.add(e.id);
  }
  private applyHyper(h: GHyperedge) {
    const old = this.hyperedges.get(h.id);
    if (old) for (const n of old.nodes) this.nodeHyper.get(n)?.delete(h.id);
    this.hyperedges.set(h.id, h);
    for (const n of h.nodes) this.nodeHyper.get(n)?.add(h.id);
  }
  private applyDelNode(id: string) {
    const n = this.nodes.get(id);
    if (!n) return;
    for (const eid of [...(this.out.get(id) ?? []), ...(this.inc.get(id) ?? [])]) this.applyDelEdge(eid);
    for (const hid of [...(this.nodeHyper.get(id) ?? [])]) this.applyDelHyper(hid);
    for (const l of n.labels) this.labelIndex.get(l)?.delete(id);
    this.nodes.delete(id);
    this.out.delete(id);
    this.inc.delete(id);
    this.nodeHyper.delete(id);
  }
  private applyDelEdge(id: string) {
    const e = this.edges.get(id);
    if (!e) return;
    this.out.get(e.from)?.delete(id);
    this.inc.get(e.to)?.delete(id);
    this.edges.delete(id);
  }
  private applyDelHyper(id: string) {
    const h = this.hyperedges.get(id);
    if (!h) return;
    for (const n of h.nodes) this.nodeHyper.get(n)?.delete(id);
    this.hyperedges.delete(id);
  }

  private commitOps(ops: Op[], witnessItems: Array<{ operation: string; recordId: string; payload?: unknown }>) {
    if (this.tx) {
      this.tx.push(...ops);
      return;
    }
    withLock(this.dir, () => {
      this.applyLogTail();
      const size = fileSize(this.logFile);
      if (size > this.logOffset) fs.truncateSync(this.logFile, this.logOffset);
      // re-validate endpoints after sync
      for (const op of ops) {
        if (op.op === 'edge' && (!this.nodes.has(op.e.from) || !this.nodes.has(op.e.to))) {
          const missing = !this.nodes.has(op.e.from) ? op.e.from : op.e.to;
          if (!ops.some((o) => o.op === 'node' && o.n.id === missing)) throw new MemError('NOT_FOUND', `edge endpoint node ${missing} does not exist`);
        }
        if (op.op === 'hyper') {
          for (const n of op.h.nodes) if (!this.nodes.has(n) && !ops.some((o) => o.op === 'node' && o.n.id === n)) throw new MemError('NOT_FOUND', `hyperedge node ${n} does not exist`);
        }
      }
      const entry: LogEntry = { g: this.generation + 1, ops };
      const line = JSON.stringify(entry) + '\n';
      appendLineSync(this.logFile, line);
      this.applyEntry(entry);
      this.logOffset += Buffer.byteLength(line);
    });
    if (this.witness && witnessItems.length) this.witness.appendMany(witnessItems, this.actor);
    const change = { generation: this.generation, ops };
    for (const s of this.subscribers) {
      try {
        s(change);
      } catch {}
    }
    if (this.generation - this.checkpointGeneration >= CHECKPOINT_EVERY) this.flush();
  }

  flush() {
    withLock(this.dir, () => {
      this.applyLogTail();
      const cp: Checkpoint = {
        generation: this.generation,
        nodes: [...this.nodes.values()].map(wireNode),
        edges: [...this.edges.values()].map((e) => wireNode(e as unknown as GNode) as unknown as Wire<GEdge>),
        hyperedges: [...this.hyperedges.values()].map((h) => wireNode(h as unknown as GNode) as unknown as Wire<GHyperedge>),
      };
      writeFileAtomic(this.cpFile, JSON.stringify(cp));
      fs.writeFileSync(this.logFile, '');
      this.logOffset = 0;
      this.checkpointGeneration = this.generation;
      this.cpStamp = fileStamp(this.cpFile);
    });
  }

  close() {
    this.flush();
  }

  sync() {
    this.applyLogTail();
  }

  // ------------------------------------------------------------------ transactions / subscriptions

  begin() {
    if (this.tx) throw new MemError('INVALID_ARGUMENT', 'transaction already open');
    this.tx = [];
  }
  commit() {
    if (!this.tx) throw new MemError('INVALID_ARGUMENT', 'no open transaction');
    const ops = this.tx;
    this.tx = null;
    if (ops.length) this.commitOps(ops, ops.map((o) => ({ operation: `graph.${o.op}`, recordId: opId(o) })));
  }
  rollback() {
    this.tx = null;
  }
  /** Run fn inside a transaction; rolled back on throw. */
  transaction<T>(fn: () => T): T {
    this.begin();
    try {
      const r = fn();
      this.commit();
      return r;
    } catch (e) {
      this.rollback();
      throw e;
    }
  }
  subscribe(cb: (c: GraphChange) => void): () => void {
    this.subscribers.add(cb);
    return () => this.subscribers.delete(cb);
  }

  // ------------------------------------------------------------------ writes

  private validateProps(p: Properties | undefined): Properties {
    const out: Properties = {};
    for (const [k, v] of Object.entries(p ?? {})) {
      const ok =
        v === null ||
        typeof v === 'string' ||
        typeof v === 'number' ||
        typeof v === 'boolean' ||
        (Array.isArray(v) && (v.every((x) => typeof x === 'string') || v.every((x) => typeof x === 'number')));
      if (!ok) invalid(`property ${k} must be string, number, boolean, null or a homogeneous array`, { key: k });
      out[k] = v;
    }
    if (Buffer.byteLength(JSON.stringify(out)) > LIMITS.maxMetadataBytes) throw new MemError('LIMIT_EXCEEDED', 'properties exceed size limit');
    return out;
  }
  private validateEmbedding(e?: Float32Array | number[]): Float32Array | undefined {
    if (e === undefined) return undefined;
    const v = toFloat32(e);
    if (v.length !== this.dim) throw new MemError('DIMENSION_MISMATCH', `embedding has ${v.length} dimensions, graph has ${this.dim}`);
    return this.manifest.distanceMetric === 'cosine' ? normalize(v) : v;
  }
  private validateId(id: unknown): string {
    if (typeof id !== 'string' || !id) invalid('id must be a non-empty string');
    if (Buffer.byteLength(id) > LIMITS.maxIdBytes) throw new MemError('LIMIT_EXCEEDED', 'id too long');
    return id;
  }

  createNode(n: { id?: string; labels?: string[]; properties?: Properties; embedding?: Float32Array | number[] }, opts: { upsert?: boolean } = {}): GNode {
    this.sync();
    const id = n.id ? this.validateId(n.id) : crypto.randomUUID();
    if (!opts.upsert && this.nodes.has(id)) throw new MemError('ALREADY_EXISTS', `node ${id} already exists`);
    const node: GNode = { id, labels: [...new Set(n.labels ?? [])], properties: this.validateProps(n.properties), embedding: this.validateEmbedding(n.embedding) };
    this.commitOps([{ op: 'node', n: wireNode(node) }], [{ operation: 'graph.node', recordId: id, payload: { labels: node.labels, properties: node.properties } }]);
    return node;
  }

  upsertNode(n: { id: string; labels?: string[]; properties?: Properties; embedding?: Float32Array | number[] }): GNode {
    const existing = this.getNode(n.id);
    if (!existing) return this.createNode(n, { upsert: true });
    return this.createNode(
      {
        id: n.id,
        labels: [...new Set([...existing.labels, ...(n.labels ?? [])])],
        properties: { ...existing.properties, ...(n.properties ?? {}) },
        embedding: n.embedding ?? existing.embedding,
      },
      { upsert: true },
    );
  }

  createEdge(e: { id?: string; from: string; to: string; type: string; confidence?: number; properties?: Properties; embedding?: Float32Array | number[] }): GEdge {
    this.sync();
    const id = e.id ? this.validateId(e.id) : crypto.randomUUID();
    if (this.edges.has(id)) throw new MemError('ALREADY_EXISTS', `edge ${id} already exists`);
    if (typeof e.type !== 'string' || !e.type) invalid('edge type must be a non-empty string');
    if (!this.tx) {
      if (!this.nodes.has(e.from)) throw new MemError('NOT_FOUND', `node ${e.from} does not exist`);
      if (!this.nodes.has(e.to)) throw new MemError('NOT_FOUND', `node ${e.to} does not exist`);
    }
    const conf = e.confidence ?? 1;
    if (typeof conf !== 'number' || conf < 0 || conf > 1) invalid('confidence must be in [0,1]');
    const edge: GEdge = { id, from: e.from, to: e.to, type: e.type, confidence: conf, properties: this.validateProps(e.properties), embedding: this.validateEmbedding(e.embedding) };
    this.commitOps([{ op: 'edge', e: wireNode(edge as unknown as GNode) as unknown as Wire<GEdge> }], [{ operation: 'graph.edge', recordId: id, payload: { from: e.from, to: e.to, type: e.type } }]);
    return edge;
  }

  createHyperedge(h: { id?: string; nodes: string[]; type: string; confidence?: number; properties?: Properties; embedding?: Float32Array | number[] }): GHyperedge {
    this.sync();
    const id = h.id ? this.validateId(h.id) : crypto.randomUUID();
    if (this.hyperedges.has(id)) throw new MemError('ALREADY_EXISTS', `hyperedge ${id} already exists`);
    if (!Array.isArray(h.nodes) || h.nodes.length < 2) invalid('a hyperedge needs at least two nodes');
    if (!this.tx) for (const n of h.nodes) if (!this.nodes.has(n)) throw new MemError('NOT_FOUND', `node ${n} does not exist`);
    const conf = h.confidence ?? 1;
    if (typeof conf !== 'number' || conf < 0 || conf > 1) invalid('confidence must be in [0,1]');
    const he: GHyperedge = { id, nodes: [...h.nodes], type: h.type, confidence: conf, properties: this.validateProps(h.properties), embedding: this.validateEmbedding(h.embedding) };
    this.commitOps([{ op: 'hyper', h: wireNode(he as unknown as GNode) as unknown as Wire<GHyperedge> }], [{ operation: 'graph.hyperedge', recordId: id, payload: { nodes: h.nodes, type: h.type } }]);
    return he;
  }

  updateHyperedge(id: string, patch: { confidence?: number; properties?: Properties }): GHyperedge {
    const h = this.getHyperedge(id);
    if (!h) throw new MemError('NOT_FOUND', `hyperedge ${id} not found`);
    const next: GHyperedge = { ...h, confidence: patch.confidence ?? h.confidence, properties: { ...h.properties, ...this.validateProps(patch.properties) } };
    this.commitOps([{ op: 'hyper', h: wireNode(next as unknown as GNode) as unknown as Wire<GHyperedge> }], [{ operation: 'graph.hyperedge.update', recordId: id, payload: patch }]);
    return next;
  }

  deleteNode(id: string): boolean {
    this.sync();
    if (!this.nodes.has(id)) return false;
    this.commitOps([{ op: 'delNode', id }], [{ operation: 'graph.delNode', recordId: id }]);
    return true;
  }
  deleteEdge(id: string): boolean {
    this.sync();
    if (!this.edges.has(id)) return false;
    this.commitOps([{ op: 'delEdge', id }], [{ operation: 'graph.delEdge', recordId: id }]);
    return true;
  }
  deleteHyperedge(id: string): boolean {
    this.sync();
    if (!this.hyperedges.has(id)) return false;
    this.commitOps([{ op: 'delHyper', id }], [{ operation: 'graph.delHyper', recordId: id }]);
    return true;
  }

  batchInsert(b: { nodes?: Parameters<GraphStore['createNode']>[0][]; edges?: Parameters<GraphStore['createEdge']>[0][]; hyperedges?: Parameters<GraphStore['createHyperedge']>[0][] }) {
    return this.transaction(() => ({
      nodes: (b.nodes ?? []).map((n) => this.createNode(n, { upsert: true })),
      edges: (b.edges ?? []).map((e) => this.createEdge(e)),
      hyperedges: (b.hyperedges ?? []).map((h) => this.createHyperedge(h)),
    }));
  }

  // ------------------------------------------------------------------ reads

  getNode(id: string): GNode | null {
    this.sync();
    return this.nodes.get(id) ?? null;
  }
  getEdge(id: string): GEdge | null {
    this.sync();
    return this.edges.get(id) ?? null;
  }
  getHyperedge(id: string): GHyperedge | null {
    this.sync();
    return this.hyperedges.get(id) ?? null;
  }
  hasNode(id: string) {
    this.sync();
    return this.nodes.has(id);
  }

  edgesOf(id: string, opts: { direction?: 'out' | 'in' | 'both'; types?: string[] } = {}): GEdge[] {
    this.sync();
    const dir = opts.direction ?? 'both';
    const ids = new Set<string>();
    if (dir !== 'in') for (const e of this.out.get(id) ?? []) ids.add(e);
    if (dir !== 'out') for (const e of this.inc.get(id) ?? []) ids.add(e);
    const out: GEdge[] = [];
    for (const eid of ids) {
      const e = this.edges.get(eid)!;
      if (opts.types && !opts.types.includes(e.type)) continue;
      out.push(e);
    }
    return out;
  }

  hyperedgesOf(id: string, types?: string[]): GHyperedge[] {
    this.sync();
    const out: GHyperedge[] = [];
    for (const hid of this.nodeHyper.get(id) ?? []) {
      const h = this.hyperedges.get(hid)!;
      if (types && !types.includes(h.type)) continue;
      out.push(h);
    }
    return out;
  }

  /** Node ids within k hops over any (or the given) edge types; hyperedges count as one hop between members. */
  kHopNeighbors(id: string, k: number, opts: { types?: string[]; direction?: 'out' | 'in' | 'both'; includeHyperedges?: boolean; limit?: number } = {}): Array<{ id: string; hops: number }> {
    this.sync();
    if (!this.nodes.has(id)) throw new MemError('NOT_FOUND', `node ${id} not found`);
    if (!Number.isInteger(k) || k < 0) invalid('k must be a non-negative integer');
    const limit = opts.limit ?? 10_000;
    const seen = new Map<string, number>([[id, 0]]);
    let frontier = [id];
    for (let hop = 1; hop <= k && frontier.length; hop++) {
      const next: string[] = [];
      for (const cur of frontier) {
        const nbrs: string[] = [];
        for (const e of this.edgesOf(cur, { direction: opts.direction, types: opts.types })) nbrs.push(e.from === cur ? e.to : e.from);
        if (opts.includeHyperedges !== false) for (const h of this.hyperedgesOf(cur, opts.types)) for (const n of h.nodes) if (n !== cur) nbrs.push(n);
        for (const n of nbrs) {
          if (seen.has(n)) continue;
          seen.set(n, hop);
          next.push(n);
          if (seen.size > limit) break;
        }
      }
      frontier = next;
    }
    seen.delete(id);
    return [...seen.entries()].map(([nid, hops]) => ({ id: nid, hops }));
  }

  private similar<T extends { id: string; embedding?: Float32Array }>(items: Iterable<T>, q: Float32Array, k: number, pred?: (t: T) => boolean) {
    let qv = toFloat32(q);
    if (qv.length !== this.dim) throw new MemError('DIMENSION_MISMATCH', `query has ${qv.length} dimensions, graph has ${this.dim}`);
    if (this.manifest.distanceMetric === 'cosine') qv = normalize(qv);
    const scored: Array<{ item: T; score: number; similarity: number }> = [];
    for (const it of items) {
      if (!it.embedding) continue;
      if (pred && !pred(it)) continue;
      const d = this.distance(qv, it.embedding);
      scored.push({ item: it, score: d, similarity: similarityFor(this.manifest.distanceMetric, d) });
    }
    scored.sort((a, b) => a.score - b.score);
    return scored.slice(0, k);
  }

  searchNodes(embedding: Float32Array | number[], k: number, opts: { labels?: string[] } = {}) {
    this.sync();
    return this.similar(this.nodes.values(), toFloat32(embedding), k, opts.labels ? (n) => opts.labels!.some((l) => n.labels.includes(l)) : undefined).map((s) => ({ node: s.item, score: s.score, similarity: s.similarity }));
  }
  searchEdges(embedding: Float32Array | number[], k: number, opts: { types?: string[] } = {}) {
    this.sync();
    return this.similar(this.edges.values(), toFloat32(embedding), k, opts.types ? (e) => opts.types!.includes(e.type) : undefined).map((s) => ({ edge: s.item, score: s.score, similarity: s.similarity }));
  }
  searchHyperedges(embedding: Float32Array | number[], k: number, opts: { types?: string[] } = {}) {
    this.sync();
    return this.similar(this.hyperedges.values(), toFloat32(embedding), k, opts.types ? (h) => opts.types!.includes(h.type) : undefined).map((s) => ({ hyperedge: s.item, score: s.score, similarity: s.similarity }));
  }

  stats() {
    this.sync();
    let degree = 0;
    for (const s of this.out.values()) degree += s.size;
    for (const s of this.inc.values()) degree += s.size;
    const labels: Record<string, number> = {};
    for (const [l, s] of this.labelIndex) labels[l] = s.size;
    const edgeTypes: Record<string, number> = {};
    for (const e of this.edges.values()) edgeTypes[e.type] = (edgeTypes[e.type] ?? 0) + 1;
    return {
      nodes: this.nodes.size,
      edges: this.edges.size,
      hyperedges: this.hyperedges.size,
      averageDegree: this.nodes.size ? degree / this.nodes.size : 0,
      labels,
      edgeTypes,
      generation: this.generation,
      dimensions: this.dim,
    };
  }

  *allNodes(): Generator<GNode> {
    this.sync();
    yield* this.nodes.values();
  }
  *allEdges(): Generator<GEdge> {
    yield* this.edges.values();
  }
  *allHyperedges(): Generator<GHyperedge> {
    yield* this.hyperedges.values();
  }

  export() {
    this.sync();
    return {
      nodes: [...this.nodes.values()].map(wireNode),
      edges: [...this.edges.values()].map((e) => wireNode(e as unknown as GNode) as unknown as Wire<GEdge>),
      hyperedges: [...this.hyperedges.values()].map((h) => wireNode(h as unknown as GNode) as unknown as Wire<GHyperedge>),
    };
  }

  import(data: ReturnType<GraphStore['export']>) {
    this.transaction(() => {
      for (const n of data.nodes) this.createNode(unwire<GNode>(n), { upsert: true });
      for (const e of data.edges) this.createEdge(unwire<GEdge>(e));
      for (const h of data.hyperedges) this.createHyperedge(unwire<GHyperedge>(h));
    });
  }

  // ------------------------------------------------------------------ cypher

  query(cypher: string, opts: { maxRows?: number } = {}): QueryResult {
    const t0 = performance.now();
    this.sync();
    if (typeof cypher !== 'string' || cypher.length > LIMITS.maxQueryChars) invalid('query must be a string under the length limit');
    const q = parseCypher(cypher);
    const seeds = idSeeds(q.where);
    let bindings: Bindings[] = [{}];
    for (const pat of q.patterns) bindings = this.joinPattern(bindings, pat, seeds);
    if (q.where) {
      const w = q.where;
      bindings = bindings.filter((b) => evalExpr(w, b) === true);
    }
    const nBindings = bindings.length;

    let columns: string[];
    let rows: unknown[][];
    if (q.returnAll) {
      const vars = new Set<string>();
      for (const pat of q.patterns) {
        for (const n of pat.nodes) if (n.v) vars.add(n.v);
        for (const r of pat.rels) if (r.v) vars.add(r.v);
      }
      columns = [...vars];
      rows = bindings.map((b) => columns.map((c) => b[c] ?? null));
    } else {
      columns = q.items.map((it) => it.alias);
      const aggIdx = q.items.map((it) => hasAggregate(it.e));
      if (aggIdx.some(Boolean)) {
        const groups = new Map<string, Bindings[]>();
        for (const b of bindings) {
          const key = JSON.stringify(q.items.map((it, i) => (aggIdx[i] ? null : plain(evalExpr(it.e, b)))));
          let g = groups.get(key);
          if (!g) groups.set(key, (g = []));
          g.push(b);
        }
        if (!groups.size && aggIdx.every(Boolean)) groups.set('[]', []);
        rows = [...groups.values()].map((g) => q.items.map((it, i) => (aggIdx[i] ? evalAggregate(it.e, g) : evalExpr(it.e, g[0]))));
      } else {
        rows = bindings.map((b) => q.items.map((it) => evalExpr(it.e, b)));
      }
    }
    if (q.orderBy.length) {
      const keyed = rows.map((row, i) => {
        const b = bindings[i] ?? {};
        const scope: Bindings = { ...b };
        return { row, keys: q.orderBy.map((o) => sortKey(o.e, row, columns, scope)) };
      });
      keyed.sort((a, b) => {
        for (let i = 0; i < q.orderBy.length; i++) {
          const c = compareSort(a.keys[i], b.keys[i]);
          if (c !== 0) return q.orderBy[i].desc ? -c : c;
        }
        return 0;
      });
      rows = keyed.map((k) => k.row);
    }
    if (q.distinct) {
      const seen = new Set<string>();
      rows = rows.filter((r) => {
        const k = JSON.stringify(r.map(plain));
        if (seen.has(k)) return false;
        seen.add(k);
        return true;
      });
    }
    if (q.skip) rows = rows.slice(q.skip);
    if (q.limit !== undefined) rows = rows.slice(0, q.limit);
    const maxRows = opts.maxRows ?? 1000;
    if (rows.length > maxRows) rows = rows.slice(0, maxRows);

    const nodes = new Map<string, GNode>();
    const edges = new Map<string, GEdge>();
    for (const row of rows) for (const v of row) collectElements(v, nodes, edges);
    return {
      columns,
      rows: rows.map((r) => r.map(plain)),
      nodes: [...nodes.values()],
      edges: [...edges.values()],
      stats: { bindings: nBindings, rows: rows.length, tookMs: Math.round((performance.now() - t0) * 1000) / 1000 },
    };
  }

  private nodeCandidates(p: NodePat, seeds: Map<string, string>): Iterable<GNode> {
    if (p.v && seeds.has(p.v)) {
      const n = this.nodes.get(seeds.get(p.v)!);
      return n ? [n] : [];
    }
    if (typeof p.props.id === 'string') {
      const n = this.nodes.get(p.props.id);
      return n ? [n] : [];
    }
    if (p.labels.length) {
      let best: Set<string> | null = null;
      for (const l of p.labels) {
        const s = this.labelIndex.get(l) ?? new Set<string>();
        if (!best || s.size < best.size) best = s;
      }
      return [...best!].map((id) => this.nodes.get(id)!);
    }
    return this.nodes.values();
  }

  private nodeMatches(n: GNode, p: NodePat): boolean {
    for (const l of p.labels) if (!n.labels.includes(l)) return false;
    for (const [k, v] of Object.entries(p.props)) {
      if (k === 'id') {
        if (n.id !== v) return false;
        continue;
      }
      if (!eqProp(n.properties[k], v)) return false;
    }
    return true;
  }

  private relMatches(e: GEdge, p: RelPat): boolean {
    if (p.types.length && !p.types.includes(e.type)) return false;
    for (const [k, v] of Object.entries(p.props)) {
      if (k === 'confidence') {
        if (e.confidence !== v) return false;
        continue;
      }
      if (!eqProp(e.properties[k], v)) return false;
    }
    return true;
  }

  private joinPattern(prev: Bindings[], pat: { nodes: NodePat[]; rels: RelPat[] }, seeds: Map<string, string>): Bindings[] {
    const out: Bindings[] = [];
    for (const base of prev) {
      const first = pat.nodes[0];
      const starts: GNode[] = first.v && base[first.v] ? [this.nodes.get(base[first.v].id)!].filter((n) => n && this.nodeMatches(n, first)) : [...this.nodeCandidates(first, seeds)].filter((n) => this.nodeMatches(n, first));
      for (const s of starts) {
        const b0: Bindings = { ...base };
        if (first.v) b0[first.v] = s;
        this.expand(b0, s, pat, 0, out);
      }
    }
    return out;
  }

  private expand(b: Bindings, cur: GNode, pat: { nodes: NodePat[]; rels: RelPat[] }, i: number, out: Bindings[]) {
    if (i >= pat.rels.length) {
      out.push(b);
      return;
    }
    const rel = pat.rels[i];
    const nextPat = pat.nodes[i + 1];
    const edges = this.edgesOf(cur.id, { direction: rel.dir === 'both' ? 'both' : rel.dir });
    for (const e of edges) {
      if (!this.relMatches(e, rel)) continue;
      if (rel.v && b[rel.v] && b[rel.v].id !== e.id) continue;
      const otherId = e.from === cur.id ? e.to : e.from;
      if (rel.dir === 'out' && e.from !== cur.id) continue;
      if (rel.dir === 'in' && e.to !== cur.id) continue;
      const other = this.nodes.get(otherId);
      if (!other || !this.nodeMatches(other, nextPat)) continue;
      if (nextPat.v && b[nextPat.v] && b[nextPat.v].id !== other.id) continue;
      // no edge reuse within one pattern
      if (Object.values(b).some((x) => x === e)) continue;
      const nb: Bindings = { ...b };
      if (rel.v) nb[rel.v] = e as unknown as GraphElement;
      if (nextPat.v) nb[nextPat.v] = other;
      this.expand(nb, other, pat, i + 1, out);
    }
  }
}

function opId(o: Op): string {
  switch (o.op) {
    case 'node':
      return o.n.id;
    case 'edge':
      return o.e.id;
    case 'hyper':
      return o.h.id;
    default:
      return o.id;
  }
}

function eqProp(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((x, i) => x === b[i]);
  return false;
}

function plain(v: unknown): unknown {
  if (v instanceof Float32Array) return undefined;
  if (Array.isArray(v)) return v.map(plain);
  if (v && typeof v === 'object') {
    const el = v as GraphElement & { embedding?: Float32Array; nodes?: string[] };
    const o: Record<string, unknown> = { id: el.id };
    if (el.labels) o.labels = el.labels;
    if (el.type) o.type = el.type;
    if (el.from) o.from = el.from;
    if (el.to) o.to = el.to;
    if (el.nodes) o.nodes = el.nodes;
    if (el.confidence !== undefined) o.confidence = el.confidence;
    o.properties = el.properties;
    return o;
  }
  return v;
}

function sortKey(e: import('./cypher.js').Expr, row: unknown[], columns: string[], scope: Bindings): unknown {
  if (e.k === 'var') {
    const ci = columns.indexOf(e.name);
    if (ci >= 0) return row[ci];
  }
  try {
    return evalExpr(e, scope);
  } catch {
    return null;
  }
}

function compareSort(a: unknown, b: unknown): number {
  if (a === null || a === undefined) return b === null || b === undefined ? 0 : 1;
  if (b === null || b === undefined) return -1;
  if (typeof a === 'number' && typeof b === 'number') return a - b;
  const sa = String(a), sb = String(b);
  return sa < sb ? -1 : sa > sb ? 1 : 0;
}

function collectElements(v: unknown, nodes: Map<string, GNode>, edges: Map<string, GEdge>) {
  if (Array.isArray(v)) {
    for (const x of v) collectElements(x, nodes, edges);
    return;
  }
  if (v && typeof v === 'object' && 'properties' in (v as object)) {
    const el = v as GNode & GEdge;
    if (el.labels) nodes.set(el.id, el);
    else if (el.from) edges.set(el.id, el);
  }
}
