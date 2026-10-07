/**
 * Typed agent memory over the vector collection and graph store (spec §8, §9, §10).
 *
 * Namespace directory layout:
 *   <root>/<namespace>/memory/        vector collection (facts, decisions, episodes, skills, turns ...)
 *   <root>/<namespace>/graph/         relations, causal hyperedges, provenance edges
 *   <root>/<namespace>/sessions.json  working memory (sessions with TTL)
 *   <root>/<namespace>/learning.json  routing values and feedback switch
 *   <root>/<namespace>/queries.json   recent recall ids (so outcomes can be attributed)
 *   <root>/<namespace>/witness.jsonl  hash-linked audit log of every write
 *   <root>/<namespace>/branches/      copy-on-write branches
 *   <root>/<namespace>/snapshots/     default snapshot location
 *
 * Reads never mutate memory. Learned state changes only through recordOutcome / recordSkillOutcome,
 * and every such change is a witness entry.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { Collection, type CompactResult } from '../core/vector-store.js';
import { GraphStore, type GHyperedge } from '../core/graph-store.js';
import { WitnessLog } from '../core/witness.js';
import { Branch } from '../core/branch.js';
import { MemError, invalid } from '../core/errors.js';
import { confinePath, readJsonIfExists, withLock, writeFileAtomic } from '../core/fsutil.js';
import { readSnapshot, writeSnapshot } from '../core/snapshot.js';
import { base64ToF32, f32ToBase64, normalizedCosineDistance, similarityFor } from '../core/distance.js';
import type { EmbeddingProvider } from '../embedding/provider.js';
import type { CollectionOptions, EmbeddingSpaceIdentity, Explain, Filter, MemRecord, Metadata, SearchResult } from '../core/types.js';
import { LIMITS } from '../core/types.js';

export const MAX_DATE_MS = 8_640_000_000_000_000;
const RESERVED = new Set(['kind', 'text', 'createdAt', 'createdAtMs', 'updatedAtMs', 'validFromMs', 'validToMs', 'supersededBy', 'tokens', 'importance', 'novelty', '_caps', 'expiresAtMs']);
const NAMESPACE_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;

export interface AgentMemoryOptions {
  root: string;
  namespace?: string;
  embedder: EmbeddingProvider;
  fallbackActivated?: boolean;
  actor?: string;
  capacity?: number;
  learning?: boolean;
  index?: Partial<CollectionOptions['index']>;
}

export interface RememberInput {
  text: string;
  kind?: string;
  tags?: string[];
  source?: string;
  confidence?: number;
  /** Explicit rating in [0,1]; combined with novelty into the stored importance. */
  importance?: number;
  metadata?: Metadata;
  id?: string;
  supersedes?: string;
  validFrom?: string | number;
  validTo?: string | number;
  pinned?: boolean;
  capabilities?: number | string;
  sessionId?: string;
  /** Relations to other memory ids or entity ids (graph edges). */
  relations?: Array<{ to: string; type: string; confidence?: number }>;
  expiresAt?: string | number;
  branch?: string;
}

export interface RememberResult {
  id: string;
  kind: string;
  importance: number;
  novelty: number;
  /** Existing memories very close to the new one, so the caller can decide about duplicates or contradictions. */
  similar: Array<{ id: string; similarity: number; text: string; kind: string }>;
  supersededId?: string;
  embedder: string;
}

export interface RecallInput {
  text: string;
  k?: number;
  kinds?: string[];
  tags?: string[];
  filter?: Filter;
  since?: string | number;
  until?: string | number;
  asOf?: string | number;
  includeSuperseded?: boolean;
  decay?: { halfLifeDays: number } | boolean;
  hybrid?: boolean;
  diversity?: number;
  coherence?: number;
  budgetTokens?: number;
  capabilities?: number | string;
  sessionId?: string;
  explain?: boolean;
  /** Attach graph neighbours within this many hops to each hit. */
  expandGraph?: number;
  minSimilarity?: number;
  branch?: string;
}

export interface RecallHit {
  id: string;
  kind: string;
  text: string;
  similarity: number;
  finalScore: number;
  tags: string[];
  source?: string;
  confidence?: number;
  importance?: number;
  createdAt: string;
  metadata: Metadata;
  explain?: Explain;
  related?: Array<{ id: string; type: string; direction: 'out' | 'in'; hops: number }>;
}

export interface RecallResult {
  queryId: string;
  results: RecallHit[];
  fetched: number;
  matched: number;
  complete: boolean;
  strategy: string;
  tookMs: number;
  embedder: string;
  semantic: boolean;
  warning?: string;
}

interface Session {
  id: string;
  startedAt: string;
  expiresAtMs: number;
  ttlMs: number;
  metadata: Metadata;
  turns: Array<{ id: string; role: string; content: string; toolCalls?: unknown[]; createdAt: string; rememberedAs?: string }>;
  endedAt?: string;
}

interface LearningState {
  enabled: boolean;
  policy: Record<string, Record<string, { value: number; n: number }>>;
  updatedAt?: string;
}

interface RecentQuery {
  id: string;
  at: number;
  text: string;
  resultIds: string[];
}

function toMsOrNull(v: unknown): number | null {
  if (v === undefined || v === null) return null;
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string') {
    const t = Date.parse(v);
    if (Number.isNaN(t)) invalid(`invalid timestamp ${v}`);
    return t;
  }
  invalid('timestamp must be a number (ms) or ISO string');
}

function approxTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

export class AgentMemory {
  readonly root: string;
  readonly namespace: string;
  readonly dir: string;
  readonly embedder: EmbeddingProvider;
  readonly identity: EmbeddingSpaceIdentity;
  readonly fallbackActivated: boolean;
  readonly actor: string;
  readonly witness: WitnessLog;
  readonly store: Collection;
  readonly graph: GraphStore;
  private learning: LearningState;
  private branches = new Map<string, Branch>();
  private closed = false;

  constructor(opts: AgentMemoryOptions) {
    if (!opts.embedder.isReady()) throw new MemError('EMBEDDER_UNAVAILABLE', 'embedder must be initialised before creating AgentMemory');
    this.root = path.resolve(opts.root);
    this.namespace = opts.namespace ?? 'default';
    if (!NAMESPACE_RE.test(this.namespace)) throw new MemError('PATH_UNSAFE', `invalid namespace ${this.namespace}`);
    this.dir = confinePath(this.root, this.namespace);
    fs.mkdirSync(this.dir, { recursive: true });
    this.embedder = opts.embedder;
    this.identity = opts.embedder.identity();
    this.fallbackActivated = !!opts.fallbackActivated;
    this.actor = opts.actor ?? 'agent';
    this.witness = new WitnessLog(this.dir);
    const memDir = path.join(this.dir, 'memory');
    if (Collection.exists(memDir)) {
      this.store = Collection.open(memDir, { witness: this.witness, actor: this.actor });
      this.store.assertSpace(this.identity);
    } else {
      this.store = Collection.create(
        memDir,
        { dimensions: this.identity.dimension, distanceMetric: 'cosine', embedding: this.identity, capacity: opts.capacity ?? 0, index: opts.index },
        { witness: this.witness, actor: this.actor, fallbackEmbedder: this.fallbackActivated },
      );
    }
    this.graph = GraphStore.openOrCreate(path.join(this.dir, 'graph'), { dimensions: this.identity.dimension }, { witness: this.witness, actor: this.actor });
    this.learning = readJsonIfExists<LearningState>(path.join(this.dir, 'learning.json')) ?? { enabled: !!opts.learning, policy: {} };
    if (opts.learning !== undefined) this.learning.enabled = opts.learning;
  }

  // ------------------------------------------------------------------ helpers

  private target(branch?: string): Collection | Branch {
    if (!branch) return this.store;
    return this.openBranch(branch);
  }

  private openBranch(name: string): Branch {
    if (!NAMESPACE_RE.test(name)) throw new MemError('PATH_UNSAFE', `invalid branch name ${name}`);
    let b = this.branches.get(name);
    if (b) return b;
    const dir = path.join(this.dir, 'branches', name);
    if (!fs.existsSync(path.join(dir, 'branch.json'))) throw new MemError('NOT_FOUND', `branch ${name} does not exist`);
    b = Branch.open(this.store, dir, { witness: this.witness, actor: this.actor });
    this.branches.set(name, b);
    return b;
  }

  private saveLearning() {
    this.learning.updatedAt = new Date().toISOString();
    writeFileAtomic(path.join(this.dir, 'learning.json'), JSON.stringify(this.learning, null, 2));
  }

  private loadSessions(): Record<string, Session> {
    return readJsonIfExists<Record<string, Session>>(path.join(this.dir, 'sessions.json')) ?? {};
  }
  private saveSessions(s: Record<string, Session>) {
    writeFileAtomic(path.join(this.dir, 'sessions.json'), JSON.stringify(s));
  }
  private withSessions<T>(fn: (s: Record<string, Session>) => T): T {
    return withLock(this.dir, () => {
      const s = this.loadSessions();
      const r = fn(s);
      this.saveSessions(s);
      return r;
    }, { name: 'sessions' });
  }

  private loadQueries(): RecentQuery[] {
    return readJsonIfExists<RecentQuery[]>(path.join(this.dir, 'queries.json')) ?? [];
  }
  private pushQuery(q: RecentQuery) {
    try {
      withLock(this.dir, () => {
        const list = this.loadQueries();
        list.push(q);
        while (list.length > 200) list.shift();
        writeFileAtomic(path.join(this.dir, 'queries.json'), JSON.stringify(list));
      }, { name: 'queries' });
    } catch {
      /* query bookkeeping is best-effort */
    }
  }

  private async embedPassages(texts: string[]): Promise<Float32Array[]> {
    for (const t of texts) if (t.length > LIMITS.maxQueryChars) throw new MemError('LIMIT_EXCEEDED', `text exceeds ${LIMITS.maxQueryChars} characters`);
    return this.embedder.embedBatch(texts, 'passage');
  }

  private async embedQuery(text: string): Promise<Float32Array> {
    if (typeof text !== 'string' || !text.trim()) invalid('text must be a non-empty string');
    if (text.length > LIMITS.maxQueryChars) throw new MemError('LIMIT_EXCEEDED', `query exceeds ${LIMITS.maxQueryChars} characters`);
    return this.embedder.embedQuery(text);
  }

  private hit(r: SearchResult): RecallHit {
    const m = r.metadata ?? {};
    const { text: _t, ...rest } = m;
    return {
      id: r.id,
      kind: String(m.kind ?? 'fact'),
      text: String(m.text ?? ''),
      similarity: round(r.similarity),
      finalScore: round(r.finalScore ?? r.similarity),
      tags: Array.isArray(m.tags) ? (m.tags as string[]) : [],
      source: m.source as string | undefined,
      confidence: m.confidence as number | undefined,
      importance: m.importance as number | undefined,
      createdAt: String(m.createdAt ?? ''),
      metadata: rest,
      explain: r.explain,
    };
  }

  private ensureMemoryNode(id: string, kind: string) {
    if (!this.graph.hasNode(id)) this.graph.createNode({ id, labels: ['Memory', capitalize(kind)], properties: { kind } });
  }

  // ------------------------------------------------------------------ remember / recall

  async remember(input: RememberInput): Promise<RememberResult> {
    if (typeof input.text !== 'string' || !input.text.trim()) invalid('text must be a non-empty string');
    const kind = (input.kind ?? 'fact').toLowerCase();
    if (!/^[a-z][a-z0-9_-]{0,31}$/.test(kind)) invalid(`invalid kind ${kind}`);
    const now = Date.now();
    const id = input.id ?? crypto.randomUUID();
    const target = this.target(input.branch);
    const [vector] = await this.embedPassages([input.text]);

    // novelty: distance to the nearest existing memory of any kind
    const near = target.search({ vector, k: 5, includeMetadata: true, filter: { validToMs: { $gt: now } } });
    const novelty = near.results.length ? round(Math.max(0, Math.min(1, 1 - near.results[0].similarity))) : 1;
    const similar = near.results.filter((r) => r.similarity >= 0.8 && r.id !== id).map((r) => ({ id: r.id, similarity: round(r.similarity), text: String(r.metadata?.text ?? '').slice(0, 300), kind: String(r.metadata?.kind ?? 'fact') }));
    const explicit = input.importance;
    if (explicit !== undefined && (typeof explicit !== 'number' || explicit < 0 || explicit > 1)) invalid('importance must be in [0,1]');
    const importance = round(explicit !== undefined ? 0.6 * explicit + 0.4 * novelty : 0.5 * novelty + 0.5 * 0.5);

    const metadata: Metadata = {};
    for (const [k, v] of Object.entries(input.metadata ?? {})) {
      if (RESERVED.has(k)) invalid(`metadata key ${k} is reserved`);
      metadata[k] = v;
    }
    const validFromMs = toMsOrNull(input.validFrom) ?? now;
    const validToMs = toMsOrNull(input.validTo) ?? MAX_DATE_MS;
    if (validToMs <= validFromMs) invalid('validTo must be after validFrom');
    Object.assign(metadata, {
      kind,
      text: input.text,
      tags: [...new Set((input.tags ?? []).map(String))],
      createdAt: new Date(now).toISOString(),
      createdAtMs: now,
      updatedAtMs: now,
      validFromMs,
      validToMs,
      expiresAtMs: toMsOrNull(input.expiresAt) ?? MAX_DATE_MS,
      tokens: approxTokens(input.text),
      importance,
      novelty,
    });
    if (input.source !== undefined) metadata.source = String(input.source);
    if (input.confidence !== undefined) {
      if (typeof input.confidence !== 'number' || input.confidence < 0 || input.confidence > 1) invalid('confidence must be in [0,1]');
      metadata.confidence = input.confidence;
    }
    if (input.pinned) metadata.pinned = true;
    if (input.sessionId) metadata.sessionId = input.sessionId;
    if (input.capabilities !== undefined) metadata._caps = String(BigInt(input.capabilities));
    if (input.supersedes) metadata.supersedes = input.supersedes;

    target.upsert({ id, vector, metadata }, this.identity);

    let supersededId: string | undefined;
    if (input.supersedes) {
      const old = target.get(input.supersedes);
      if (!old) throw new MemError('NOT_FOUND', `supersedes target ${input.supersedes} not found`);
      target.updateMetadata(input.supersedes, { supersededBy: id, validToMs: now, updatedAtMs: now });
      supersededId = input.supersedes;
      if (!input.branch) {
        this.ensureMemoryNode(id, kind);
        this.ensureMemoryNode(input.supersedes, String(old.metadata.kind ?? 'fact'));
        this.graph.createEdge({ from: id, to: input.supersedes, type: 'SUPERSEDES', confidence: 1 });
      }
    }
    if (input.relations?.length && !input.branch) {
      this.ensureMemoryNode(id, kind);
      for (const rel of input.relations) {
        if (!rel.to || !rel.type) invalid('relation needs `to` and `type`');
        if (!this.graph.hasNode(rel.to)) {
          const other = this.store.get(rel.to);
          this.graph.createNode({ id: rel.to, labels: other ? ['Memory', capitalize(String(other.metadata.kind ?? 'fact'))] : ['Entity'], properties: other ? { kind: String(other.metadata.kind ?? 'fact') } : {} });
        }
        this.graph.createEdge({ from: id, to: rel.to, type: rel.type, confidence: rel.confidence ?? 1 });
      }
    }
    return { id, kind, importance, novelty, similar, supersededId, embedder: this.embedder.name };
  }

  async recall(input: RecallInput): Promise<RecallResult> {
    const k = input.k ?? 5;
    if (!Number.isInteger(k) || k < 1 || k > LIMITS.maxK) invalid(`k must be an integer in [1, ${LIMITS.maxK}]`);
    const now = Date.now();
    const target = this.target(input.branch);
    const vector = await this.embedQuery(input.text);
    const filter: Filter = { ...(input.filter ?? {}) };
    if (input.kinds?.length) filter.kind = { $in: input.kinds.map((x) => x.toLowerCase()) };
    if (input.tags?.length) filter.tags = { $all: input.tags };
    if (input.sessionId) filter.sessionId = input.sessionId;
    const since = toMsOrNull(input.since), until = toMsOrNull(input.until);
    if (since !== null || until !== null) filter.createdAtMs = { ...(since !== null ? { $gte: since } : {}), ...(until !== null ? { $lte: until } : {}) };
    const asOf = toMsOrNull(input.asOf);
    if (asOf !== null) {
      filter.validFromMs = { $lte: asOf };
      filter.validToMs = { $gt: asOf };
    } else if (!input.includeSuperseded) {
      filter.validToMs = { $gt: now };
    }
    filter.expiresAtMs = { $gt: now };

    const decay = input.decay === true ? { halfLifeDays: 30 } : input.decay && typeof input.decay === 'object' ? input.decay : undefined;
    const hybrid = input.hybrid ?? true;
    const res = target.search({
      vector,
      k: input.budgetTokens ? Math.min(k * 3, LIMITS.maxK) : k,
      filter,
      capabilities: input.capabilities !== undefined ? BigInt(input.capabilities) : undefined,
      decay: decay ? { halfLifeDays: decay.halfLifeDays, now } : undefined,
      fusion: hybrid ? { lexical: input.text, method: 'rrf', alpha: 0.6 } : undefined,
      diversity: input.diversity,
      coherence: input.coherence ? { weight: input.coherence } : undefined,
      explain: input.explain,
      identity: this.identity,
    });
    let hits = res.results.map((r) => this.hit(r));
    if (input.minSimilarity !== undefined) hits = hits.filter((h) => h.similarity >= input.minSimilarity!);
    if (this.learning.enabled) {
      // feedback-informed boost, only when learning is switched on (off => identical to a never-trained store)
      for (const h of hits) {
        const f = this.store.accessStats(h.id)?.f ?? 0;
        if (f) h.finalScore = round(h.finalScore * (1 + 0.15 * Math.tanh(f / 4)));
      }
      hits.sort((a, b) => b.finalScore - a.finalScore);
    }
    if (input.budgetTokens) {
      const packed: RecallHit[] = [];
      let used = 0;
      for (const h of hits) {
        const t = Number(h.metadata.tokens ?? approxTokens(h.text));
        if (used + t > input.budgetTokens) continue;
        packed.push(h);
        used += t;
        if (packed.length >= k) break;
      }
      hits = packed;
    } else hits = hits.slice(0, k);
    if (input.expandGraph && input.expandGraph > 0 && !input.branch) {
      for (const h of hits) {
        if (!this.graph.hasNode(h.id)) continue;
        const related: RecallHit['related'] = [];
        for (const e of this.graph.edgesOf(h.id)) related.push({ id: e.from === h.id ? e.to : e.from, type: e.type, direction: e.from === h.id ? 'out' : 'in', hops: 1 });
        if (input.expandGraph > 1) for (const n of this.graph.kHopNeighbors(h.id, input.expandGraph, { limit: 50 })) if (n.hops > 1) related.push({ id: n.id, type: '*', direction: 'out', hops: n.hops });
        h.related = related.slice(0, 50);
      }
    }
    const queryId = crypto.randomUUID();
    this.pushQuery({ id: queryId, at: now, text: input.text.slice(0, 200), resultIds: hits.map((h) => h.id) });
    return {
      queryId,
      results: hits,
      fetched: res.fetched,
      matched: res.matched,
      complete: res.complete,
      strategy: res.strategy,
      tookMs: res.tookMs,
      embedder: this.embedder.name,
      semantic: this.embedder.semantic,
      warning: this.embedder.semantic ? undefined : 'lexical fallback embedder in use: results are keyword-based, not semantic',
    };
  }

  get(id: string, opts: { branch?: string } = {}): RecallHit | null {
    const r = this.target(opts.branch).get(id);
    return r ? this.hit({ id: r.id, score: 0, similarity: 1, metadata: r.metadata }) : null;
  }

  list(opts: { kinds?: string[]; tags?: string[]; filter?: Filter; limit?: number; offset?: number; includeSuperseded?: boolean; branch?: string } = {}) {
    const filter: Filter = { ...(opts.filter ?? {}) };
    if (opts.kinds?.length) filter.kind = { $in: opts.kinds };
    if (opts.tags?.length) filter.tags = { $all: opts.tags };
    if (!opts.includeSuperseded) filter.validToMs = { $gt: Date.now() };
    const limit = Math.min(opts.limit ?? 50, 1000);
    const offset = opts.offset ?? 0;
    const out: RecallHit[] = [];
    let total = 0;
    for (const r of this.target(opts.branch).iterate(filter)) {
      if (total >= offset && out.length < limit) out.push(this.hit({ id: r.id, score: 0, similarity: 1, metadata: r.metadata }));
      total++;
    }
    return { results: out, total, limit, offset };
  }

  async update(id: string, patch: { text?: string; tags?: string[]; metadata?: Metadata; confidence?: number; importance?: number; pinned?: boolean; validTo?: string | number; branch?: string }): Promise<RecallHit> {
    const target = this.target(patch.branch);
    const cur = target.get(id);
    if (!cur) throw new MemError('NOT_FOUND', `memory ${id} not found`);
    const meta: Metadata = {};
    for (const [k, v] of Object.entries(patch.metadata ?? {})) {
      if (RESERVED.has(k)) invalid(`metadata key ${k} is reserved`);
      meta[k] = v;
    }
    if (patch.tags) meta.tags = [...new Set(patch.tags.map(String))];
    if (patch.confidence !== undefined) meta.confidence = patch.confidence;
    if (patch.importance !== undefined) meta.importance = patch.importance;
    if (patch.pinned !== undefined) meta.pinned = patch.pinned;
    if (patch.validTo !== undefined) meta.validToMs = toMsOrNull(patch.validTo) ?? MAX_DATE_MS;
    meta.updatedAtMs = Date.now();
    if (patch.text !== undefined && patch.text !== cur.metadata.text) {
      const [vector] = await this.embedPassages([patch.text]);
      meta.text = patch.text;
      meta.tokens = approxTokens(patch.text);
      target.upsert({ id, vector, metadata: { ...cur.metadata, ...meta } }, this.identity);
    } else target.updateMetadata(id, meta);
    return this.get(id, { branch: patch.branch })!;
  }

  forget(id: string, opts: { branch?: string } = {}): boolean {
    const ok = this.target(opts.branch).delete(id);
    if (ok && !opts.branch && this.graph.hasNode(id)) this.graph.deleteNode(id);
    return ok;
  }

  // ------------------------------------------------------------------ episodes

  async storeEpisode(e: { task: string; actions?: string[]; observations?: string[]; critique?: string; outcome?: string; reward?: number; tags?: string[]; sessionId?: string; metadata?: Metadata }) {
    if (!e.task?.trim()) invalid('task is required');
    const text = `Task: ${e.task}` + (e.critique ? `\nCritique: ${e.critique}` : '') + (e.outcome ? `\nOutcome: ${e.outcome}` : '');
    return this.remember({
      text,
      kind: 'episode',
      tags: e.tags,
      sessionId: e.sessionId,
      importance: e.reward !== undefined ? Math.max(0, Math.min(1, (e.reward + 1) / 2)) : undefined,
      metadata: { ...(e.metadata ?? {}), task: e.task, actions: e.actions ?? [], observations: e.observations ?? [], critique: e.critique ?? '', outcome: e.outcome ?? null, reward: e.reward ?? null },
    });
  }

  async retrieveEpisodes(text: string, k = 5, opts: { tags?: string[]; since?: string | number } = {}) {
    const r = await this.recall({ text, k, kinds: ['episode'], tags: opts.tags, since: opts.since, hybrid: true });
    // newest first on ties
    r.results.sort((a, b) => b.finalScore - a.finalScore || Number(b.metadata.createdAtMs) - Number(a.metadata.createdAtMs));
    return r;
  }

  // ------------------------------------------------------------------ skills

  async createSkill(s: { name: string; description: string; parameters?: Record<string, unknown>; examples?: string[]; tags?: string[]; id?: string }) {
    if (!s.name?.trim() || !s.description?.trim()) invalid('name and description are required');
    return this.remember({
      id: s.id ?? `skill:${slug(s.name)}`,
      text: `${s.name}: ${s.description}`,
      kind: 'skill',
      tags: s.tags,
      importance: 0.7,
      metadata: { name: s.name, description: s.description, parameters: s.parameters ?? {}, examples: s.examples ?? [], successRate: 0, usageCount: 0, successes: 0 },
    });
  }

  async searchSkills(text: string, k = 5) {
    const r = await this.recall({ text, k, kinds: ['skill'] });
    return { ...r, results: r.results.map((h) => ({ ...h, name: h.metadata.name, description: h.metadata.description, parameters: h.metadata.parameters, examples: h.metadata.examples, successRate: h.metadata.successRate, usageCount: h.metadata.usageCount })) };
  }

  /** The only path that changes a skill's success rate and usage count. */
  recordSkillOutcome(skillId: string, success: boolean) {
    const cur = this.store.get(skillId);
    if (!cur || cur.metadata.kind !== 'skill') throw new MemError('NOT_FOUND', `skill ${skillId} not found`);
    const usageCount = Number(cur.metadata.usageCount ?? 0) + 1;
    const successes = Number(cur.metadata.successes ?? 0) + (success ? 1 : 0);
    const meta = this.store.updateMetadata(skillId, { usageCount, successes, successRate: round(successes / usageCount), updatedAtMs: Date.now() });
    this.store.recordFeedback(skillId, success ? 1 : -1);
    this.witness.append('skill-outcome', skillId, { success, usageCount, successRate: meta.successRate }, this.actor);
    return { id: skillId, usageCount, successRate: meta.successRate as number };
  }

  // ------------------------------------------------------------------ causal

  async addCausalEdge(c: { causes: string[]; effects: string[]; confidence?: number; context?: string; id?: string }) {
    if (!c.causes?.length || !c.effects?.length) invalid('causes and effects must be non-empty');
    const text = `${c.causes.join(' and ')} causes ${c.effects.join(' and ')}` + (c.context ? ` (${c.context})` : '');
    const [embedding] = await this.embedPassages([text]);
    const nodeIds = [...c.causes, ...c.effects].map((s) => this.eventNodeId(s));
    return this.graph.transaction(() => {
      for (let i = 0; i < nodeIds.length; i++) {
        const label = i < c.causes.length ? c.causes[i] : c.effects[i - c.causes.length];
        if (!this.graph.hasNode(nodeIds[i])) this.graph.createNode({ id: nodeIds[i], labels: this.store.has(label) ? ['Memory'] : ['Event'], properties: { text: label } });
      }
      const h = this.graph.createHyperedge({ id: c.id, nodes: nodeIds, type: 'CAUSES', confidence: c.confidence ?? 0.5, properties: { causes: c.causes, effects: c.effects, context: c.context ?? '', text }, embedding });
      return { id: h.id, causes: c.causes, effects: c.effects, confidence: h.confidence, context: c.context ?? '' };
    });
  }

  private eventNodeId(s: string): string {
    return this.store.has(s) || this.graph.hasNode(s) ? s : `event:${slug(s)}`;
  }

  async queryCausal(text: string, k = 5) {
    const q = await this.embedQuery(text);
    const hits = this.graph.searchHyperedges(q, Math.min(k * 4, 200), { types: ['CAUSES'] });
    return hits
      .map((h) => ({ id: h.hyperedge.id, causes: h.hyperedge.properties.causes, effects: h.hyperedge.properties.effects, context: h.hyperedge.properties.context, confidence: h.hyperedge.confidence, similarity: round(h.similarity), score: round(h.similarity * h.hyperedge.confidence) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }

  /** Outcome feedback for a causal edge: moves confidence toward 1 (confirmed) or 0 (refuted). */
  recordCausalOutcome(id: string, confirmed: boolean, rate = 0.2): GHyperedge {
    const h = this.graph.getHyperedge(id);
    if (!h) throw new MemError('NOT_FOUND', `causal edge ${id} not found`);
    const next = round(h.confidence + rate * ((confirmed ? 1 : 0) - h.confidence));
    const out = this.graph.updateHyperedge(id, { confidence: next });
    this.witness.append('causal-outcome', id, { confirmed, confidence: next }, this.actor);
    return out;
  }

  // ------------------------------------------------------------------ sessions (working memory)

  startSession(opts: { id?: string; ttlMinutes?: number; metadata?: Metadata } = {}) {
    const ttlMinutes = opts.ttlMinutes ?? 240;
    if (!(ttlMinutes > 0)) invalid('ttlMinutes must be positive');
    const ttlMs = Math.max(1, Math.round(ttlMinutes * 60_000));
    const id = opts.id ?? crypto.randomUUID();
    return this.withSessions((s) => {
      this.expireIn(s);
      if (s[id]) throw new MemError('ALREADY_EXISTS', `session ${id} exists`);
      const now = Date.now();
      s[id] = { id, startedAt: new Date(now).toISOString(), expiresAtMs: now + ttlMs, ttlMs, metadata: opts.metadata ?? {}, turns: [] };
      this.witness.append('session-start', id, { ttlMs }, this.actor);
      return { id, expiresAt: new Date(now + ttlMs).toISOString(), ttlMinutes: ttlMs / 60_000 };
    });
  }

  async appendTurn(t: { sessionId: string; role: string; content: string; toolCalls?: unknown[]; remember?: boolean }) {
    const now = Date.now();
    const turnId = crypto.randomUUID();
    const sess = this.withSessions((s) => {
      this.expireIn(s);
      const sess = s[t.sessionId];
      if (!sess) throw new MemError('NOT_FOUND', `session ${t.sessionId} not found or expired`);
      sess.turns.push({ id: turnId, role: t.role, content: t.content, toolCalls: t.toolCalls, createdAt: new Date(now).toISOString() });
      sess.expiresAtMs = now + sess.ttlMs;
      return { ...sess };
    });
    let rememberedAs: string | undefined;
    if (t.remember) {
      const r = await this.remember({ id: `turn:${turnId}`, text: `${t.role}: ${t.content}`, kind: 'turn', sessionId: t.sessionId, expiresAt: sess.expiresAtMs, metadata: { role: t.role, turnId } });
      rememberedAs = r.id;
      this.withSessions((s) => {
        const turn = s[t.sessionId]?.turns.find((x) => x.id === turnId);
        if (turn) turn.rememberedAs = rememberedAs;
      });
    }
    return { sessionId: t.sessionId, turnId, turns: sess.turns.length + 0, expiresAt: new Date(sess.expiresAtMs).toISOString(), rememberedAs };
  }

  getSession(id: string, opts: { lastN?: number } = {}) {
    return this.withSessions((s) => {
      this.expireIn(s);
      const sess = s[id];
      if (!sess) return null;
      const turns = opts.lastN ? sess.turns.slice(-opts.lastN) : sess.turns;
      return { ...sess, turns, expiresAt: new Date(sess.expiresAtMs).toISOString() };
    });
  }

  listSessions() {
    return this.withSessions((s) => {
      this.expireIn(s);
      return Object.values(s).map((x) => ({ id: x.id, startedAt: x.startedAt, expiresAt: new Date(x.expiresAtMs).toISOString(), turns: x.turns.length, metadata: x.metadata }));
    });
  }

  endSession(id: string, opts: { summary?: string; tags?: string[] } = {}) {
    const sess = this.withSessions((s) => {
      const x = s[id];
      if (!x) throw new MemError('NOT_FOUND', `session ${id} not found`);
      delete s[id];
      return x;
    });
    this.witness.append('session-end', id, { turns: sess.turns.length }, this.actor);
    const dropped = this.dropTurnRecords(id);
    const result: { id: string; turns: number; droppedTurnRecords: number; summaryId?: string } = { id, turns: sess.turns.length, droppedTurnRecords: dropped };
    if (opts.summary) {
      // caller-provided summary becomes a durable episode
      return this.storeEpisode({ task: `Session ${id}`, critique: opts.summary, tags: opts.tags, sessionId: id, actions: sess.turns.filter((t) => t.role !== 'user').map((t) => t.content.slice(0, 200)) }).then((r) => ({ ...result, summaryId: r.id }));
    }
    return Promise.resolve(result);
  }

  private expireIn(s: Record<string, Session>) {
    const now = Date.now();
    for (const [id, sess] of Object.entries(s)) {
      if (sess.expiresAtMs <= now) {
        delete s[id];
        this.dropTurnRecords(id);
      }
    }
  }

  private dropTurnRecords(sessionId: string): number {
    const ids: string[] = [];
    for (const r of this.store.iterate({ kind: 'turn', sessionId })) ids.push(r.id);
    if (ids.length) this.store.deleteBatch(ids, 'expire');
    return ids.length;
  }

  expireSessions() {
    const before = Object.keys(this.loadSessions()).length;
    const after = this.withSessions((s) => {
      this.expireIn(s);
      return Object.keys(s).length;
    });
    // expired turn records whose session is long gone
    const now = Date.now();
    const stale: string[] = [];
    for (const r of this.store.iterate({ expiresAtMs: { $lte: now } })) stale.push(r.id);
    if (stale.length) this.store.deleteBatch(stale, 'expire');
    return { expiredSessions: before - after, expiredRecords: stale.length };
  }

  // ------------------------------------------------------------------ learning / feedback

  recordOutcome(o: { queryId?: string; resultIds?: string[]; chosenIds: string[]; reward: number; context?: { stateKey?: string; action?: string } }) {
    if (typeof o.reward !== 'number' || o.reward < -1 || o.reward > 1) invalid('reward must be in [-1, 1]');
    const q = o.queryId ? this.loadQueries().find((x) => x.id === o.queryId) : undefined;
    const shown = new Set(o.resultIds ?? q?.resultIds ?? []);
    const applied: string[] = [];
    for (const id of o.chosenIds ?? []) {
      if (!this.store.has(id)) continue;
      this.store.recordFeedback(id, o.reward);
      applied.push(id);
    }
    for (const id of shown) if (!o.chosenIds.includes(id) && this.store.has(id) && o.reward > 0) this.store.recordFeedback(id, -0.25 * o.reward);
    let routing: { stateKey: string; action: string; value: number } | undefined;
    if (o.context?.stateKey && o.context.action) {
      const st = (this.learning.policy[o.context.stateKey] ??= {});
      const a = (st[o.context.action] ??= { value: 0, n: 0 });
      a.n++;
      a.value = round(a.value + (1 / Math.min(a.n, 20)) * (o.reward - a.value));
      routing = { stateKey: o.context.stateKey, action: o.context.action, value: a.value };
      this.saveLearning();
    }
    const entry = this.witness.append('outcome', o.queryId ?? 'adhoc', { chosenIds: applied, reward: o.reward, context: o.context ?? null }, this.actor);
    this.store.flush();
    return { applied, routing, witnessSeq: entry.seq, learningEnabled: this.learning.enabled };
  }

  /** Action values for a state key, with epsilon-greedy recommendation. */
  recommendAction(stateKey: string, candidates: string[], epsilon = 0.1) {
    const st = this.learning.policy[stateKey] ?? {};
    const values = candidates.map((a) => ({ action: a, value: st[a]?.value ?? 0, n: st[a]?.n ?? 0 }));
    const explore = Math.random() < epsilon;
    const pick = explore ? values[Math.floor(Math.random() * values.length)] : values.reduce((b, v) => (v.value > b.value ? v : b), values[0]);
    return { stateKey, recommended: pick?.action ?? null, explored: explore, values };
  }

  setLearning(enabled: boolean) {
    this.learning.enabled = enabled;
    this.saveLearning();
    this.witness.append('learning-toggle', 'learning', { enabled }, this.actor);
    return { enabled };
  }

  resetLearning(scope: 'routing' | 'feedback' | 'all' = 'all') {
    if (scope === 'routing' || scope === 'all') this.learning.policy = {};
    if (scope === 'feedback' || scope === 'all') for (const r of this.store.iterate()) this.store.recordFeedback(r.id, -(this.store.accessStats(r.id)?.f ?? 0));
    this.saveLearning();
    this.store.flush();
    this.witness.append('learning-reset', scope, null, this.actor);
    return { scope };
  }

  // ------------------------------------------------------------------ consolidation

  /**
   * Promote repeated episodes into a durable procedure: cluster episodes by embedding, require a minimum
   * cluster size and success rate, write the distilled record and link it to its sources.
   * Without a language model the distilled text is the medoid episode plus cluster statistics; use
   * `dryRun` to get the clusters and write a better summary with `remember` + `derivedFrom`.
   */
  async consolidate(opts: { minClusterSize?: number; similarityThreshold?: number; minSuccessRate?: number; kinds?: string[]; dryRun?: boolean; maxClusters?: number } = {}) {
    const minSize = opts.minClusterSize ?? 3;
    const thr = opts.similarityThreshold ?? 0.75;
    const minSuccess = opts.minSuccessRate ?? 0.6;
    const kinds = opts.kinds ?? ['episode'];
    const items: MemRecord[] = [];
    for (const r of this.store.iterate({ kind: { $in: kinds }, consolidatedInto: { $exists: false }, validToMs: { $gt: Date.now() } })) items.push(r);
    items.sort((a, b) => Number(a.metadata.createdAtMs) - Number(b.metadata.createdAtMs));
    const clusters: Array<{ members: MemRecord[]; centroid: Float32Array }> = [];
    for (const it of items) {
      let best = -1, bestSim = thr;
      for (let i = 0; i < clusters.length; i++) {
        const s = 1 - normalizedCosineDistance(it.vector, clusters[i].centroid);
        if (s >= bestSim) {
          bestSim = s;
          best = i;
        }
      }
      if (best < 0) clusters.push({ members: [it], centroid: Float32Array.from(it.vector) });
      else {
        const c = clusters[best];
        c.members.push(it);
        const n = c.members.length;
        for (let d = 0; d < c.centroid.length; d++) c.centroid[d] = (c.centroid[d] * (n - 1) + it.vector[d]) / n;
        let norm = 0;
        for (let d = 0; d < c.centroid.length; d++) norm += c.centroid[d] ** 2;
        norm = Math.sqrt(norm) || 1;
        for (let d = 0; d < c.centroid.length; d++) c.centroid[d] /= norm;
      }
    }
    const proposals = clusters
      .filter((c) => c.members.length >= minSize)
      .map((c) => {
        const successes = c.members.filter((m) => m.metadata.outcome === 'success' || (typeof m.metadata.reward === 'number' && m.metadata.reward > 0)).length;
        const rated = c.members.filter((m) => m.metadata.outcome !== null || typeof m.metadata.reward === 'number').length;
        const successRate = rated ? successes / rated : 1;
        let medoid = c.members[0], bestMean = -Infinity;
        for (const m of c.members) {
          let s = 0;
          for (const o of c.members) s += 1 - normalizedCosineDistance(m.vector, o.vector);
          if (s > bestMean) {
            bestMean = s;
            medoid = m;
          }
        }
        const tags = [...new Set(c.members.flatMap((m) => (m.metadata.tags as string[]) ?? []))];
        return { memberIds: c.members.map((m) => m.id), size: c.members.length, successRate: round(successRate), eligible: successRate >= minSuccess, medoidId: medoid.id, medoidText: String(medoid.metadata.text), tags, critiques: c.members.map((m) => String(m.metadata.critique ?? '')).filter(Boolean).slice(0, 10) };
      })
      .slice(0, opts.maxClusters ?? 20);
    if (opts.dryRun) return { dryRun: true, examined: items.length, clusters: proposals, created: [] };
    const created: Array<{ id: string; derivedFrom: string[] }> = [];
    for (const p of proposals) {
      if (!p.eligible) continue;
      const text = `Procedure distilled from ${p.size} episodes (success rate ${p.successRate}): ${p.medoidText}`;
      const r = await this.remember({ text, kind: 'procedure', tags: p.tags, importance: 0.8, source: 'consolidation', metadata: { derivedFrom: p.memberIds, clusterSize: p.size, successRate: p.successRate } });
      this.ensureMemoryNode(r.id, 'procedure');
      this.graph.transaction(() => {
        for (const mid of p.memberIds) {
          this.ensureMemoryNode(mid, 'episode');
          this.graph.createEdge({ from: r.id, to: mid, type: 'DERIVED_FROM', confidence: 1 });
        }
      });
      for (const mid of p.memberIds) this.store.updateMetadata(mid, { consolidatedInto: r.id });
      created.push({ id: r.id, derivedFrom: p.memberIds });
    }
    this.witness.append('consolidate', 'memory', { created: created.map((c) => c.id) }, this.actor);
    return { dryRun: false, examined: items.length, clusters: proposals, created };
  }

  // ------------------------------------------------------------------ lifecycle

  compact(opts: { target?: number; policy?: string } = {}): CompactResult {
    return this.store.compact(opts);
  }

  snapshot(file?: string) {
    const dir = path.join(this.dir, 'snapshots');
    fs.mkdirSync(dir, { recursive: true });
    const target = file ? path.resolve(file) : path.join(dir, `${this.namespace}-${new Date().toISOString().replace(/[:.]/g, '-')}.snap`);
    this.store.flush();
    this.graph.flush();
    const records = [...this.store.iterate()].map((r) => ({ id: r.id, v: f32ToBase64(r.vector), m: r.metadata }));
    const body = {
      namespace: this.namespace,
      manifest: this.store.manifest,
      records,
      graph: this.graph.export(),
      sessions: this.loadSessions(),
      learning: this.learning,
      state: this.store.exportState(),
      witnessHead: this.witness.headInfo(),
    };
    const { bytes, checksum } = writeSnapshot(target, 'agent-memory-namespace', body);
    this.witness.append('snapshot', path.basename(target), { checksum, records: records.length }, this.actor);
    return { file: target, bytes, checksum, records: records.length, graph: { nodes: body.graph.nodes.length, edges: body.graph.edges.length, hyperedges: body.graph.hyperedges.length } };
  }

  /** Restore a snapshot into a namespace directory (static: the namespace must not be open). */
  static restore(file: string, root: string, opts: { namespace?: string; overwrite?: boolean } = {}) {
    type Body = { namespace: string; manifest: Collection['manifest']; records: Array<{ id: string; v: string; m: Metadata }>; graph: ReturnType<GraphStore['export']>; sessions: Record<string, Session>; learning: LearningState; state: ReturnType<Collection['exportState']>; witnessHead: { seq: number; hash: string } };
    const snap = readSnapshot<Body>(file, 'agent-memory-namespace');
    const ns = opts.namespace ?? snap.body.namespace;
    if (!NAMESPACE_RE.test(ns)) throw new MemError('PATH_UNSAFE', `invalid namespace ${ns}`);
    const dir = confinePath(path.resolve(root), ns);
    if (fs.existsSync(dir)) {
      if (!opts.overwrite) throw new MemError('ALREADY_EXISTS', `namespace ${ns} exists; pass overwrite to replace it`);
      fs.rmSync(dir, { recursive: true, force: true });
    }
    fs.mkdirSync(dir, { recursive: true });
    const witness = new WitnessLog(dir);
    const col = Collection.create(path.join(dir, 'memory'), snap.body.manifest.options, { witness, actor: 'restore', fallbackEmbedder: snap.body.manifest.fallbackEmbedder });
    const recs = snap.body.records.map((r) => ({ id: r.id, vector: base64ToF32(r.v), metadata: r.m }));
    for (let i = 0; i < recs.length; i += 1000) col.insertBatch(recs.slice(i, i + 1000));
    col.importState(snap.body.state ?? {});
    col.close();
    const graph = GraphStore.openOrCreate(path.join(dir, 'graph'), { dimensions: snap.body.manifest.options.dimensions }, { witness, actor: 'restore' });
    graph.import(snap.body.graph);
    graph.close();
    writeFileAtomic(path.join(dir, 'sessions.json'), JSON.stringify(snap.body.sessions ?? {}));
    writeFileAtomic(path.join(dir, 'learning.json'), JSON.stringify(snap.body.learning ?? { enabled: false, policy: {} }));
    witness.append('restore', path.basename(file), { snapshotChecksum: snap.checksum, snapshotWitnessHead: snap.body.witnessHead }, 'restore');
    return { namespace: ns, dir, records: recs.length, snapshotCreatedAt: snap.createdAt, checksum: snap.checksum };
  }

  listSnapshots() {
    const dir = path.join(this.dir, 'snapshots');
    if (!fs.existsSync(dir)) return [];
    return fs
      .readdirSync(dir)
      .filter((f) => f.endsWith('.snap'))
      .map((f) => ({ file: path.join(dir, f), bytes: fs.statSync(path.join(dir, f)).size, mtime: fs.statSync(path.join(dir, f)).mtime.toISOString() }));
  }

  async verify() {
    const w = await this.witness.verify();
    return { witness: w, records: this.store.count(), graph: this.graph.stats() };
  }

  // branches
  createBranch(name: string) {
    if (!NAMESPACE_RE.test(name)) throw new MemError('PATH_UNSAFE', `invalid branch name ${name}`);
    const b = Branch.create(this.store, path.join(this.dir, 'branches', name), name, { witness: this.witness, actor: this.actor });
    this.branches.set(name, b);
    this.witness.append('branch-create', name, null, this.actor);
    return b.info();
  }
  listBranches() {
    return Branch.list(path.join(this.dir, 'branches')).map((n) => this.openBranch(n).info());
  }
  mergeBranch(name: string, opts: { force?: boolean; discard?: boolean } = {}) {
    const b = this.openBranch(name);
    const r = b.merge({ force: opts.force });
    this.witness.append('branch-merge', name, r, this.actor);
    if (!r.conflicts.length && opts.discard !== false) this.discardBranch(name);
    return r;
  }
  discardBranch(name: string) {
    const b = this.openBranch(name);
    b.discard();
    this.branches.delete(name);
    this.witness.append('branch-discard', name, null, this.actor);
    return { discarded: name };
  }

  /** Remove a record from the live store and every copy this engine knows about (snapshots in the namespace, branches). */
  purge(id: string, opts: { everywhere?: boolean } = {}) {
    const removedFrom: string[] = [];
    const unreachable: string[] = [];
    if (this.store.delete(id, 'purge')) removedFrom.push('memory');
    if (this.graph.hasNode(id)) {
      this.graph.deleteNode(id);
      removedFrom.push('graph');
    }
    if (opts.everywhere) {
      for (const name of Branch.list(path.join(this.dir, 'branches'))) {
        const b = this.openBranch(name);
        if (b.overlay.has(id)) {
          b.overlay.delete(id, 'purge');
          removedFrom.push(`branch:${name}`);
        }
      }
      for (const s of this.listSnapshots()) {
        try {
          const snap = readSnapshot<{ records: Array<{ id: string }>; graph: { nodes: Array<{ id: string }>; edges: Array<{ from: string; to: string }>; hyperedges: Array<{ nodes: string[] }> } }>(s.file, 'agent-memory-namespace');
          const before = snap.body.records.length;
          snap.body.records = snap.body.records.filter((r) => r.id !== id);
          snap.body.graph.nodes = snap.body.graph.nodes.filter((n) => n.id !== id);
          snap.body.graph.edges = snap.body.graph.edges.filter((e) => e.from !== id && e.to !== id);
          snap.body.graph.hyperedges = snap.body.graph.hyperedges.filter((h) => !h.nodes.includes(id));
          if (snap.body.records.length !== before) {
            writeSnapshot(s.file, 'agent-memory-namespace', snap.body);
            removedFrom.push(`snapshot:${path.basename(s.file)}`);
          }
        } catch (e: any) {
          unreachable.push(`snapshot:${path.basename(s.file)} (${e?.message ?? e})`);
        }
      }
      unreachable.push('snapshots stored outside the namespace, exports, and other hosts are not tracked by this engine');
    }
    this.witness.append('purge', id, { everywhere: !!opts.everywhere, removedFrom }, this.actor);
    return { id, removedFrom, unreachable };
  }

  /** Re-embed every record with another embedder into a fresh collection and swap it in. */
  async reembed(to: EmbeddingProvider, opts: { batch?: number } = {}) {
    if (!to.isReady()) await to.init();
    const ident = to.identity();
    const tmpDir = path.join(this.dir, `memory.reembed.${Date.now()}`);
    const next = Collection.create(tmpDir, { ...this.store.options, dimensions: ident.dimension, embedding: ident }, { witness: this.witness, actor: this.actor, fallbackEmbedder: !to.semantic });
    const batch = opts.batch ?? 64;
    const recs = [...this.store.iterate()];
    for (let i = 0; i < recs.length; i += batch) {
      const chunk = recs.slice(i, i + batch);
      const vecs = await to.embedBatch(chunk.map((r) => String(r.metadata[this.store.options.textKey] ?? '')), 'passage');
      next.insertBatch(chunk.map((r, j) => ({ id: r.id, vector: vecs[j], metadata: r.metadata })));
    }
    next.close();
    this.store.close();
    const memDir = path.join(this.dir, 'memory');
    const oldDir = path.join(this.dir, `memory.old.${Date.now()}`);
    fs.renameSync(memDir, oldDir);
    fs.renameSync(tmpDir, memDir);
    // graph embeddings must move to the new space too
    const graphExport = this.graph.export();
    this.graph.close();
    const graphOld = path.join(this.dir, `graph.old.${Date.now()}`);
    fs.renameSync(path.join(this.dir, 'graph'), graphOld);
    const g = GraphStore.openOrCreate(path.join(this.dir, 'graph'), { dimensions: ident.dimension }, { witness: this.witness, actor: this.actor });
    const reembedEl = async <T extends { e?: string; properties: Record<string, unknown> }>(items: T[]) => {
      const withText = items.filter((x) => x.e && typeof x.properties.text === 'string');
      const vecs = await to.embedBatch(withText.map((x) => String(x.properties.text)), 'passage');
      const map = new Map(withText.map((x, i) => [x, vecs[i]]));
      return items.map((x) => {
        const { e, ...rest } = x;
        const v = map.get(x);
        return v ? { ...rest, e: f32ToBase64(v) } : rest;
      });
    };
    g.import({ nodes: (await reembedEl(graphExport.nodes)) as typeof graphExport.nodes, edges: (await reembedEl(graphExport.edges)) as typeof graphExport.edges, hyperedges: (await reembedEl(graphExport.hyperedges)) as typeof graphExport.hyperedges });
    g.close();
    this.witness.append('reembed', 'memory', { from: this.identity, to: ident, records: recs.length }, this.actor);
    return { records: recs.length, from: this.identity.modelId, to: ident.modelId, previousCollection: oldDir, note: 'reopen the namespace with the new embedder to use it' };
  }

  stats() {
    const s = this.store.stats();
    const kinds: Record<string, number> = {};
    for (const r of this.store.iterate()) kinds[String(r.metadata.kind ?? 'fact')] = (kinds[String(r.metadata.kind ?? 'fact')] ?? 0) + 1;
    return {
      namespace: this.namespace,
      dir: this.dir,
      records: s.count,
      kinds,
      tombstones: s.tombstones,
      generation: s.generation,
      index: s.options.index,
      capacity: s.options.capacity,
      compactionPolicy: s.options.compactionPolicy,
      embedding: s.embedding,
      embedder: this.embedder.name,
      semantic: this.embedder.semantic,
      fallbackActivated: this.fallbackActivated,
      graph: this.graph.stats(),
      sessions: Object.keys(this.loadSessions()).length,
      learning: { enabled: this.learning.enabled, stateKeys: Object.keys(this.learning.policy).length },
      witnessHead: this.witness.headInfo(),
      branches: Branch.list(path.join(this.dir, 'branches')),
      features: this.store.manifest.features,
    };
  }

  close() {
    if (this.closed) return;
    this.closed = true;
    for (const b of this.branches.values()) b.close();
    this.store.close();
    this.graph.close();
  }
}

function round(x: number): number {
  return Math.round(x * 10000) / 10000;
}
function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 80) || crypto.randomUUID();
}
function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export { similarityFor };
