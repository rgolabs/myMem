/**
 * Copy-on-write branch of a collection (spec §10). Reads fall through to the parent until a record is
 * written (or deleted) in the branch. `merge` applies branch writes to the parent and reports conflicts
 * by record id; `discard` removes the branch directory.
 */
import fs from 'node:fs';
import path from 'node:path';
import { Collection } from './vector-store.js';
import { MemError } from './errors.js';
import { readJsonIfExists, writeFileAtomic } from './fsutil.js';
import type { EmbeddingSpaceIdentity, Filter, MemRecord, Metadata, SearchQuery, SearchResponse, SearchResult } from './types.js';
import type { WitnessLog } from './witness.js';

interface BranchMeta {
  name: string;
  createdAt: string;
  parentGeneration: number;
  deleted: string[];
}

export interface MergeResult {
  applied: { upserted: number; deleted: number };
  conflicts: Array<{ id: string; reason: string }>;
}

export class Branch {
  readonly name: string;
  private meta: BranchMeta;
  private metaFile: string;
  private tombstones: Set<string>;

  private constructor(
    readonly parent: Collection,
    readonly overlay: Collection,
    readonly dir: string,
    meta: BranchMeta,
  ) {
    this.name = meta.name;
    this.meta = meta;
    this.metaFile = path.join(dir, 'branch.json');
    this.tombstones = new Set(meta.deleted);
  }

  static create(parent: Collection, dir: string, name: string, extra: { witness?: WitnessLog; actor?: string } = {}): Branch {
    if (fs.existsSync(path.join(dir, 'branch.json'))) throw new MemError('ALREADY_EXISTS', `branch ${name} already exists`);
    fs.mkdirSync(dir, { recursive: true });
    const overlay = Collection.create(path.join(dir, 'memory'), { ...parent.options, capacity: 0 }, { ...extra, fallbackEmbedder: parent.manifest.fallbackEmbedder });
    const meta: BranchMeta = { name, createdAt: new Date().toISOString(), parentGeneration: parent.currentGeneration, deleted: [] };
    writeFileAtomic(path.join(dir, 'branch.json'), JSON.stringify(meta, null, 2));
    return new Branch(parent, overlay, dir, meta);
  }

  static open(parent: Collection, dir: string, extra: { witness?: WitnessLog; actor?: string } = {}): Branch {
    const meta = readJsonIfExists<BranchMeta>(path.join(dir, 'branch.json'));
    if (!meta) throw new MemError('NOT_FOUND', `no branch at ${dir}`);
    const overlay = Collection.open(path.join(dir, 'memory'), extra);
    return new Branch(parent, overlay, dir, meta);
  }

  static list(branchesDir: string): string[] {
    if (!fs.existsSync(branchesDir)) return [];
    return fs.readdirSync(branchesDir).filter((n) => fs.existsSync(path.join(branchesDir, n, 'branch.json')));
  }

  private saveMeta() {
    this.meta.deleted = [...this.tombstones];
    writeFileAtomic(this.metaFile, JSON.stringify(this.meta, null, 2));
  }

  get(id: string): MemRecord | null {
    const o = this.overlay.get(id);
    if (o) return o;
    if (this.tombstones.has(id)) return null;
    return this.parent.get(id);
  }

  has(id: string): boolean {
    return this.get(id) !== null;
  }

  count(): number {
    let n = this.overlay.count();
    for (const r of this.parent.iterate()) if (!this.tombstones.has(r.id) && !this.overlay.has(r.id)) n++;
    return n;
  }

  upsert(r: { id: string; vector: Float32Array | number[]; metadata?: Metadata }, identity?: EmbeddingSpaceIdentity): string {
    const id = this.overlay.upsert(r, identity);
    if (this.tombstones.delete(id)) this.saveMeta();
    return id;
  }

  updateMetadata(id: string, patch: Metadata): Metadata {
    const cur = this.get(id);
    if (!cur) throw new MemError('NOT_FOUND', `record ${id} not found`);
    const merged = { ...cur.metadata, ...patch };
    for (const k of Object.keys(patch)) if (patch[k] === undefined || patch[k] === null) delete merged[k];
    this.overlay.upsert({ id, vector: cur.vector, metadata: merged });
    return merged;
  }

  delete(id: string): boolean {
    let found = false;
    if (this.overlay.has(id)) found = this.overlay.delete(id);
    if (this.parent.has(id)) {
      found = true;
      if (!this.tombstones.has(id)) {
        this.tombstones.add(id);
        this.saveMeta();
      }
    }
    return found;
  }

  *iterate(filter?: Filter): Generator<MemRecord> {
    yield* this.overlay.iterate(filter);
    for (const r of this.parent.iterate(filter)) if (!this.tombstones.has(r.id) && !this.overlay.has(r.id)) yield r;
  }

  search(q: SearchQuery): SearchResponse {
    const t0 = performance.now();
    const a = this.overlay.search(q);
    const b = this.parent.search({ ...q, k: Math.min(q.k * 2, 1000) });
    const seen = new Set(a.results.map((r) => r.id));
    const merged: SearchResult[] = [...a.results];
    for (const r of b.results) {
      if (seen.has(r.id) || this.tombstones.has(r.id)) continue;
      merged.push(r);
    }
    const key = (r: SearchResult) => r.finalScore ?? r.similarity;
    merged.sort((x, y) => key(y) - key(x));
    const results = merged.slice(0, q.k);
    return {
      results,
      fetched: a.fetched + b.fetched,
      matched: a.matched + b.matched,
      complete: results.length >= q.k || (a.complete && b.complete),
      strategy: b.strategy,
      tookMs: Math.round((performance.now() - t0) * 1000) / 1000,
    };
  }

  /** Apply branch writes to the parent. A record modified in the parent after the branch was created is a conflict. */
  merge(opts: { force?: boolean } = {}): MergeResult {
    const conflicts: MergeResult['conflicts'] = [];
    const toUpsert: MemRecord[] = [];
    for (const r of this.overlay.iterate()) {
      const pg = this.parent.modificationGeneration(r.id);
      if (!opts.force && pg > this.meta.parentGeneration) conflicts.push({ id: r.id, reason: `modified in parent at generation ${pg} (branch base ${this.meta.parentGeneration})` });
      else toUpsert.push(r);
    }
    const toDelete: string[] = [];
    for (const id of this.tombstones) {
      const pg = this.parent.modificationGeneration(id);
      if (!this.parent.has(id)) continue;
      if (!opts.force && pg > this.meta.parentGeneration) conflicts.push({ id, reason: `deleted in branch but modified in parent at generation ${pg}` });
      else toDelete.push(id);
    }
    if (conflicts.length && !opts.force) return { applied: { upserted: 0, deleted: 0 }, conflicts };
    for (let i = 0; i < toUpsert.length; i += 1000) this.parent.upsertBatch(toUpsert.slice(i, i + 1000));
    if (toDelete.length) this.parent.deleteBatch(toDelete, 'branch-merge-delete');
    return { applied: { upserted: toUpsert.length, deleted: toDelete.length }, conflicts };
  }

  discard(): void {
    this.overlay.close();
    fs.rmSync(this.dir, { recursive: true, force: true });
  }

  close() {
    this.overlay.close();
  }

  info() {
    return { name: this.name, createdAt: this.meta.createdAt, parentGeneration: this.meta.parentGeneration, overlayRecords: this.overlay.count(), tombstones: this.tombstones.size };
  }
}
