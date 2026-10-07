/**
 * Hierarchical Navigable Small World index (Malkov & Yashunin 2018).
 *
 * - Vectors live outside the index (the collection owns them); the index holds the graph only.
 * - `search` accepts a predicate that is evaluated during traversal: non-matching nodes are still
 *   walked through (so connectivity is preserved) but never enter the result set. This is the
 *   "predicate-aware traversal" the spec requires for selective filters.
 * - Deletions are tombstones; the collection rebuilds the index when tombstones exceed a ratio.
 * - `serialize`/`deserialize` persist the graph so a reopened collection does not rebuild it.
 */
import { Heap } from './heap.js';
import type { DistanceFn } from './distance.js';

export interface HnswParams {
  m: number;
  efConstruction: number;
  seed?: number;
}

export interface HnswSerialized {
  m: number;
  efConstruction: number;
  maxLevel: number;
  entryPoint: number;
  count: number;
  levels: number[];
  /** links[node][level] = neighbour ids */
  links: number[][][];
  deleted: number[];
  seedState: number;
}

export type Predicate = (internalId: number) => boolean;

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class HNSW {
  readonly m: number;
  readonly mMax0: number;
  readonly efConstruction: number;
  private readonly mL: number;
  private rand: () => number;
  private seedState: number;

  private levels: number[] = [];
  private links: number[][][] = [];
  private deleted: Uint8Array = new Uint8Array(64);
  private visited: Uint32Array = new Uint32Array(64);
  private epoch = 0;
  private entryPoint = -1;
  private maxLevel = -1;
  private liveCount = 0;

  constructor(
    params: HnswParams,
    private readonly getVector: (id: number) => Float32Array,
    private readonly dist: DistanceFn,
  ) {
    this.m = Math.max(2, params.m | 0);
    this.mMax0 = this.m * 2;
    this.efConstruction = Math.max(this.m, params.efConstruction | 0);
    this.mL = 1 / Math.log(this.m);
    this.seedState = params.seed ?? 0x9e3779b9;
    this.rand = mulberry32(this.seedState);
  }

  get size() {
    return this.levels.length;
  }
  get live() {
    return this.liveCount;
  }
  get tombstones() {
    return this.levels.length - this.liveCount;
  }
  has(id: number) {
    return id < this.levels.length && this.levels[id] >= 0;
  }
  isDeleted(id: number) {
    return id < this.deleted.length && this.deleted[id] === 1;
  }

  private ensureCapacity(id: number) {
    if (id >= this.deleted.length) {
      const n = Math.max(id + 1, this.deleted.length * 2);
      const d = new Uint8Array(n);
      d.set(this.deleted);
      this.deleted = d;
      const v = new Uint32Array(n);
      v.set(this.visited);
      this.visited = v;
    }
    while (this.levels.length <= id) {
      this.levels.push(-1);
      this.links.push([]);
    }
  }

  private nextEpoch(): number {
    this.epoch++;
    if (this.epoch === 0xffffffff) {
      this.visited.fill(0);
      this.epoch = 1;
    }
    return this.epoch;
  }

  private randomLevel(): number {
    let r = this.rand();
    if (r === 0) r = 1e-12;
    return Math.floor(-Math.log(r) * this.mL);
  }

  /** Greedy descent on upper layers: returns the closest node to q reachable from ep on `level`. */
  private greedy(q: Float32Array, ep: number, level: number): number {
    let cur = ep;
    let curD = this.dist(q, this.getVector(cur));
    for (;;) {
      let changed = false;
      const nbrs = this.links[cur][level];
      if (nbrs) {
        for (let i = 0; i < nbrs.length; i++) {
          const nb = nbrs[i];
          const d = this.dist(q, this.getVector(nb));
          if (d < curD) {
            curD = d;
            cur = nb;
            changed = true;
          }
        }
      }
      if (!changed) return cur;
    }
  }

  /** Beam search on one layer. Returns a max-heap (farthest on top) of at most ef accepted nodes. */
  private searchLayer(q: Float32Array, eps: number[], ef: number, level: number, allow?: Predicate): Heap<number> {
    const epoch = this.nextEpoch();
    const visited = this.visited;
    const candidates = new Heap<number>(false);
    const results = new Heap<number>(true);
    for (let i = 0; i < eps.length; i++) {
      const ep = eps[i];
      if (visited[ep] === epoch) continue;
      visited[ep] = epoch;
      const d = this.dist(q, this.getVector(ep));
      candidates.push(d, ep);
      if (this.accept(ep, allow)) results.push(d, ep);
    }
    while (candidates.size) {
      const cd = candidates.peekKey();
      if (results.size >= ef && cd > results.peekKey()) break;
      const c = candidates.pop()[1];
      const nbrs = this.links[c][level];
      if (!nbrs) continue;
      for (let i = 0; i < nbrs.length; i++) {
        const nb = nbrs[i];
        if (visited[nb] === epoch) continue;
        visited[nb] = epoch;
        const nd = this.dist(q, this.getVector(nb));
        if (results.size < ef || nd < results.peekKey()) {
          candidates.push(nd, nb);
          if (this.accept(nb, allow)) {
            results.push(nd, nb);
            if (results.size > ef) results.pop();
          }
        }
      }
    }
    return results;
  }

  private accept(id: number, allow?: Predicate) {
    if (this.deleted[id] === 1) return false;
    return allow ? allow(id) : true;
  }

  /** Heuristic neighbour selection (paper Algorithm 4) with pruned-candidate backfill. ids/dists sorted ascending. */
  private selectNeighbors(ids: number[], dists: number[], m: number): number[] {
    const selected: number[] = [];
    const selectedVecs: Float32Array[] = [];
    const pruned: number[] = [];
    for (let i = 0; i < ids.length; i++) {
      if (selected.length >= m) break;
      const c = ids[i];
      const cd = dists[i];
      const cv = this.getVector(c);
      let good = true;
      for (let j = 0; j < selectedVecs.length; j++) {
        if (this.dist(cv, selectedVecs[j]) < cd) {
          good = false;
          break;
        }
      }
      if (good) {
        selected.push(c);
        selectedVecs.push(cv);
      } else pruned.push(c);
    }
    for (let i = 0; i < pruned.length && selected.length < m; i++) selected.push(pruned[i]);
    return selected;
  }

  private sortedFromHeap(heap: Heap<number>): { ids: number[]; dists: number[] } {
    const n = heap.size;
    const ids = new Array<number>(n);
    const dists = new Array<number>(n);
    for (let i = n - 1; i >= 0; i--) {
      const [d, id] = heap.pop();
      ids[i] = id;
      dists[i] = d;
    }
    return { ids, dists };
  }

  insert(id: number): void {
    this.ensureCapacity(id);
    if (this.levels[id] >= 0) throw new Error(`hnsw: internal id ${id} already present`);
    const level = this.randomLevel();
    this.levels[id] = level;
    this.deleted[id] = 0;
    const own: number[][] = [];
    for (let l = 0; l <= level; l++) own.push([]);
    this.links[id] = own;
    this.liveCount++;

    if (this.entryPoint < 0) {
      this.entryPoint = id;
      this.maxLevel = level;
      return;
    }
    const q = this.getVector(id);
    let ep = this.entryPoint;
    for (let l = this.maxLevel; l > level; l--) ep = this.greedy(q, ep, l);
    let eps = [ep];
    for (let l = Math.min(level, this.maxLevel); l >= 0; l--) {
      const heap = this.searchLayer(q, eps, this.efConstruction, l);
      const { ids, dists } = this.sortedFromHeap(heap);
      const mMax = l === 0 ? this.mMax0 : this.m;
      const neighbors = this.selectNeighbors(ids, dists, this.m);
      own[l] = neighbors;
      for (let i = 0; i < neighbors.length; i++) {
        const nb = neighbors[i];
        const nl = this.links[nb][l];
        nl.push(id);
        if (nl.length > mMax) {
          const nv = this.getVector(nb);
          const order = nl.map((x) => ({ x, d: this.dist(nv, this.getVector(x)) })).sort((a, b) => a.d - b.d);
          this.links[nb][l] = this.selectNeighbors(order.map((o) => o.x), order.map((o) => o.d), mMax);
        }
      }
      eps = ids.length ? ids : [ep];
    }
    if (level > this.maxLevel) {
      this.maxLevel = level;
      this.entryPoint = id;
    }
  }

  markDeleted(id: number): void {
    if (!this.has(id) || this.deleted[id] === 1) return;
    this.deleted[id] = 1;
    this.liveCount--;
  }

  /**
   * k nearest live nodes satisfying `allow`. `ef` is the beam width (>= k).
   * Returns ascending by distance.
   */
  search(q: Float32Array, k: number, ef: number, allow?: Predicate): Array<{ id: number; distance: number }> {
    if (this.entryPoint < 0 || k <= 0) return [];
    ef = Math.max(ef, k);
    let ep = this.entryPoint;
    for (let l = this.maxLevel; l > 0; l--) ep = this.greedy(q, ep, l);
    const heap = this.searchLayer(q, [ep], ef, 0, allow);
    const { ids, dists } = this.sortedFromHeap(heap);
    const n = Math.min(k, ids.length);
    const out = new Array<{ id: number; distance: number }>(n);
    for (let i = 0; i < n; i++) out[i] = { id: ids[i], distance: dists[i] };
    return out;
  }

  serialize(): HnswSerialized {
    const del: number[] = [];
    for (let i = 0; i < this.levels.length; i++) if (this.deleted[i] === 1) del.push(i);
    return {
      m: this.m,
      efConstruction: this.efConstruction,
      maxLevel: this.maxLevel,
      entryPoint: this.entryPoint,
      count: this.liveCount,
      levels: this.levels,
      links: this.links,
      deleted: del,
      seedState: this.seedState,
    };
  }

  static deserialize(s: HnswSerialized, getVector: (id: number) => Float32Array, dist: DistanceFn): HNSW {
    const h = new HNSW({ m: s.m, efConstruction: s.efConstruction, seed: s.seedState }, getVector, dist);
    h.levels = s.levels;
    h.links = s.links;
    h.maxLevel = s.maxLevel;
    h.entryPoint = s.entryPoint;
    const n = Math.max(64, s.levels.length);
    h.deleted = new Uint8Array(n);
    h.visited = new Uint32Array(n);
    for (const d of s.deleted) h.deleted[d] = 1;
    h.liveCount = s.count;
    return h;
  }
}
