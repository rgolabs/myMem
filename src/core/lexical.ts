/**
 * In-memory BM25 index over one text field, with precomputed document frequencies and lengths,
 * plus reciprocal-rank and relative-score fusion for hybrid search (spec §6, §16 enhancement 6).
 */
const STOP = new Set(
  'a an and are as at be by for from has he in is it its of on or that the to was were will with this these those i you we they not no but if then than so do does did'.split(' '),
);

export function tokenize(text: string): string[] {
  const out: string[] = [];
  for (const raw of text.toLowerCase().split(/[^\p{L}\p{N}_.\-/:@]+/u)) {
    if (!raw) continue;
    // keep identifiers like "api/v2", "user_id", "3.14" whole, but also index their parts
    const t = raw.replace(/^[._\-/:@]+|[._\-/:@]+$/g, '');
    if (!t || STOP.has(t)) continue;
    out.push(t);
    if (/[._\-/:@]/.test(t)) {
      for (const part of t.split(/[._\-/:@]+/)) if (part.length > 1 && !STOP.has(part)) out.push(part);
    }
  }
  return out;
}

export class Bm25Index {
  private postings = new Map<string, Map<number, number>>(); // term -> (doc -> tf)
  private docLen = new Map<number, number>();
  private docTerms = new Map<number, string[]>(); // doc -> distinct terms, so removal is O(|doc|)
  private totalLen = 0;
  constructor(private readonly k1 = 1.2, private readonly b = 0.75) {}

  get size() {
    return this.docLen.size;
  }

  add(doc: number, text: string) {
    if (this.docLen.has(doc)) this.remove(doc);
    const toks = tokenize(text);
    this.docLen.set(doc, toks.length);
    this.totalLen += toks.length;
    for (const t of toks) {
      let p = this.postings.get(t);
      if (!p) this.postings.set(t, (p = new Map()));
      p.set(doc, (p.get(doc) ?? 0) + 1);
    }
    this.docTerms.set(doc, [...new Set(toks)]);
  }

  remove(doc: number) {
    const len = this.docLen.get(doc);
    if (len === undefined) return;
    this.docLen.delete(doc);
    this.totalLen -= len;
    for (const t of this.docTerms.get(doc) ?? []) {
      const p = this.postings.get(t);
      if (p && p.delete(doc) && p.size === 0) this.postings.delete(t);
    }
    this.docTerms.delete(doc);
  }

  search(query: string, k: number, allow?: (doc: number) => boolean): Array<{ doc: number; score: number }> {
    const toks = tokenize(query);
    if (!toks.length || !this.docLen.size) return [];
    const N = this.docLen.size;
    const avgdl = this.totalLen / N;
    const scores = new Map<number, number>();
    const seen = new Set<string>();
    for (const t of toks) {
      if (seen.has(t)) continue;
      seen.add(t);
      const p = this.postings.get(t);
      if (!p) continue;
      const idf = Math.log(1 + (N - p.size + 0.5) / (p.size + 0.5));
      for (const [doc, tf] of p) {
        if (allow && !allow(doc)) continue;
        const dl = this.docLen.get(doc) ?? 0;
        const s = idf * ((tf * (this.k1 + 1)) / (tf + this.k1 * (1 - this.b + (this.b * dl) / avgdl)));
        scores.set(doc, (scores.get(doc) ?? 0) + s);
      }
    }
    return [...scores.entries()]
      .map(([doc, score]) => ({ doc, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }

  clear() {
    this.postings.clear();
    this.docLen.clear();
    this.docTerms.clear();
    this.totalLen = 0;
  }
}

export interface RankedList<T> {
  items: Array<{ key: T; score: number }>; // sorted best-first, higher score better
  weight: number;
}

/** Reciprocal rank fusion, normalised so a result ranked first in every list scores 1. */
export function rrf<T>(lists: RankedList<T>[], kConst = 60): Array<{ key: T; score: number; parts: number[]; ranks: number[] }> {
  const acc = new Map<T, { score: number; parts: number[]; ranks: number[] }>();
  const totalWeight = lists.reduce((a, l) => a + l.weight, 0) || 1;
  const scale = (kConst + 1) / totalWeight;
  lists.forEach((list, li) => {
    list.items.forEach((it, rank) => {
      let a = acc.get(it.key);
      if (!a) acc.set(it.key, (a = { score: 0, parts: new Array(lists.length).fill(0), ranks: new Array(lists.length).fill(-1) }));
      const contrib = (scale * list.weight) / (kConst + rank + 1);
      a.score += contrib;
      a.parts[li] = it.score;
      a.ranks[li] = rank;
    });
  });
  return [...acc.entries()].map(([key, v]) => ({ key, ...v })).sort((a, b) => b.score - a.score);
}

/** Relative score fusion: min-max normalise each list, then weighted sum. */
export function rsf<T>(lists: RankedList<T>[]): Array<{ key: T; score: number; parts: number[]; ranks: number[] }> {
  const acc = new Map<T, { score: number; parts: number[]; ranks: number[] }>();
  lists.forEach((list, li) => {
    if (!list.items.length) return;
    const max = list.items[0].score;
    const min = list.items[list.items.length - 1].score;
    const range = max - min || 1;
    list.items.forEach((it, rank) => {
      let a = acc.get(it.key);
      if (!a) acc.set(it.key, (a = { score: 0, parts: new Array(lists.length).fill(0), ranks: new Array(lists.length).fill(-1) }));
      const n = range === 0 ? 1 : (it.score - min) / range;
      a.score += list.weight * n;
      a.parts[li] = it.score;
      a.ranks[li] = rank;
    });
  });
  return [...acc.entries()].map(([key, v]) => ({ key, ...v })).sort((a, b) => b.score - a.score);
}
