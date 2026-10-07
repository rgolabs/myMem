/**
 * Deterministic lexical fallback: hashed character trigrams + word unigrams, signed feature hashing,
 * L2-normalised. Not semantic. Used for tests, offline builds, and only as an opt-in fallback.
 */
import { sha256 } from '../core/witness.js';
import type { EmbeddingSpaceIdentity } from '../core/types.js';
import type { EmbeddingProvider, EmbedRole } from './provider.js';

function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

export class NgramEmbedder implements EmbeddingProvider {
  readonly name: string;
  readonly semantic = false;
  constructor(readonly dimensions = 256) {
    this.name = `ngram-${dimensions}`;
  }
  async init() {}
  isReady() {
    return true;
  }
  private embed(text: string): Float32Array {
    const v = new Float32Array(this.dimensions);
    const t = text.toLowerCase().replace(/\s+/g, ' ').trim();
    const feats: string[] = [];
    for (const w of t.split(/[^\p{L}\p{N}]+/u)) if (w) feats.push('w:' + w);
    const padded = ` ${t} `;
    for (let i = 0; i + 3 <= padded.length; i++) feats.push('g:' + padded.slice(i, i + 3));
    for (const f of feats) {
      const h = fnv1a(f);
      const idx = h % this.dimensions;
      const sign = (h >>> 31) & 1 ? -1 : 1;
      v[idx] += sign;
    }
    let n = 0;
    for (let i = 0; i < v.length; i++) n += v[i] * v[i];
    n = Math.sqrt(n) || 1;
    for (let i = 0; i < v.length; i++) v[i] /= n;
    return v;
  }
  async embedQuery(text: string) {
    return this.embed(text);
  }
  async embedPassage(text: string) {
    return this.embed(text);
  }
  async embedBatch(texts: string[], _role: EmbedRole) {
    return texts.map((t) => this.embed(t));
  }
  identity(): EmbeddingSpaceIdentity {
    return {
      embedderKind: 'ngram',
      modelId: `ngram-hash-v1-${this.dimensions}`,
      dimension: this.dimensions,
      normalize: true,
      prefixPolicy: 'none',
      promptTemplateHash: sha256('').slice(0, 16),
    };
  }
}
