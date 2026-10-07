import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { Collection } from '../src/core/vector-store.js';
import { HNSW } from '../src/core/hnsw.js';
import { WitnessLog } from '../src/core/witness.js';
import { Heap } from '../src/core/heap.js';
import { cosineDistance, normalize, normalizedCosineDistance, euclideanDistance } from '../src/core/distance.js';
import { matchesFilter } from '../src/core/filter.js';
import { Bm25Index, rrf, rsf } from '../src/core/lexical.js';
import { MemError } from '../src/core/errors.js';
import { tmpDir, rng, randomVector } from './helpers.js';

test('heap orders min and max', () => {
  const h = new Heap<string>(false);
  h.push(3, 'c'); h.push(1, 'a'); h.push(2, 'b');
  assert.deepEqual(h.drain().map((x) => x[1]), ['a', 'b', 'c']);
  const m = new Heap<string>(true);
  m.push(3, 'c'); m.push(1, 'a'); m.push(2, 'b');
  assert.equal(m.pop()[1], 'c');
});

test('distance functions', () => {
  const a = Float32Array.from([1, 0]), b = Float32Array.from([0, 1]);
  assert.ok(Math.abs(cosineDistance(a, b) - 1) < 1e-6);
  assert.ok(Math.abs(cosineDistance(a, a)) < 1e-6);
  assert.ok(Math.abs(euclideanDistance(a, b) - Math.SQRT2) < 1e-6);
  assert.ok(Math.abs(normalizedCosineDistance(normalize(Float32Array.from([2, 0])), a)) < 1e-6);
});

test('filter operators', () => {
  const m = { kind: 'fact', n: 5, tags: ['a', 'b'], s: 'hello' };
  assert.ok(matchesFilter(m, { kind: 'fact' }));
  assert.ok(matchesFilter(m, { tags: 'a' }));
  assert.ok(matchesFilter(m, { n: { $gt: 4, $lte: 5 } }));
  assert.ok(matchesFilter(m, { kind: { $in: ['x', 'fact'] }, s: { $prefix: 'he' } }));
  assert.ok(matchesFilter(m, { tags: { $all: ['a', 'b'] }, missing: { $exists: false } }));
  assert.ok(!matchesFilter(m, { tags: { $contains: 'z' } }));
  assert.ok(!matchesFilter(m, { n: { $ne: 5 } }));
});

test('bm25 and fusion', () => {
  const idx = new Bm25Index();
  idx.add(1, 'deploy pipeline uses github actions');
  idx.add(2, 'the customer requires data residency in canada');
  idx.add(3, 'api/v2 user_id endpoint');
  assert.equal(idx.search('github actions', 5)[0].doc, 1);
  assert.equal(idx.search('user_id', 5)[0].doc, 3);
  assert.equal(idx.search('v2', 5)[0].doc, 3);
  const fused = rrf([{ items: [{ key: 'a', score: 1 }, { key: 'b', score: 0.5 }], weight: 0.5 }, { items: [{ key: 'b', score: 3 }, { key: 'a', score: 1 }], weight: 0.5 }]);
  assert.equal(fused.length, 2);
  assert.ok(Math.abs(fused[0].score - fused[1].score) < 1e-9);
  const top = rrf([{ items: [{ key: 'a', score: 1 }], weight: 0.5 }, { items: [{ key: 'a', score: 1 }], weight: 0.5 }]);
  assert.ok(Math.abs(top[0].score - 1) < 1e-9, 'first in every list scores 1');
  const r2 = rsf([{ items: [{ key: 'a', score: 2 }, { key: 'b', score: 1 }], weight: 1 }]);
  assert.equal(r2[0].key, 'a');
});

test('hnsw recall against exact search and predicate traversal', () => {
  const r = rng(7);
  const dim = 32, N = 3000;
  const vecs = Array.from({ length: N }, () => normalize(randomVector(dim, r)));
  const h = new HNSW({ m: 16, efConstruction: 100 }, (i) => vecs[i], normalizedCosineDistance);
  for (let i = 0; i < N; i++) h.insert(i);
  let hit = 0;
  for (let q = 0; q < 30; q++) {
    const qv = normalize(randomVector(dim, r));
    const exact = vecs.map((v, i) => [normalizedCosineDistance(qv, v), i] as const).sort((a, b) => a[0] - b[0]).slice(0, 10).map((x) => x[1]);
    const approx = h.search(qv, 10, 64).map((x) => x.id);
    hit += approx.filter((i) => exact.includes(i)).length;
  }
  assert.ok(hit / 300 >= 0.9, `recall@10 ${hit / 300}`);
  const even = h.search(normalize(randomVector(dim, r)), 10, 64, (i) => i % 2 === 0);
  assert.equal(even.length, 10);
  assert.ok(even.every((x) => x.id % 2 === 0));
  h.markDeleted(even[0].id);
  assert.ok(!h.search(normalize(randomVector(dim, r)), 3000, 3000).some((x) => x.id === even[0].id));
  const s = h.serialize();
  const h2 = HNSW.deserialize(JSON.parse(JSON.stringify(s)), (i) => vecs[i], normalizedCosineDistance);
  assert.equal(h2.live, N - 1);
  assert.deepEqual(h2.search(vecs[5], 1, 16)[0].id, 5);
});

test('collection: insert, search, filter k semantics, persistence, torn log line', () => {
  const dir = tmpDir('col');
  const r = rng(3);
  const dim = 16, N = 2500;
  const w = new WitnessLog(path.join(dir, 'ns'));
  let c = Collection.create(path.join(dir, 'memory'), { dimensions: dim }, { witness: w });
  const recs = Array.from({ length: N }, (_, i) => ({ id: `r${i}`, vector: randomVector(dim, r), metadata: { kind: i % 4 === 0 ? 'fact' : 'note', text: `document ${i} ${i % 9 === 0 ? 'zebra' : 'apple'}`, createdAtMs: Date.now() - i * 60_000, tags: [`t${i % 5}`], group: i % 50 } }));
  c.insertBatch(recs);
  assert.equal(c.count(), N);
  assert.throws(() => c.insert(recs[0]), (e: any) => e.code === 'ALREADY_EXISTS');
  assert.throws(() => c.insert({ id: 'bad', vector: new Float32Array(dim + 1) }), (e: any) => e.code === 'DIMENSION_MISMATCH');
  assert.throws(() => Collection.create(path.join(dir, 'q'), { dimensions: 4, quantization: 'scalar' as any }), (e: any) => e.code === 'UNSUPPORTED');

  const q = randomVector(dim, r);
  const plain = c.search({ vector: q, k: 10 });
  assert.equal(plain.results.length, 10);
  assert.equal(plain.strategy, 'hnsw');
  for (let i = 1; i < plain.results.length; i++) assert.ok(plain.results[i].score >= plain.results[i - 1].score);

  const selective = c.search({ vector: q, k: 10, filter: { group: 7 } });
  assert.equal(selective.strategy, 'exact-filtered');
  assert.equal(selective.results.length, 10, 'k means matching rows');
  assert.ok(selective.results.every((x) => x.metadata!.group === 7));
  assert.ok(selective.complete);

  const broad = c.search({ vector: q, k: 10, filter: { kind: 'note' } });
  assert.equal(broad.results.length, 10);
  assert.ok(['hnsw-predicate', 'exact-filtered'].includes(broad.strategy));
  assert.ok(broad.results.every((x) => x.metadata!.kind === 'note'));

  const tiny = c.search({ vector: q, k: 10, filter: { group: 7, kind: 'fact' } });
  assert.ok(tiny.results.length <= 13 && tiny.results.every((x) => x.metadata!.kind === 'fact' && x.metadata!.group === 7));

  // hybrid finds the lexical outlier
  const hy = c.search({ vector: q, k: 5, fusion: { lexical: 'document 2403', alpha: 0.3 }, explain: true });
  assert.ok(hy.results.some((x) => x.id === 'r2403'));
  assert.ok(hy.results[0].explain?.fusion);
  // decay strongly favours newest
  const dec = c.search({ vector: q, k: 5, decay: { halfLifeDays: 0.001 }, explain: true });
  assert.ok(dec.results.every((x) => x.explain!.decay! <= 1));
  // mmr runs
  const mmr = c.search({ vector: q, k: 5, diversity: 0.5, explain: true });
  assert.equal(mmr.results.length, 5);
  assert.ok(mmr.results.every((x) => x.explain!.mmr !== undefined));
  // coherence
  const coh = c.search({ vector: q, k: 5, coherence: { weight: 1, threshold: 0.1 }, explain: true });
  assert.equal(coh.results.length, 5);

  // update / delete / list
  c.updateMetadata('r1', { extra: 'x' });
  assert.equal(c.get('r1')!.metadata.extra, 'x');
  assert.ok(c.delete('r2'));
  assert.equal(c.get('r2'), null);
  assert.equal(c.list({ filter: { group: 7 }, limit: 5 }).total, 50);

  // reopen with persisted index (no rebuild) and same answers
  c.flush();
  const cpBefore = fs.statSync(path.join(dir, 'memory', 'checkpoint.json')).mtimeMs;
  c = Collection.open(path.join(dir, 'memory'), { witness: w });
  assert.equal(c.count(), N - 1);
  assert.equal(fs.statSync(path.join(dir, 'memory', 'checkpoint.json')).mtimeMs, cpBefore, 'open did not rewrite the checkpoint');
  assert.deepEqual(c.search({ vector: q, k: 5 }).results.map((x) => x.id), plain.results.slice(0, 5).map((x) => x.id).filter((id) => id !== 'r2').slice(0, 5));

  // crash safety: a torn trailing line is ignored, and a writer cleans it up
  c.insert({ id: 'after', vector: randomVector(dim, r), metadata: { kind: 'fact', text: 'after' } });
  fs.appendFileSync(path.join(dir, 'memory', 'log.jsonl'), '{"g":99999,"op":"insert","recs":[{"id":"torn"');
  const c2 = Collection.open(path.join(dir, 'memory'));
  assert.equal(c2.count(), N);
  assert.ok(!c2.has('torn'));
  c2.insert({ id: 'after2', vector: randomVector(dim, r), metadata: { kind: 'fact', text: 'after2' } });
  const c3 = Collection.open(path.join(dir, 'memory'));
  assert.ok(c3.has('after2') && c3.has('after') && !c3.has('torn'));

  // capability gating inside traversal
  c3.insert({ id: 'secret', vector: q, metadata: { kind: 'fact', text: 'secret', _caps: '4' } });
  const noCaps = c3.search({ vector: q, k: 1 });
  assert.notEqual(noCaps.results[0].id, 'secret');
  const withCaps = c3.search({ vector: q, k: 1, capabilities: 6n });
  assert.equal(withCaps.results[0].id, 'secret');
  const wrongCaps = c3.search({ vector: q, k: 1, capabilities: 2n });
  assert.notEqual(wrongCaps.results[0].id, 'secret');
});

test('collection: embedding space provenance is enforced', () => {
  const dir = tmpDir('space');
  const ident = { embedderKind: 'ngram', modelId: 'a', dimension: 4, normalize: true, prefixPolicy: 'none', promptTemplateHash: 'x' };
  const c = Collection.create(path.join(dir, 'm'), { dimensions: 4, embedding: ident });
  c.insert({ id: '1', vector: [1, 0, 0, 0] }, ident);
  assert.throws(() => c.insert({ id: '2', vector: [1, 0, 0, 0] }, { ...ident, modelId: 'b' }), (e: any) => e.code === 'EMBEDDING_SPACE_MISMATCH' && /modelId|differs/.test(e.message));
  assert.throws(() => c.search({ vector: [1, 0, 0, 0], k: 1, identity: { ...ident, modelId: 'b' } }), (e: any) => e.code === 'EMBEDDING_SPACE_MISMATCH');
  assert.throws(() => Collection.create(path.join(dir, 'bad'), { dimensions: 5, embedding: ident }), (e: any) => e.code === 'DIMENSION_MISMATCH');
});

test('collection: compaction respects pins and cluster diversity, logs evictions', async () => {
  const dir = tmpDir('compact');
  const r = rng(11);
  const w = new WitnessLog(dir);
  const c = Collection.create(path.join(dir, 'm'), { dimensions: 8, capacity: 0 }, { witness: w });
  for (let i = 0; i < 100; i++) c.insert({ id: `r${i}`, vector: randomVector(8, r), metadata: { kind: i < 90 ? 'a' : 'b', tags: [i < 90 ? 'x' : 'y'], text: 't', pinned: i === 0 } });
  const res = c.compact({ target: 10, policy: 'coherence', minPerCluster: 3 });
  assert.equal(res.after, 10);
  assert.ok(c.has('r0'), 'pinned survives');
  assert.ok(c.list({ filter: { kind: 'b' } }).total >= 3, 'minority cluster keeps survivors');
  const v = await w.verify();
  assert.ok(v.ok);
  let compactEntries = 0;
  for await (const e of w.entries()) if (e.operation === 'compact') compactEntries++;
  assert.equal(compactEntries, 90);
  // auto compaction when capacity exceeded
  const c2 = Collection.create(path.join(dir, 'auto'), { dimensions: 8, capacity: 20 });
  for (let i = 0; i < 30; i++) c2.insert({ id: `a${i}`, vector: randomVector(8, r), metadata: { text: 't' } });
  assert.ok(c2.count() <= 20);
});

test('witness log detects tampering', async () => {
  const dir = tmpDir('witness');
  const w = new WitnessLog(dir);
  w.append('insert', 'a', { x: 1 });
  w.append('insert', 'b', { x: 2 });
  w.appendMany([{ operation: 'delete', recordId: 'a' }, { operation: 'delete', recordId: 'b' }]);
  assert.ok((await w.verify()).ok);
  const file = path.join(dir, 'witness.jsonl');
  const lines = fs.readFileSync(file, 'utf8').trim().split('\n');
  const e = JSON.parse(lines[1]);
  e.recordId = 'tampered';
  lines[1] = JSON.stringify(e);
  fs.writeFileSync(file, lines.join('\n') + '\n');
  const v = await w.verify();
  assert.equal(v.ok, false);
  assert.equal(v.firstBreak!.seq, 2);
});

test('limits fail with typed errors', () => {
  const dir = tmpDir('limits');
  const c = Collection.create(path.join(dir, 'm'), { dimensions: 2 });
  assert.throws(() => c.insert({ id: 'x'.repeat(600), vector: [1, 0] }), (e: any) => e instanceof MemError && e.code === 'LIMIT_EXCEEDED');
  assert.throws(() => c.insert({ id: 'big', vector: [1, 0], metadata: { blob: 'y'.repeat(70_000) } }), (e: any) => e.code === 'LIMIT_EXCEEDED');
  assert.throws(() => c.search({ vector: [1, 0], k: 0 }), (e: any) => e.code === 'INVALID_ARGUMENT');
  assert.throws(() => c.search({ vector: [1, 0], k: 1, filter: { a: { $regex: 'x' } } as any }), (e: any) => e.code === 'UNSUPPORTED');
});
