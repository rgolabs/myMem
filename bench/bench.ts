/**
 * Benchmark with the reporting the spec demands: dataset, dimension, index parameters, recall target,
 * hardware, latency percentiles, throughput and recall together.
 *
 *   npx tsx bench/bench.ts [--n 100000] [--dim 384] [--k 10] [--queries 200] [--ef 100] [--real]
 * --real uses the local MiniLM model on generated sentences (clustered, realistic) instead of uniform random vectors.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { Collection } from '../src/core/vector-store.js';
import { normalize, normalizedCosineDistance } from '../src/core/distance.js';
import { rng, randomVector } from '../test/helpers.js';

const { values } = parseArgs({ options: { n: { type: 'string' }, dim: { type: 'string' }, k: { type: 'string' }, queries: { type: 'string' }, ef: { type: 'string' }, real: { type: 'boolean' }, m: { type: 'string' } } });
const N = parseInt(values.n ?? '100000', 10);
const DIM = parseInt(values.dim ?? '384', 10);
const K = parseInt(values.k ?? '10', 10);
const Q = parseInt(values.queries ?? '200', 10);
const EF = parseInt(values.ef ?? '100', 10);
const M = parseInt(values.m ?? '16', 10);

const pct = (xs: number[], p: number) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
};

async function vectors(): Promise<{ data: Float32Array[]; queries: Float32Array[]; dataset: string }> {
  const r = rng(42);
  if (values.real) {
    const { resolveEmbedder } = await import('../src/embedding/index.js');
    const e = (await resolveEmbedder({ spec: 'onnx:all-MiniLM-L6-v2' })).provider;
    const topics = ['deployment pipeline', 'customer data residency', 'flaky tests', 'api rate limits', 'database migration', 'user preferences', 'incident postmortem', 'release notes', 'security review', 'performance regression'];
    const verbs = ['decided', 'observed', 'learned', 'fixed', 'documented', 'questioned', 'measured', 'rejected'];
    const sentences = Array.from({ length: N }, (_, i) => `We ${verbs[i % verbs.length]} that the ${topics[Math.floor(r() * topics.length)]} for module ${i % 97} ${['needs review', 'is stable', 'blocks the release', 'was changed on Tuesday', 'should be cached'][i % 5]} (${i}).`);
    const t0 = performance.now();
    const data: Float32Array[] = [];
    for (let i = 0; i < sentences.length; i += 256) data.push(...(await e.embedBatch(sentences.slice(i, i + 256), 'passage')));
    console.log(`embedded ${N} sentences in ${Math.round(performance.now() - t0)} ms (${((performance.now() - t0) / N).toFixed(3)} ms/sentence, batch 256)`);
    const queries = await e.embedBatch(Array.from({ length: Q }, (_, i) => `what did we decide about ${topics[i % topics.length]}?`), 'query');
    return { data, queries, dataset: `minilm-sentences(${N})` };
  }
  return { data: Array.from({ length: N }, () => normalize(randomVector(DIM, r))), queries: Array.from({ length: Q }, () => normalize(randomVector(DIM, r))), dataset: `uniform-random(${N})` };
}

async function main() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mem-bench-'));
  const { data, queries, dataset } = await vectors();
  const dim = data[0].length;
  const c = Collection.create(path.join(dir, 'c'), { dimensions: dim, index: { m: M, efConstruction: 2 * EF, efSearch: EF } });
  const t0 = performance.now();
  for (let i = 0; i < N; i += 1000) c.insertBatch(data.slice(i, i + 1000).map((v, j) => ({ id: `r${i + j}`, vector: v, metadata: { kind: (i + j) % 3 === 0 ? 'fact' : 'note', text: `record ${i + j}`, group: (i + j) % 100 } })));
  const insertMs = performance.now() - t0;
  const exact = (q: Float32Array) => data.map((v, i) => [normalizedCosineDistance(q, v), i] as const).sort((a, b) => a[0] - b[0]).slice(0, K).map((x) => `r${x[1]}`);
  const lat: number[] = [];
  let hits = 0;
  for (const q of queries) {
    const t = performance.now();
    const res = c.search({ vector: q, k: K });
    lat.push(performance.now() - t);
    const truth = new Set(exact(q));
    hits += res.results.filter((x) => truth.has(x.id)).length;
  }
  const flat: number[] = [];
  for (const q of queries.slice(0, 20)) {
    const t = performance.now();
    exact(q);
    flat.push(performance.now() - t);
  }
  const filt: number[] = [];
  let fm = 0;
  for (const q of queries) {
    const t = performance.now();
    const res = c.search({ vector: q, k: K, filter: { kind: 'note' } });
    filt.push(performance.now() - t);
    fm += res.results.length;
  }
  const sel: number[] = [];
  for (const q of queries) {
    const t = performance.now();
    c.search({ vector: q, k: K, filter: { group: 7 } });
    sel.push(performance.now() - t);
  }
  const tf = performance.now();
  c.flush();
  const flushMs = performance.now() - tf;
  const t1 = performance.now();
  const c2 = Collection.open(path.join(dir, 'c'));
  c2.search({ vector: queries[0], k: K });
  const openMs = performance.now() - t1;
  const tc = performance.now();
  c2.compact({ target: Math.floor(N / 2) });
  const compactMs = performance.now() - tc;
  const rss = process.memoryUsage().rss / N;
  const report = {
    dataset,
    records: N,
    dimension: dim,
    metric: 'cosine',
    index: { kind: 'hnsw', m: M, efConstruction: 2 * EF, efSearch: EF },
    k: K,
    queries: Q,
    hardware: { cpu: os.cpus()[0]?.model, cores: os.cpus().length, arch: os.arch(), node: process.version, platform: `${os.platform()} ${os.release()}` },
    insert: { totalMs: Math.round(insertMs), recordsPerSecond: Math.round(N / (insertMs / 1000)), note: 'includes HNSW build and write-ahead log fsync per 1000-record batch' },
    search: { recallAt10: +(hits / (Q * K)).toFixed(4), p50Ms: +pct(lat, 50).toFixed(3), p95Ms: +pct(lat, 95).toFixed(3), p99Ms: +pct(lat, 99).toFixed(3) },
    exactScan: { p50Ms: +pct(flat, 50).toFixed(2) },
    filteredBroad: { filter: 'kind=note (~67%)', p50Ms: +pct(filt, 50).toFixed(3), p99Ms: +pct(filt, 99).toFixed(3), avgMatched: +(fm / Q).toFixed(2) },
    filteredSelective: { filter: 'group=7 (1%)', p50Ms: +pct(sel, 50).toFixed(3), p99Ms: +pct(sel, 99).toFixed(3) },
    checkpoint: { flushMs: Math.round(flushMs), coldOpenWithPersistedIndexMs: Math.round(openMs) },
    compaction: { to50PercentMs: Math.round(compactMs) },
    memory: { rssBytesPerRecord: Math.round(rss) },
  };
  console.log(JSON.stringify(report, null, 2));
  fs.rmSync(dir, { recursive: true, force: true });
}
main();
