#!/usr/bin/env node
/**
 * mem — command line for the agent memory engine (npm: agent-memory-engine).
 *
 *   mem remember [--kind K] [--tags a,b] [--source S] [--importance 0.8] "text"
 *   mem recall [--top-k 5] [--kinds a,b] [--no-hybrid] [--decay] [--explain] "question"
 *   mem list [--kinds a,b] [--limit 50]
 *   mem get ID | mem forget ID
 *   mem stats | mem info | mem verify
 *   mem snapshot [FILE] | mem restore FILE [--namespace NS] [--overwrite]
 *   mem compact --target N [--policy coherence|lru|lfu]
 *   mem consolidate [--dry-run]
 *   mem reembed --to onnx:bge-small-en-v1.5
 *   mem graph "MATCH (n) RETURN n LIMIT 5"
 *   mem init            (download and cache the embedding model)
 *   mem serve [...]     (start the MCP server on stdio)
 *   mem sse [--port 3939] [--host 127.0.0.1] [--token SECRET]   (MCP over HTTP: /sse and /mcp)
 * Global: --root DIR --namespace NS --embedder SPEC --allow-fallback --json
 */
import { parseArgs } from 'node:util';
import { loadConfig } from './config.js';
import { AgentMemory } from './memory/agent-memory.js';
import { createEmbedder, resolveEmbedder } from './embedding/index.js';
import { MemError } from './core/errors.js';

const { values: v, positionals } = parseArgs({
  args: process.argv.slice(2),
  allowPositionals: true,
  strict: false,
  options: {
    root: { type: 'string' },
    namespace: { type: 'string' },
    embedder: { type: 'string' },
    'allow-fallback': { type: 'boolean' },
    json: { type: 'boolean' },
    kind: { type: 'string' },
    kinds: { type: 'string' },
    tags: { type: 'string' },
    source: { type: 'string' },
    importance: { type: 'string' },
    supersedes: { type: 'string' },
    'top-k': { type: 'string' },
    'no-hybrid': { type: 'boolean' },
    decay: { type: 'boolean' },
    explain: { type: 'boolean' },
    limit: { type: 'string' },
    target: { type: 'string' },
    policy: { type: 'string' },
    'dry-run': { type: 'boolean' },
    to: { type: 'string' },
    overwrite: { type: 'boolean' },
    profile: { type: 'string' },
    sse: { type: 'boolean' },
    port: { type: 'string' },
    host: { type: 'string' },
    token: { type: 'string' },
    cors: { type: 'string' },
    help: { type: 'boolean', short: 'h' },
  },
});

const cmd = positionals[0];
const arg = positionals.slice(1).join(' ');

function out(x: unknown) {
  console.log(JSON.stringify(x, (_k, val) => (val instanceof Float32Array ? undefined : val), 2));
}

async function openMemory() {
  const config = loadConfig({ root: v.root as string | undefined, namespace: v.namespace as string | undefined, embedder: v.embedder as string | undefined, allowFallback: v['allow-fallback'] as boolean | undefined, actor: 'cli' });
  const r = await resolveEmbedder({ spec: config.embedder, allowFallback: config.allowFallback, log: (m) => console.error('[mem] ' + m) });
  const mem = new AgentMemory({ root: config.root, namespace: config.namespace, embedder: r.provider, fallbackActivated: r.fallbackActivated, actor: 'cli', capacity: config.capacity, learning: config.learning || undefined });
  return { mem, config, resolved: r };
}

function list(s: unknown): string[] | undefined {
  return typeof s === 'string' && s ? s.split(',').map((x) => x.trim()).filter(Boolean) : undefined;
}

async function run() {
  if (!cmd || v.help) {
    console.error(`usage: mem <remember|recall|list|get|forget|stats|info|verify|snapshot|restore|compact|consolidate|reembed|graph|init|serve|sse> [options] [text]
  --root DIR  --namespace NS  --embedder onnx:MODEL|ngram  --allow-fallback  --json
  remember: --kind K --tags a,b --source S --importance 0..1 --supersedes ID
  recall:   --top-k N --kinds a,b --no-hybrid --decay --explain
  compact:  --target N --policy coherence|lru|lfu      consolidate: --dry-run
  restore:  FILE --namespace NS --overwrite           reembed: --to SPEC
  serve:    MCP over stdio [--profile P]              sse: MCP over HTTP [--port 3939] [--host 127.0.0.1] [--token S] [--cors o1,o2]`);
    process.exit(cmd ? 0 : 1);
  }
  if (cmd === 'serve' || cmd === 'sse') {
    const { main } = await import('./mcp/server.js');
    const passthrough: string[] = [];
    if (v.root) passthrough.push('--root', String(v.root));
    if (v.namespace) passthrough.push('--namespace', String(v.namespace));
    if (v.embedder) passthrough.push('--embedder', String(v.embedder));
    if (v.profile) passthrough.push('--profile', String(v.profile));
    if (v['allow-fallback']) passthrough.push('--allow-fallback');
    if (cmd === 'sse' || v.sse) passthrough.push('--transport', 'sse');
    if (v.port) passthrough.push('--port', String(v.port));
    if (v.host) passthrough.push('--host', String(v.host));
    if (v.token) passthrough.push('--token', String(v.token));
    if (v.cors) passthrough.push('--cors', String(v.cors));
    await main(passthrough);
    return;
  }
  if (cmd === 'init') {
    const config = loadConfig({ embedder: v.embedder as string | undefined });
    const e = createEmbedder(config.embedder);
    const t0 = Date.now();
    await e.init();
    out({ embedder: e.name, dimensions: e.dimensions, semantic: e.semantic, identity: e.identity(), ms: Date.now() - t0 });
    return;
  }
  if (cmd === 'restore') {
    if (!arg) throw new MemError('INVALID_ARGUMENT', 'restore needs a snapshot file');
    const config = loadConfig({ root: v.root as string | undefined });
    out(AgentMemory.restore(arg, config.root, { namespace: v.namespace as string | undefined, overwrite: !!v.overwrite }));
    return;
  }
  const { mem, config, resolved } = await openMemory();
  try {
    switch (cmd) {
      case 'remember': {
        if (!arg) throw new MemError('INVALID_ARGUMENT', 'remember needs text');
        out(await mem.remember({ text: arg, kind: v.kind as string | undefined, tags: list(v.tags), source: v.source as string | undefined, importance: v.importance ? Number(v.importance) : undefined, supersedes: v.supersedes as string | undefined }));
        break;
      }
      case 'recall': {
        if (!arg) throw new MemError('INVALID_ARGUMENT', 'recall needs a query');
        const r = await mem.recall({ text: arg, k: v['top-k'] ? Number(v['top-k']) : 5, kinds: list(v.kinds), tags: list(v.tags), hybrid: !v['no-hybrid'], decay: !!v.decay, explain: !!v.explain });
        if (v.json) out(r);
        else {
          if (r.warning) console.error('[mem] ' + r.warning);
          for (const h of r.results) console.log(`${h.finalScore.toFixed(3)} sim=${h.similarity.toFixed(2)}  [${h.kind}] ${h.text.replace(/\s+/g, ' ').slice(0, 160)}  (${h.id.slice(0, 8)}${h.tags.length ? ' ' + h.tags.join(',') : ''})`);
          console.error(`${r.results.length} result(s), strategy=${r.strategy}, ${r.tookMs} ms, embedder=${r.embedder}`);
        }
        break;
      }
      case 'list':
        out(mem.list({ kinds: list(v.kinds), tags: list(v.tags), limit: v.limit ? Number(v.limit) : 50 }));
        break;
      case 'get':
        out(mem.get(arg));
        break;
      case 'forget':
        out({ id: arg, deleted: mem.forget(arg) });
        break;
      case 'stats':
        out(mem.stats());
        break;
      case 'info':
        out({ root: config.root, namespace: config.namespace, embedder: { requested: resolved.requested, active: resolved.provider.name, semantic: resolved.provider.semantic, fallbackActivated: resolved.fallbackActivated, identity: resolved.provider.identity() }, features: mem.store.manifest.features });
        break;
      case 'verify':
        out(await mem.verify());
        break;
      case 'snapshot':
        out(mem.snapshot(arg || undefined));
        break;
      case 'compact':
        out(mem.compact({ target: v.target ? Number(v.target) : undefined, policy: v.policy as string | undefined }));
        break;
      case 'consolidate':
        out(await mem.consolidate({ dryRun: !!v['dry-run'] }));
        break;
      case 'reembed': {
        if (!v.to) throw new MemError('INVALID_ARGUMENT', 'reembed needs --to SPEC');
        const to = createEmbedder(String(v.to));
        await to.init();
        out(await mem.reembed(to));
        break;
      }
      case 'graph': {
        const r = mem.graph.query(arg);
        out({ columns: r.columns, rows: r.rows, stats: r.stats });
        break;
      }
      default:
        throw new MemError('INVALID_ARGUMENT', `unknown command ${cmd}`);
    }
  } finally {
    mem.close();
  }
}

run().catch((e) => {
  if (e instanceof MemError) console.error(`error ${e.code}: ${e.message}`);
  else console.error('error:', e?.message ?? e);
  process.exit(1);
});
