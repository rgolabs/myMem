import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { tmpDir } from './helpers.js';

const SERVER = path.resolve('src/mcp/server.ts');
const TSX = path.resolve('node_modules/.bin/tsx');

async function withServer(root: string, profile: string, fn: (c: Client) => Promise<void>, extraEnv: Record<string, string> = {}) {
  const t = new StdioClientTransport({ command: TSX, args: [SERVER, '--root', root, '--profile', profile], env: { ...process.env, MEM_EMBEDDER: 'ngram', ...extraEnv }, stderr: 'pipe' });
  const c = new Client({ name: 'test', version: '0' });
  await c.connect(t);
  try {
    await fn(c);
  } finally {
    await c.close();
  }
}
const body = (r: any) => JSON.parse(r.content[0].text);

test('mcp server: profiles, tools, resources, persistence', async () => {
  const root = tmpDir('mcp');
  await withServer(root, 'standard', async (c) => {
    const tools = (await c.listTools()).tools.map((t) => t.name);
    assert.ok(tools.includes('memory_remember') && tools.includes('memory_recall') && !tools.includes('memory_purge'));
    assert.match(c.getInstructions() ?? '', /RECALL FIRST/);
    const a = body(await c.callTool({ name: 'memory_remember', arguments: { text: 'The API rate limit is 100 requests per minute.', kind: 'fact', tags: ['api'] } }));
    assert.ok(a.id);
    const r = body(await c.callTool({ name: 'memory_recall', arguments: { text: 'what is the rate limit', k: 2 } }));
    assert.equal(r.results[0].id, a.id);
    assert.ok(r.queryId);
    const o = body(await c.callTool({ name: 'memory_record_outcome', arguments: { queryId: r.queryId, chosenIds: [a.id], reward: 1 } }));
    assert.equal(o.applied.length, 1);
    const missing = (await c.callTool({ name: 'memory_compact', arguments: { target: 0 } })) as any;
    assert.ok(missing.isError);
    const notFound = body(await c.callTool({ name: 'memory_get', arguments: { id: 'nope' } }));
    assert.equal(notFound.error, 'NOT_FOUND');
    const bad = (await c.callTool({ name: 'graph_query', arguments: { cypher: 'MATCH (a)-[*]->(b) RETURN b' } })) as any;
    assert.ok(bad.isError);
    assert.equal(body(bad).error, 'UNSUPPORTED');
    const info = body(await c.callTool({ name: 'memory_info', arguments: {} }));
    assert.equal(info.embedder.active, 'ngram-256');
    assert.equal(info.tools.length, tools.length);
    const res = await c.readResource({ uri: 'memory://stats' });
    assert.equal(JSON.parse(res.contents[0].text as string).records, 1);
    const other = body(await c.callTool({ name: 'memory_stats', arguments: { namespace: 'other' } }));
    assert.equal(other.records, 0);
  });
  await withServer(root, 'read-only', async (c) => {
    const tools = (await c.listTools()).tools.map((t) => t.name);
    assert.ok(!tools.includes('memory_remember'));
    const r = body(await c.callTool({ name: 'memory_recall', arguments: { text: 'rate limit', k: 1 } }));
    assert.equal(r.results.length, 1, 'persisted across restarts');
  });
  await withServer(
    root,
    'administrative',
    async (c) => {
      const tools = (await c.listTools()).tools.map((t) => t.name);
      assert.ok(tools.includes('memory_purge') && !tools.includes('memory_compact'), 'deny list applied');
      const s = body(await c.callTool({ name: 'memory_snapshot', arguments: {} }));
      const r = body(await c.callTool({ name: 'memory_restore', arguments: { file: s.file, namespace: 'copy' } }));
      assert.equal(r.records, 1);
      const st = body(await c.callTool({ name: 'memory_stats', arguments: { namespace: 'copy' } }));
      assert.equal(st.records, 1);
      const v = body(await c.callTool({ name: 'memory_verify', arguments: {} }));
      assert.ok(v.witness.ok);
    },
    { MEM_DENY: 'memory_compact' },
  );
});

test('mcp server reports an unavailable embedder instead of crashing', async () => {
  const root = tmpDir('mcp2');
  await withServer(
    root,
    'standard',
    async (c) => {
      const info = body(await c.callTool({ name: 'memory_info', arguments: {} }));
      assert.equal(info.embedder.active, null);
      assert.match(info.embedder.error, /unknown model|could not load/);
      const r = (await c.callTool({ name: 'memory_recall', arguments: { text: 'x' } })) as any;
      assert.ok(r.isError);
      assert.equal(body(r).error, 'EMBEDDER_UNAVAILABLE');
    },
    { MEM_EMBEDDER: 'onnx:does-not-exist' },
  );
});
