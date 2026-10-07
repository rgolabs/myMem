import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { spawn, type ChildProcess } from 'node:child_process';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { tmpDir } from './helpers.js';

const CLI = path.resolve('src/cli.ts');
const TSX = path.resolve('node_modules/.bin/tsx');

async function startSse(root: string, extra: string[] = [], env: Record<string, string> = {}): Promise<{ proc: ChildProcess; port: number; stop: () => Promise<void> }> {
  const port = 30000 + Math.floor(Math.random() * 20000);
  const proc = spawn(TSX, [CLI, 'sse', '--root', root, '--port', String(port), ...extra], { env: { ...process.env, MEM_EMBEDDER: 'ngram', ...env }, stdio: ['ignore', 'pipe', 'pipe'] });
  let err = '';
  await new Promise<void>((resolve, reject) => {
    const t = setTimeout(() => reject(new Error('server did not start: ' + err)), 30_000);
    proc.stderr!.on('data', (d) => {
      err += String(d);
      if (err.includes('HTTP MCP listening')) {
        clearTimeout(t);
        resolve();
      }
    });
    proc.on('exit', (c) => reject(new Error(`server exited ${c}: ${err}`)));
  });
  return { proc, port, stop: () => new Promise((r) => { proc.once('exit', () => r()); proc.kill('SIGTERM'); }) };
}
const body = (r: any) => JSON.parse(r.content[0].text);

test('mem sse: legacy SSE and Streamable HTTP clients share one memory', async () => {
  const root = tmpDir('sse');
  const s = await startSse(root);
  try {
    const health = await (await fetch(`http://127.0.0.1:${s.port}/health`)).json();
    assert.equal(health.ok, true);
    assert.deepEqual(health.transport, ['sse', 'streamable-http']);

    const sseClient = new Client({ name: 'sse-test', version: '0' });
    await sseClient.connect(new SSEClientTransport(new URL(`http://127.0.0.1:${s.port}/sse`)));
    const tools = (await sseClient.listTools()).tools.map((t) => t.name);
    assert.ok(tools.includes('memory_remember'));
    assert.match(sseClient.getInstructions() ?? '', /RECALL FIRST/);
    const a = body(await sseClient.callTool({ name: 'memory_remember', arguments: { text: 'The staging URL is staging.example.test', kind: 'fact', tags: ['infra'] } }));
    assert.ok(a.id);

    const httpClient = new Client({ name: 'http-test', version: '0' });
    await httpClient.connect(new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${s.port}/mcp`)));
    const r = body(await httpClient.callTool({ name: 'memory_recall', arguments: { text: 'what is the staging url', k: 1 } }));
    assert.equal(r.results[0].id, a.id, 'second transport sees the first transport\'s write');

    const h2 = await (await fetch(`http://127.0.0.1:${s.port}/health`)).json();
    assert.equal(h2.sessions, 2);
    await sseClient.close();
    await httpClient.close();

    const bad = await fetch(`http://127.0.0.1:${s.port}/messages?sessionId=nope`, { method: 'POST', body: '{}' });
    assert.equal(bad.status, 404);
    const unknown = await fetch(`http://127.0.0.1:${s.port}/mcp`, { method: 'POST', headers: { 'mcp-session-id': 'nope', 'content-type': 'application/json' }, body: '{}' });
    assert.equal(unknown.status, 404);
  } finally {
    await s.stop();
  }
});

test('mem sse: bearer token is enforced', async () => {
  const root = tmpDir('sse-auth');
  const s = await startSse(root, ['--token', 's3cret']);
  try {
    const denied = await fetch(`http://127.0.0.1:${s.port}/sse`);
    assert.equal(denied.status, 401);
    const ok = new Client({ name: 't', version: '0' });
    await ok.connect(new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${s.port}/mcp`), { requestInit: { headers: { Authorization: 'Bearer s3cret' } } }));
    assert.ok((await ok.listTools()).tools.length > 0);
    await ok.close();
    const qs = new Client({ name: 't2', version: '0' });
    await qs.connect(new SSEClientTransport(new URL(`http://127.0.0.1:${s.port}/sse?token=s3cret`), { requestInit: { headers: { Authorization: 'Bearer s3cret' } } }));
    assert.ok((await qs.listTools()).tools.length > 0);
    await qs.close();
    const health = await fetch(`http://127.0.0.1:${s.port}/health`);
    assert.equal(health.status, 200, 'health stays public');
  } finally {
    await s.stop();
  }
});
