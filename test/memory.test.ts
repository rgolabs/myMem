import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { AgentMemory } from '../src/memory/agent-memory.js';
import { NgramEmbedder } from '../src/embedding/ngram.js';
import { tmpDir } from './helpers.js';

async function open(root: string, namespace = 'ns', extra: Partial<ConstructorParameters<typeof AgentMemory>[0]> = {}) {
  const e = new NgramEmbedder(128);
  await e.init();
  return new AgentMemory({ root, namespace, embedder: e, ...extra });
}

test('remember / recall / supersede / time travel / budget', async () => {
  const root = tmpDir('mem');
  const m = await open(root);
  const a = await m.remember({ text: 'Customer inference must stay in Canada', kind: 'decision', tags: ['compliance'], source: 'meeting', importance: 0.9 });
  assert.equal(a.novelty, 1);
  await m.remember({ text: 'Deploys go through GitHub Actions canary', kind: 'fact', tags: ['infra'] });
  const dup = await m.remember({ text: 'Customer inference must stay in Canada', kind: 'decision' });
  assert.ok(dup.similar.some((s) => s.id === a.id), 'near-duplicate reported');
  assert.ok(dup.novelty < 0.05);
  m.forget(dup.id);
  const c = await m.remember({ text: 'Customer inference must stay in Canada and backups in the EU', kind: 'decision', supersedes: a.id });
  assert.equal(c.supersededId, a.id);
  const r = await m.recall({ text: 'customer inference canada', k: 5 });
  assert.ok(!r.results.some((h) => h.id === a.id), 'superseded hidden');
  assert.ok(r.results.some((h) => h.id === c.id));
  assert.equal(r.results[0].metadata.text, undefined, 'text not duplicated inside metadata');
  assert.ok((await m.recall({ text: 'customer inference canada', k: 5, includeSuperseded: true })).results.some((h) => h.id === a.id));
  const past = await m.recall({ text: 'customer inference canada', k: 5, asOf: Date.now() - 86_400_000 });
  assert.equal(past.results.length, 0);
  assert.ok((await m.recall({ text: 'canada', k: 5, kinds: ['fact'] })).results.every((h) => h.kind === 'fact'));
  assert.ok((await m.recall({ text: 'canada', k: 5, tags: ['infra'] })).results.every((h) => h.tags.includes('infra')));
  const budget = await m.recall({ text: 'canada deploy', k: 5, budgetTokens: 12 });
  assert.ok(budget.results.reduce((s, h) => s + Number(h.metadata.tokens), 0) <= 12);
  assert.equal(m.graph.query("MATCH (a)-[r:SUPERSEDES]->(b) RETURN a.id, b.id").rows.length, 1);
  assert.throws(() => m.list({ filter: { x: { $bogus: 1 } } }), (e: any) => e.code === 'UNSUPPORTED');
  await assert.rejects(m.remember({ text: 'x', metadata: { kind: 'hack' } }), (e: any) => e.code === 'INVALID_ARGUMENT');
  const upd = await m.update(c.id, { text: 'Customer inference must stay in Canada; backups in EU; logs in US', tags: ['compliance', 'logging'] });
  assert.ok(upd.tags.includes('logging'));
  assert.ok((await m.recall({ text: 'logs in US', k: 1 })).results[0].id === c.id);
  m.close();
  const m2 = await open(root);
  assert.equal(m2.store.count(), 3);
  assert.ok((await m2.verify()).witness.ok);
  m2.close();
});

test('episodes, skills, causal edges, consolidation', async () => {
  const m = await open(tmpDir('mem2'));
  for (const mod of ['auth', 'payments', 'billing']) await m.storeEpisode({ task: `Fix flaky test in ${mod}`, critique: 'check the mock clock first', outcome: 'success', reward: 0.8, tags: ['testing'] });
  await m.storeEpisode({ task: 'Write release notes', critique: 'ask for the changelog', outcome: 'failure', reward: -0.5 });
  const ep = await m.retrieveEpisodes('flaky test', 3);
  assert.ok(ep.results.every((h) => h.kind === 'episode'));
  const sk = await m.createSkill({ name: 'Rotate key', description: 'rotate vendor api key and update secret store' });
  assert.equal(m.recordSkillOutcome(sk.id, true).successRate, 1);
  assert.equal(m.recordSkillOutcome(sk.id, false).successRate, 0.5);
  const found = await m.searchSkills('rotate api key', 1);
  assert.equal(found.results[0].usageCount, 2);
  const ce = await m.addCausalEdge({ causes: ['missing mock clock'], effects: ['flaky tests'], confidence: 0.5, context: 'timers' });
  const cq = await m.queryCausal('flaky tests mock clock', 1);
  assert.equal(cq[0].id, ce.id);
  assert.ok(m.recordCausalOutcome(ce.id, true).confidence > 0.5);
  const dry = await m.consolidate({ dryRun: true, similarityThreshold: 0.5 });
  assert.ok(dry.clusters.some((c) => c.size >= 3 && c.eligible));
  const done = await m.consolidate({ similarityThreshold: 0.5 });
  assert.equal(done.created.length, 1);
  assert.equal(m.graph.query('MATCH (p:Procedure)-[:DERIVED_FROM]->(e) RETURN count(e) AS n').rows[0][0], 3);
  assert.equal(m.list({ kinds: ['procedure'] }).total, 1);
  const again = await m.consolidate({ similarityThreshold: 0.5 });
  assert.equal(again.created.length, 0, 'already consolidated episodes are skipped');
  m.close();
});

test('sessions expire with their turn records', async () => {
  const m = await open(tmpDir('mem3'));
  const s = m.startSession({ ttlMinutes: 0.0005 });
  await m.appendTurn({ sessionId: s.id, role: 'user', content: 'deploy to staging', remember: true });
  assert.equal(m.list({ kinds: ['turn'] }).total, 1);
  assert.ok((await m.recall({ text: 'deploy staging', k: 1, sessionId: s.id })).results.length === 1);
  await new Promise((r) => setTimeout(r, 60));
  assert.equal(m.getSession(s.id), null);
  assert.equal((await m.recall({ text: 'deploy staging', k: 1 })).results.length, 0, 'expired turn hidden');
  const ex = m.expireSessions();
  assert.equal(m.list({ kinds: ['turn'] }).total, 0);
  assert.ok(ex.expiredRecords >= 0);
  const s2 = m.startSession({});
  await m.appendTurn({ sessionId: s2.id, role: 'assistant', content: 'done' });
  const end = await m.endSession(s2.id, { summary: 'Deployed fine' });
  assert.ok(end.summaryId);
  m.close();
});

test('learning: reads never change ranking unless enabled; outcomes are witnessed', async () => {
  const root = tmpDir('mem4');
  const m = await open(root, 'ns', { learning: false });
  const ids: string[] = [];
  for (let i = 0; i < 6; i++) ids.push((await m.remember({ text: `note about topic ${i} alpha beta`, kind: 'note' })).id);
  const before = await m.recall({ text: 'topic alpha beta', k: 6 });
  const order = before.results.map((h) => h.id);
  const last = order[order.length - 1];
  for (let i = 0; i < 5; i++) m.recordOutcome({ queryId: before.queryId, chosenIds: [last], reward: 1 });
  const after = await m.recall({ text: 'topic alpha beta', k: 6 });
  assert.deepEqual(after.results.map((h) => h.id), order, 'learning disabled: identical order');
  m.setLearning(true);
  const boosted = await m.recall({ text: 'topic alpha beta', k: 6 });
  assert.ok(boosted.results.findIndex((h) => h.id === last) < order.length - 1, 'learning enabled: feedback promotes the chosen record');
  m.resetLearning('all');
  const reset = await m.recall({ text: 'topic alpha beta', k: 6 });
  assert.deepEqual(reset.results.map((h) => h.id), order);
  const ops: string[] = [];
  for await (const e of m.witness.entries()) ops.push(e.operation);
  assert.equal(ops.filter((o) => o === 'outcome').length, 5);
  assert.ok(ops.includes('learning-toggle') && ops.includes('learning-reset'));
  const r = m.recordOutcome({ chosenIds: [], reward: 0.5, context: { stateKey: 'q', action: 'a' } });
  assert.equal(r.routing!.value, 0.5);
  assert.equal(m.recommendAction('q', ['a', 'b'], 0).recommended, 'a');
  m.close();
});

test('snapshot, restore, branches, purge', async () => {
  const root = tmpDir('mem5');
  const m = await open(root);
  const a = await m.remember({ text: 'alpha fact', kind: 'fact', relations: [{ to: 'person:alice', type: 'ABOUT' }] });
  const b = await m.remember({ text: 'beta fact', kind: 'fact' });
  const snap = m.snapshot();
  assert.ok(fs.existsSync(snap.file));
  assert.equal(snap.records, 2);
  // branch
  m.createBranch('exp');
  await m.remember({ text: 'gamma branch-only fact', branch: 'exp' });
  m.forget(b.id, { branch: 'exp' });
  assert.ok(m.get(b.id), 'parent untouched');
  const br = await m.recall({ text: 'gamma', k: 3, branch: 'exp' });
  assert.ok(br.results.some((h) => h.text.includes('gamma')) && !br.results.some((h) => h.id === b.id));
  await m.update(b.id, { tags: ['changed-in-parent'] });
  const conflict = m.mergeBranch('exp');
  assert.equal(conflict.conflicts.length, 1);
  assert.equal(conflict.conflicts[0].id, b.id);
  const forced = m.mergeBranch('exp', { force: true });
  assert.equal(forced.applied.deleted, 1);
  assert.ok(!m.get(b.id));
  assert.ok(m.list({}).results.some((h) => h.text.includes('gamma')));
  assert.equal(m.listBranches().length, 0);
  // purge everywhere rewrites the snapshot
  const p = m.purge(a.id, { everywhere: true });
  assert.ok(p.removedFrom.includes('memory') && p.removedFrom.includes('graph') && p.removedFrom.some((x) => x.startsWith('snapshot:')));
  assert.ok(p.unreachable.length >= 1);
  m.close();
  // restore into another namespace
  const r = AgentMemory.restore(snap.file, root, { namespace: 'restored' });
  assert.equal(r.records, 1, 'purged record is gone from the snapshot');
  const m2 = await open(root, 'restored');
  assert.equal(m2.store.count(), 1);
  assert.ok(m2.graph.hasNode('person:alice'));
  assert.ok((await m2.verify()).witness.ok);
  assert.throws(() => AgentMemory.restore(snap.file, root, { namespace: 'restored' }), (e: any) => e.code === 'ALREADY_EXISTS');
  m2.close();
  // corrupt snapshot is refused
  const buf = fs.readFileSync(snap.file);
  buf[buf.length - 20] ^= 0xff;
  fs.writeFileSync(snap.file + '.bad', buf);
  assert.throws(() => AgentMemory.restore(snap.file + '.bad', root, { namespace: 'x' }), (e: any) => e.code === 'CORRUPT');
});

test('two processes share one namespace through the log', async () => {
  const root = tmpDir('mem6');
  const m = await open(root);
  await m.remember({ text: 'first from process A', id: 'a1' });
  const script = `
    import { AgentMemory } from ${JSON.stringify(path.resolve('src/memory/agent-memory.ts'))};
    import { NgramEmbedder } from ${JSON.stringify(path.resolve('src/embedding/ngram.ts'))};
    const e = new NgramEmbedder(128); await e.init();
    const m = new AgentMemory({ root: ${JSON.stringify(root)}, namespace: 'ns', embedder: e });
    if (!m.get('a1')) throw new Error('process B cannot see a1');
    await m.remember({ text: 'second from process B', id: 'b1' });
    m.graph.createNode({ id: 'from-b', properties: {} });
    m.close();
  `;
  const file = path.join(root, 'b.mts');
  fs.writeFileSync(file, script);
  execFileSync(process.execPath, [path.resolve('node_modules/.bin/tsx'), file], { stdio: 'pipe' });
  assert.ok(m.get('b1'), 'process A sees B write without reopening');
  assert.ok(m.graph.hasNode('from-b'));
  assert.ok((await m.recall({ text: 'second process', k: 1 })).results[0].id === 'b1');
  assert.ok((await m.verify()).witness.ok);
  m.close();
});
