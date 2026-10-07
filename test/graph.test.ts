import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { GraphStore } from '../src/core/graph-store.js';
import { parseCypher } from '../src/core/cypher.js';
import { tmpDir } from './helpers.js';

function seed(g: GraphStore) {
  g.batchInsert({
    nodes: [
      { id: 'alice', labels: ['Person'], properties: { name: 'Alice', age: 34, text: 'Alice the engineer' }, embedding: [1, 0, 0, 0] },
      { id: 'bob', labels: ['Person'], properties: { name: 'Bob', age: 28 }, embedding: [0, 1, 0, 0] },
      { id: 'carol', labels: ['Person', 'Manager'], properties: { name: 'Carol', age: 45 } },
      { id: 'proj', labels: ['Project'], properties: { name: 'Memory Engine', tags: ['rust', 'ts'] } },
    ],
    edges: [
      { id: 'e1', from: 'alice', to: 'proj', type: 'WORKS_ON', confidence: 0.9, properties: { since: 2024 } },
      { id: 'e2', from: 'bob', to: 'proj', type: 'WORKS_ON', confidence: 0.5 },
      { id: 'e3', from: 'carol', to: 'alice', type: 'MANAGES' },
      { id: 'e4', from: 'carol', to: 'bob', type: 'MANAGES' },
    ],
    hyperedges: [{ id: 'h1', nodes: ['alice', 'bob', 'carol'], type: 'MEETING', properties: { topic: 'planning' }, embedding: [0, 0, 1, 0] }],
  });
}

test('graph persists and hydrates on reopen', () => {
  const dir = tmpDir('graph');
  let g = GraphStore.openOrCreate(path.join(dir, 'g'), { dimensions: 4 });
  seed(g);
  assert.equal(g.stats().nodes, 4);
  g.close();
  g = GraphStore.openOrCreate(path.join(dir, 'g'), { dimensions: 4 });
  const s = g.stats();
  assert.deepEqual([s.nodes, s.edges, s.hyperedges], [4, 4, 1], 'reopened graph is not empty');
  assert.equal(g.getNode('alice')!.embedding!.length, 4);
  // log tail without checkpoint also hydrates
  g.createNode({ id: 'dave', labels: ['Person'], properties: {} });
  const g2 = GraphStore.openOrCreate(path.join(dir, 'g'), { dimensions: 4 });
  assert.ok(g2.hasNode('dave'));
  assert.throws(() => GraphStore.openOrCreate(path.join(dir, 'g'), { dimensions: 5 }), (e: any) => e.code === 'DIMENSION_MISMATCH');
});

test('graph traversal and similarity', () => {
  const g = GraphStore.openOrCreate(path.join(tmpDir('graph2'), 'g'), { dimensions: 4 });
  seed(g);
  const hops = g.kHopNeighbors('carol', 1, { includeHyperedges: false });
  assert.deepEqual(hops.map((h) => h.id).sort(), ['alice', 'bob']);
  const two = g.kHopNeighbors('carol', 2, { includeHyperedges: false });
  assert.ok(two.some((h) => h.id === 'proj' && h.hops === 2));
  const out = g.kHopNeighbors('alice', 1, { direction: 'out', includeHyperedges: false });
  assert.deepEqual(out.map((h) => h.id), ['proj']);
  assert.ok(g.kHopNeighbors('alice', 1).some((h) => h.id === 'carol'), 'hyperedge membership counts as a hop');
  assert.equal(g.searchNodes([0.9, 0.1, 0, 0], 1)[0].node.id, 'alice');
  assert.equal(g.searchHyperedges([0, 0, 1, 0], 1)[0].hyperedge.id, 'h1');
  assert.throws(() => g.createEdge({ from: 'alice', to: 'nobody', type: 'X' }), (e: any) => e.code === 'NOT_FOUND');
  assert.throws(() => g.createNode({ id: 'bad', properties: { nested: { a: 1 } as any } }), (e: any) => e.code === 'INVALID_ARGUMENT');
  // delete cascades
  g.deleteNode('proj');
  assert.equal(g.stats().edges, 2);
  // transaction rollback
  assert.throws(() => g.transaction(() => { g.createNode({ id: 'tx', properties: {} }); throw new Error('boom'); }));
  assert.ok(!g.hasNode('tx'));
  let notified = 0;
  g.subscribe(() => notified++);
  g.createNode({ id: 'n', properties: {} });
  assert.equal(notified, 1);
});

test('cypher subset executes what it parses and rejects the rest by name', () => {
  const g = GraphStore.openOrCreate(path.join(tmpDir('cypher'), 'g'), { dimensions: 4 });
  seed(g);
  const q = (c: string) => g.query(c);
  assert.equal(q('MATCH (n) RETURN n').rows.length, 4);
  assert.equal(q('MATCH (n:Person) RETURN n.name ORDER BY n.age DESC').rows.map((r) => r[0]).join(','), 'Carol,Alice,Bob');
  assert.equal(q("MATCH (n) WHERE n.id = 'alice' RETURN n.name").rows[0][0], 'Alice');
  assert.equal(q("MATCH (n:Person {name: 'Bob'}) RETURN n.age").rows[0][0], 28);
  assert.equal(q('MATCH (p:Person)-[r:WORKS_ON]->(x:Project) WHERE r.confidence > 0.6 RETURN p.name, x.name, r.since').rows[0].join('|'), 'Alice|Memory Engine|2024');
  assert.equal(q('MATCH (m)-[:MANAGES]->(p)-[:WORKS_ON]->(x) RETURN DISTINCT m.name').rows.length, 1, 'chained patterns');
  assert.equal(q('MATCH (p)<-[:MANAGES]-(m) RETURN p.name ORDER BY p.name').rows.map((r) => r[0]).join(','), 'Alice,Bob');
  assert.equal(q('MATCH (p:Person) RETURN count(*) AS c').rows[0][0], 3);
  assert.equal(q('MATCH (m)-[:MANAGES]->(p) RETURN m.name, count(p) AS reports, collect(p.name) AS names').rows[0][1], 2);
  assert.equal(q('MATCH (p:Person) RETURN avg(p.age) AS a').rows[0][0], (34 + 28 + 45) / 3);
  assert.equal(q('MATCH (p:Person) RETURN p.name ORDER BY p.age SKIP 1 LIMIT 1').rows[0][0], 'Alice');
  assert.equal(q("MATCH (p:Person) WHERE p.name STARTS WITH 'A' OR p.age IN [28] RETURN p.name ORDER BY p.name").rows.length, 2);
  assert.equal(q("MATCH (p:Person) WHERE p.name CONTAINS 'ar' AND NOT p.age < 40 RETURN p.name").rows[0][0], 'Carol');
  assert.equal(q('MATCH (p:Person) WHERE p.missing IS NULL RETURN count(*) AS c').rows[0][0], 3);
  assert.equal(q('MATCH (p:Person) RETURN toLower(p.name) AS n, p.age * 2 + 1 AS x ORDER BY x LIMIT 1').rows[0].join(','), 'bob,57');
  assert.equal(q('MATCH (a:Person), (b:Project) RETURN count(*) AS c').rows[0][0], 3, 'cartesian product of comma patterns');
  assert.equal(q('MATCH (a)-[r]->(b) RETURN type(r), labels(b) ORDER BY type(r)').rows.length, 4);
  const res = q('MATCH (p:Person)-[r:WORKS_ON]->(x) RETURN p, r LIMIT 1');
  assert.equal(res.nodes.length, 1);
  assert.equal(res.edges.length, 1);
  for (const bad of ["CREATE (n:Person {name: 'x'})", 'MATCH (n) SET n.x = 1 RETURN n', 'MATCH (a)-[*1..3]->(b) RETURN b', "MATCH (n) WHERE n.name =~ 'A.*' RETURN n", 'MATCH (n) WITH n RETURN n', 'OPTIONAL MATCH (n) RETURN n']) {
    assert.throws(() => q(bad), (e: any) => e.code === 'UNSUPPORTED' && e.message.length > 20, bad);
  }
  assert.throws(() => q('MATCH (n RETURN n'), (e: any) => e.code === 'INVALID_ARGUMENT');
  assert.throws(() => q('MATCH (n) RETURN m.name'), (e: any) => e.code === 'INVALID_ARGUMENT');
  const ast = parseCypher('MATCH (n:A)-[r:X|Y]->(m) RETURN n ORDER BY n.z SKIP 2 LIMIT 3');
  assert.equal(ast.patterns[0].rels[0].types.length, 2);
  assert.equal(ast.limit, 3);
});
