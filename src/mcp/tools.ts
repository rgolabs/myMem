/**
 * MCP tool definitions over AgentMemory. Each tool belongs to a profile (read-only < standard <
 * administrative); allow and deny lists are applied by name; the live list is reported by memory_info.
 */
import { z } from 'zod';
import type { AgentMemory } from '../memory/agent-memory.js';
import type { Profile } from '../config.js';

export type Level = 'read-only' | 'standard' | 'administrative';
const LEVEL_RANK: Record<Level, number> = { 'read-only': 0, standard: 1, administrative: 2 };

export interface ToolDef {
  name: string;
  level: Level;
  title: string;
  description: string;
  schema: z.ZodRawShape;
  readOnlyHint?: boolean;
  destructiveHint?: boolean;
  /** false: the tool runs without opening a namespace (mem is null). */
  needsMemory?: boolean;
  run: (mem: AgentMemory, args: any, ctx: ToolContext) => Promise<unknown> | unknown;
}

export interface ToolContext {
  /** Open another namespace (for restore / cross-namespace operations). */
  open: (namespace: string) => AgentMemory;
  root: string;
  restore: (file: string, namespace?: string, overwrite?: boolean) => unknown;
  reembed: (mem: AgentMemory, to: string) => Promise<unknown>;
  toolNames: () => string[];
  profile: Profile;
  embedderInfo: () => Record<string, unknown>;
}

const ns = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/).optional().describe('Namespace (isolated memory store). Defaults to the server namespace.');
const branch = z.string().optional().describe('Operate on a copy-on-write branch instead of the main store.');
const ts = z.union([z.string(), z.number()]).optional();
const filterSchema = z.record(z.string(), z.any()).optional().describe('Structured metadata filter: {key: value} or {key: {$in|$nin|$gt|$gte|$lt|$lte|$prefix|$exists|$contains|$all|$ne}}.');

export function allowed(level: Level, profile: Profile): boolean {
  return LEVEL_RANK[level] <= LEVEL_RANK[profile];
}

export const TOOLS: ToolDef[] = [
  // ---------------------------------------------------------------- core memory
  {
    name: 'memory_recall',
    level: 'read-only',
    title: 'Recall memories',
    description:
      'Semantic + keyword recall of stored memories (facts, decisions, preferences, episodes, skills, notes). Call this FIRST, before searching files, databases or the web, whenever the answer may have been learned or decided earlier. Returns ranked hits with similarity, kind, tags and a queryId for memory_record_outcome.',
    schema: {
      text: z.string().min(1).describe('What you are trying to remember, phrased as a question or topic.'),
      k: z.number().int().min(1).max(100).optional().describe('Number of results (default 5).'),
      kinds: z.array(z.string()).optional().describe('Restrict to kinds, e.g. ["decision","fact"].'),
      tags: z.array(z.string()).optional().describe('Require all of these tags.'),
      filter: filterSchema,
      since: ts.describe('Only memories created at/after this time (ISO or ms).'),
      until: ts.describe('Only memories created at/before this time.'),
      asOf: ts.describe('Time-travel: memories valid at this instant (uses validFrom/validTo).'),
      includeSuperseded: z.boolean().optional().describe('Include memories that were superseded by newer ones (default false).'),
      decay: z.union([z.boolean(), z.object({ halfLifeDays: z.number().positive() })]).optional().describe('Prefer recent memories; true = 30-day half-life.'),
      hybrid: z.boolean().optional().describe('Fuse BM25 keyword ranking with the vector ranking (default true; good for identifiers and names).'),
      diversity: z.number().min(0).max(1).optional().describe('Maximal-marginal-relevance diversity, 0 (off) to 1.'),
      coherence: z.number().min(0).max(2).optional().describe('Boost memories supported by many similar memories (weight).'),
      budgetTokens: z.number().int().positive().optional().describe('Pack the best memories that fit in this many tokens.'),
      minSimilarity: z.number().min(-1).max(1).optional().describe('Drop hits below this cosine similarity.'),
      capabilities: z.union([z.number(), z.string()]).optional().describe('Held capability mask (bigint as string) for gated records.'),
      sessionId: z.string().optional().describe('Restrict to memories written in this session.'),
      expandGraph: z.number().int().min(0).max(3).optional().describe('Attach graph neighbours within N hops to each hit.'),
      explain: z.boolean().optional().describe('Include score explanations.'),
      branch,
      namespace: ns,
    },
    readOnlyHint: true,
    run: (mem, a) => mem.recall(a),
  },
  {
    name: 'memory_remember',
    level: 'standard',
    title: 'Remember',
    description:
      'Store something worth remembering across sessions: a fact, decision, user preference, constraint, lesson, or outcome. Write one memory per fact, in plain language with the context needed to understand it later. Returns the id, an importance score, and any near-duplicate memories (consider `supersedes` to replace an outdated one instead of adding a conflicting duplicate).',
    schema: {
      text: z.string().min(1).max(32768).describe('The memory, self-contained.'),
      kind: z.string().regex(/^[A-Za-z][A-Za-z0-9_-]{0,31}$/).optional().describe('fact | decision | preference | constraint | lesson | note | entity | procedure | episode (default fact).'),
      tags: z.array(z.string()).optional(),
      source: z.string().optional().describe('Where this came from (file, conversation, URL, person).'),
      confidence: z.number().min(0).max(1).optional(),
      importance: z.number().min(0).max(1).optional().describe('Your rating of how important this is; combined with novelty.'),
      metadata: z.record(z.string(), z.any()).optional().describe('Extra filterable fields (project, tenant, ...).'),
      id: z.string().max(512).optional().describe('Stable id to upsert (default: generated).'),
      supersedes: z.string().optional().describe('Id of an older memory this one replaces; the old one is kept for audit but hidden from recall.'),
      validFrom: ts,
      validTo: ts,
      expiresAt: ts.describe('Drop from recall after this time.'),
      pinned: z.boolean().optional().describe('Never evict during compaction.'),
      capabilities: z.union([z.number(), z.string()]).optional().describe('Required capability mask to read this record.'),
      sessionId: z.string().optional(),
      relations: z.array(z.object({ to: z.string(), type: z.string(), confidence: z.number().min(0).max(1).optional() })).optional().describe('Graph edges to other memory ids or entity ids (e.g. person:alice).'),
      branch,
      namespace: ns,
    },
    run: (mem, a) => mem.remember(a),
  },
  {
    name: 'memory_get',
    level: 'read-only',
    title: 'Get memory by id',
    description: 'Fetch one memory by id, including metadata.',
    schema: { id: z.string(), branch, namespace: ns },
    readOnlyHint: true,
    run: (mem, a) => mem.get(a.id, { branch: a.branch }) ?? { error: 'NOT_FOUND', id: a.id },
  },
  {
    name: 'memory_list',
    level: 'read-only',
    title: 'List memories',
    description: 'Browse memories by kind, tags or filter with paging (no similarity ranking).',
    schema: { kinds: z.array(z.string()).optional(), tags: z.array(z.string()).optional(), filter: filterSchema, limit: z.number().int().min(1).max(1000).optional(), offset: z.number().int().min(0).optional(), includeSuperseded: z.boolean().optional(), branch, namespace: ns },
    readOnlyHint: true,
    run: (mem, a) => mem.list(a),
  },
  {
    name: 'memory_update',
    level: 'standard',
    title: 'Update memory',
    description: 'Patch a memory: text (re-embedded), tags, confidence, importance, pinned, validTo or extra metadata.',
    schema: {
      id: z.string(),
      text: z.string().optional(),
      tags: z.array(z.string()).optional(),
      confidence: z.number().min(0).max(1).optional(),
      importance: z.number().min(0).max(1).optional(),
      pinned: z.boolean().optional(),
      validTo: ts,
      metadata: z.record(z.string(), z.any()).optional(),
      branch,
      namespace: ns,
    },
    run: (mem, a) => mem.update(a.id, a),
  },
  {
    name: 'memory_forget',
    level: 'standard',
    title: 'Forget memory',
    description: 'Delete one memory from the live store (copies in snapshots remain; use memory_purge to remove everywhere).',
    schema: { id: z.string(), branch, namespace: ns },
    destructiveHint: true,
    run: (mem, a) => ({ id: a.id, deleted: mem.forget(a.id, { branch: a.branch }) }),
  },
  // ---------------------------------------------------------------- episodes / skills / causal
  {
    name: 'memory_store_episode',
    level: 'standard',
    title: 'Store episode',
    description: 'Record what happened in a task (Reflexion-style): task, actions, observations, self-critique, outcome and reward. Call at the end of a task so the lesson is available next time.',
    schema: {
      task: z.string().min(1),
      actions: z.array(z.string()).optional(),
      observations: z.array(z.string()).optional(),
      critique: z.string().optional().describe('What you would do differently / what worked.'),
      outcome: z.string().optional().describe('success | failure | partial, or free text.'),
      reward: z.number().min(-1).max(1).optional(),
      tags: z.array(z.string()).optional(),
      sessionId: z.string().optional(),
      metadata: z.record(z.string(), z.any()).optional(),
      namespace: ns,
    },
    run: (mem, a) => mem.storeEpisode(a),
  },
  {
    name: 'memory_retrieve_episodes',
    level: 'read-only',
    title: 'Retrieve episodes',
    description: 'Find past episodes similar to the current task, newest first on ties.',
    schema: { text: z.string().min(1), k: z.number().int().min(1).max(50).optional(), tags: z.array(z.string()).optional(), since: ts, namespace: ns },
    readOnlyHint: true,
    run: (mem, a) => mem.retrieveEpisodes(a.text, a.k ?? 5, { tags: a.tags, since: a.since }),
  },
  {
    name: 'memory_create_skill',
    level: 'standard',
    title: 'Create skill',
    description: 'Store a reusable procedure (name, description, parameters, examples). Success rate and usage count start at zero and change only through memory_record_skill_outcome.',
    schema: { name: z.string().min(1), description: z.string().min(1), parameters: z.record(z.string(), z.any()).optional(), examples: z.array(z.string()).optional(), tags: z.array(z.string()).optional(), id: z.string().optional(), namespace: ns },
    run: (mem, a) => mem.createSkill(a),
  },
  {
    name: 'memory_search_skills',
    level: 'read-only',
    title: 'Search skills',
    description: 'Find relevant skills with their success rate and usage count.',
    schema: { text: z.string().min(1), k: z.number().int().min(1).max(50).optional(), namespace: ns },
    readOnlyHint: true,
    run: (mem, a) => mem.searchSkills(a.text, a.k ?? 5),
  },
  {
    name: 'memory_record_skill_outcome',
    level: 'standard',
    title: 'Record skill outcome',
    description: 'Report whether using a skill succeeded; updates its success rate and usage count (the only path that does).',
    schema: { skillId: z.string(), success: z.boolean(), namespace: ns },
    run: (mem, a) => mem.recordSkillOutcome(a.skillId, a.success),
  },
  {
    name: 'memory_add_causal_edge',
    level: 'standard',
    title: 'Add causal edge',
    description: 'Record a cause-and-effect relation (hyperedge) with confidence and context, e.g. causes ["missing index"] effects ["slow query"].',
    schema: { causes: z.array(z.string()).min(1), effects: z.array(z.string()).min(1), confidence: z.number().min(0).max(1).optional(), context: z.string().optional(), id: z.string().optional(), namespace: ns },
    run: (mem, a) => mem.addCausalEdge(a),
  },
  {
    name: 'memory_query_causal',
    level: 'read-only',
    title: 'Query causal edges',
    description: 'Find causal relations relevant to a situation, ranked by similarity × confidence.',
    schema: { text: z.string().min(1), k: z.number().int().min(1).max(50).optional(), namespace: ns },
    readOnlyHint: true,
    run: (mem, a) => mem.queryCausal(a.text, a.k ?? 5),
  },
  {
    name: 'memory_record_causal_outcome',
    level: 'standard',
    title: 'Record causal outcome',
    description: 'Confirm or refute a causal edge; moves its confidence accordingly.',
    schema: { id: z.string(), confirmed: z.boolean(), namespace: ns },
    run: (mem, a) => mem.recordCausalOutcome(a.id, a.confirmed),
  },
  // ---------------------------------------------------------------- graph
  {
    name: 'graph_create_node',
    level: 'standard',
    title: 'Create graph node',
    description: 'Create or upsert an entity node (person, project, file, concept) with labels and typed properties; optional text is embedded for similarity search.',
    schema: { id: z.string().optional(), labels: z.array(z.string()).optional(), properties: z.record(z.string(), z.any()).optional(), text: z.string().optional().describe('Text to embed for graph_search_nodes.'), upsert: z.boolean().optional(), namespace: ns },
    run: async (mem, a) => {
      const embedding = a.text ? await mem.embedder.embedPassage(a.text) : undefined;
      const props = { ...(a.properties ?? {}), ...(a.text ? { text: a.text } : {}) };
      return a.upsert && a.id ? mem.graph.upsertNode({ id: a.id, labels: a.labels, properties: props, embedding }) : mem.graph.createNode({ id: a.id, labels: a.labels, properties: props, embedding });
    },
  },
  {
    name: 'graph_create_edge',
    level: 'standard',
    title: 'Create graph edge',
    description: 'Create a directed, typed relation between two existing nodes (memory ids are node ids too).',
    schema: { from: z.string(), to: z.string(), type: z.string(), confidence: z.number().min(0).max(1).optional(), properties: z.record(z.string(), z.any()).optional(), id: z.string().optional(), namespace: ns },
    run: (mem, a) => mem.graph.createEdge(a),
  },
  {
    name: 'graph_create_hyperedge',
    level: 'standard',
    title: 'Create hyperedge',
    description: 'One relation over N nodes (a meeting, a collaboration, a causal set).',
    schema: { nodes: z.array(z.string()).min(2), type: z.string(), confidence: z.number().min(0).max(1).optional(), properties: z.record(z.string(), z.any()).optional(), text: z.string().optional(), id: z.string().optional(), namespace: ns },
    run: async (mem, a) => mem.graph.createHyperedge({ ...a, properties: { ...(a.properties ?? {}), ...(a.text ? { text: a.text } : {}) }, embedding: a.text ? await mem.embedder.embedPassage(a.text) : undefined }),
  },
  {
    name: 'graph_get_node',
    level: 'read-only',
    title: 'Get node',
    description: 'Fetch a node with its edges and hyperedges.',
    schema: { id: z.string(), namespace: ns },
    readOnlyHint: true,
    run: (mem, a) => {
      const n = mem.graph.getNode(a.id);
      if (!n) return { error: 'NOT_FOUND', id: a.id };
      const { embedding, ...node } = n;
      return { node, edges: mem.graph.edgesOf(a.id).map(({ embedding: _e, ...e }) => e), hyperedges: mem.graph.hyperedgesOf(a.id).map(({ embedding: _e, ...h }) => h) };
    },
  },
  {
    name: 'graph_query',
    level: 'read-only',
    title: 'Graph query (Cypher subset)',
    description: 'Run a read-only Cypher-subset query: MATCH (n:Label {k:v})-[r:TYPE]->(m) WHERE ... RETURN ... ORDER BY ... SKIP ... LIMIT ... with count()/collect(). Unsupported constructs raise an error naming them.',
    schema: { cypher: z.string().min(1), maxRows: z.number().int().min(1).max(5000).optional(), namespace: ns },
    readOnlyHint: true,
    run: (mem, a) => {
      const r = mem.graph.query(a.cypher, { maxRows: a.maxRows });
      return { ...r, nodes: r.nodes.map(({ embedding, ...n }) => n), edges: r.edges.map(({ embedding, ...e }) => e) };
    },
  },
  {
    name: 'graph_k_hop',
    level: 'read-only',
    title: 'k-hop neighbours',
    description: 'Node ids reachable within k hops (optionally restricted to edge types / direction).',
    schema: { id: z.string(), k: z.number().int().min(0).max(6), types: z.array(z.string()).optional(), direction: z.enum(['out', 'in', 'both']).optional(), limit: z.number().int().optional(), namespace: ns },
    readOnlyHint: true,
    run: (mem, a) => mem.graph.kHopNeighbors(a.id, a.k, a),
  },
  {
    name: 'graph_search_nodes',
    level: 'read-only',
    title: 'Search nodes by similarity',
    description: 'Nodes whose embedded text is most similar to the query.',
    schema: { text: z.string().min(1), k: z.number().int().min(1).max(100).optional(), labels: z.array(z.string()).optional(), namespace: ns },
    readOnlyHint: true,
    run: async (mem, a) => mem.graph.searchNodes(await mem.embedder.embedQuery(a.text), a.k ?? 5, { labels: a.labels }).map((h) => ({ id: h.node.id, labels: h.node.labels, properties: h.node.properties, similarity: h.similarity })),
  },
  {
    name: 'graph_delete',
    level: 'standard',
    title: 'Delete graph element',
    description: 'Delete a node (and its edges), an edge, or a hyperedge.',
    schema: { id: z.string(), kind: z.enum(['node', 'edge', 'hyperedge']), namespace: ns },
    destructiveHint: true,
    run: (mem, a) => ({ id: a.id, deleted: a.kind === 'node' ? mem.graph.deleteNode(a.id) : a.kind === 'edge' ? mem.graph.deleteEdge(a.id) : mem.graph.deleteHyperedge(a.id) }),
  },
  // ---------------------------------------------------------------- sessions (working memory)
  {
    name: 'session_start',
    level: 'standard',
    title: 'Start session',
    description: 'Open a working-memory session with a time-to-live; turns appended to it expire with it.',
    schema: { id: z.string().optional(), ttlMinutes: z.number().positive().optional(), metadata: z.record(z.string(), z.any()).optional(), namespace: ns },
    run: (mem, a) => mem.startSession(a),
  },
  {
    name: 'session_append_turn',
    level: 'standard',
    title: 'Append turn',
    description: 'Append a conversation turn or tool result to a session. remember=true also makes it recallable until the session expires.',
    schema: { sessionId: z.string(), role: z.string(), content: z.string(), toolCalls: z.array(z.any()).optional(), remember: z.boolean().optional(), namespace: ns },
    run: (mem, a) => mem.appendTurn(a),
  },
  {
    name: 'session_get',
    level: 'read-only',
    title: 'Get session',
    description: 'Read a session and its most recent turns.',
    schema: { sessionId: z.string(), lastN: z.number().int().positive().optional(), namespace: ns },
    readOnlyHint: true,
    run: (mem, a) => mem.getSession(a.sessionId, { lastN: a.lastN }) ?? { error: 'NOT_FOUND', sessionId: a.sessionId },
  },
  {
    name: 'session_list',
    level: 'read-only',
    title: 'List sessions',
    description: 'List live (unexpired) sessions.',
    schema: { namespace: ns },
    readOnlyHint: true,
    run: (mem) => mem.listSessions(),
  },
  {
    name: 'session_end',
    level: 'standard',
    title: 'End session',
    description: 'Close a session, drop its temporary turn records, and optionally store a summary as a durable episode.',
    schema: { sessionId: z.string(), summary: z.string().optional(), tags: z.array(z.string()).optional(), namespace: ns },
    run: (mem, a) => mem.endSession(a.sessionId, a),
  },
  // ---------------------------------------------------------------- learning
  {
    name: 'memory_record_outcome',
    level: 'standard',
    title: 'Record outcome',
    description: 'Feedback on a recall: which results were actually useful and a reward in [-1,1]. This is the only way recall learns; reads never change memory. Optionally updates an action value for a state key (outcome-aware routing).',
    schema: { queryId: z.string().optional(), resultIds: z.array(z.string()).optional(), chosenIds: z.array(z.string()), reward: z.number().min(-1).max(1), context: z.object({ stateKey: z.string().optional(), action: z.string().optional() }).optional(), namespace: ns },
    run: (mem, a) => mem.recordOutcome(a),
  },
  {
    name: 'memory_recommend_action',
    level: 'read-only',
    title: 'Recommend action',
    description: 'Epsilon-greedy choice among candidate actions for a state key, from recorded outcomes.',
    schema: { stateKey: z.string(), candidates: z.array(z.string()).min(1), epsilon: z.number().min(0).max(1).optional(), namespace: ns },
    readOnlyHint: true,
    run: (mem, a) => mem.recommendAction(a.stateKey, a.candidates, a.epsilon),
  },
  {
    name: 'memory_learning',
    level: 'administrative',
    title: 'Learning switch',
    description: 'Enable/disable feedback-informed ranking, or reset learned state (routing values, feedback scores).',
    schema: { action: z.enum(['enable', 'disable', 'reset']), scope: z.enum(['routing', 'feedback', 'all']).optional(), namespace: ns },
    run: (mem, a) => (a.action === 'reset' ? mem.resetLearning(a.scope ?? 'all') : mem.setLearning(a.action === 'enable')),
  },
  // ---------------------------------------------------------------- lifecycle / admin
  {
    name: 'memory_consolidate',
    level: 'administrative',
    title: 'Consolidate episodes',
    description: 'Cluster similar episodes and promote repeated successful patterns into durable procedures linked to their sources. dryRun returns the clusters so you can write a better summary with memory_remember (metadata.derivedFrom).',
    schema: { minClusterSize: z.number().int().min(2).optional(), similarityThreshold: z.number().min(0).max(1).optional(), minSuccessRate: z.number().min(0).max(1).optional(), kinds: z.array(z.string()).optional(), dryRun: z.boolean().optional(), maxClusters: z.number().int().positive().optional(), namespace: ns },
    run: (mem, a) => mem.consolidate(a),
  },
  {
    name: 'memory_compact',
    level: 'administrative',
    title: 'Compact',
    description: 'Evict the least valuable memories down to a target count using lru | lfu | coherence policy (pinned memories survive; evictions are audited).',
    schema: { target: z.number().int().min(0), policy: z.enum(['lru', 'lfu', 'coherence']).optional(), namespace: ns },
    destructiveHint: true,
    run: (mem, a) => mem.compact(a),
  },
  {
    name: 'memory_expire',
    level: 'administrative',
    title: 'Expire',
    description: 'Drop expired sessions and expired records now (they are also hidden from recall automatically).',
    schema: { namespace: ns },
    run: (mem) => mem.expireSessions(),
  },
  {
    name: 'memory_snapshot',
    level: 'standard',
    title: 'Snapshot',
    description: 'Write a checksummed, compressed snapshot of the namespace (records, graph, sessions, learned state, audit head). Default location: <namespace>/snapshots/.',
    schema: { file: z.string().optional(), namespace: ns },
    run: (mem, a) => mem.snapshot(a.file),
  },
  {
    name: 'memory_restore',
    level: 'administrative',
    title: 'Restore snapshot',
    description: 'Restore a snapshot into a namespace (refuses to overwrite unless overwrite=true).',
    schema: { file: z.string(), namespace: ns.describe('Target namespace (default: the one recorded in the snapshot).'), overwrite: z.boolean().optional() },
    destructiveHint: true,
    needsMemory: false,
    run: (_mem, a, ctx) => ctx.restore(a.file, a.namespace, a.overwrite),
  },
  {
    name: 'memory_branch',
    level: 'standard',
    title: 'Branches',
    description: 'Copy-on-write branches for experiments: create, list, merge (reports conflicts by id), discard. Use the `branch` argument on remember/recall to work inside one.',
    schema: { action: z.enum(['create', 'list', 'merge', 'discard']), name: z.string().optional(), force: z.boolean().optional(), namespace: ns },
    run: (mem, a, ctx) => {
      switch (a.action) {
        case 'create':
          return mem.createBranch(required(a.name, 'name'));
        case 'list':
          return mem.listBranches();
        case 'merge':
          if (a.force && ctx.profile !== 'administrative') throw new Error('force merge requires the administrative profile');
          return mem.mergeBranch(required(a.name, 'name'), { force: a.force });
        case 'discard':
          if (ctx.profile !== 'administrative') throw new Error('discard requires the administrative profile');
          return mem.discardBranch(required(a.name, 'name'));
      }
    },
  },
  {
    name: 'memory_purge',
    level: 'administrative',
    title: 'Purge everywhere',
    description: 'Remove a record from the live store, graph, branches and namespace snapshots; reports what could not be reached.',
    schema: { id: z.string(), everywhere: z.boolean().optional(), namespace: ns },
    destructiveHint: true,
    run: (mem, a) => mem.purge(a.id, { everywhere: a.everywhere ?? true }),
  },
  {
    name: 'memory_reembed',
    level: 'administrative',
    title: 'Re-embed',
    description: 'Migrate the namespace to another embedding model (e.g. onnx:bge-small-en-v1.5). Re-reads source text and rebuilds vectors and graph embeddings.',
    schema: { to: z.string().describe('Embedder spec: onnx:<model>[@dtype] | ngram[:dim]'), namespace: ns },
    run: (mem, a, ctx) => ctx.reembed(mem, a.to),
  },
  {
    name: 'memory_verify',
    level: 'read-only',
    title: 'Verify audit chain',
    description: 'Recompute the hash-linked witness log and report the first break, plus store and graph counts.',
    schema: { namespace: ns },
    readOnlyHint: true,
    run: (mem) => mem.verify(),
  },
  {
    name: 'memory_stats',
    level: 'read-only',
    title: 'Stats',
    description: 'Record counts by kind, index parameters, graph stats, sessions, learning state, audit head, branches.',
    schema: { namespace: ns },
    readOnlyHint: true,
    run: (mem) => mem.stats(),
  },
  {
    name: 'memory_info',
    level: 'read-only',
    title: 'Server info',
    description: 'Which embedder is active (and whether it is the lexical fallback), the storage root, the profile, and the live tool list.',
    schema: {},
    readOnlyHint: true,
    needsMemory: false,
    run: (_mem, _a, ctx) => ({ root: ctx.root, profile: ctx.profile, tools: ctx.toolNames(), embedder: ctx.embedderInfo() }),
  },
];

function required<T>(v: T | undefined, name: string): T {
  if (v === undefined || v === null || v === '') throw new Error(`${name} is required`);
  return v;
}
