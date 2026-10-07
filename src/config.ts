import path from 'node:path';
import os from 'node:os';

export type Profile = 'read-only' | 'standard' | 'administrative';

export interface EngineConfig {
  root: string;
  namespace: string;
  embedder: string;
  allowFallback: boolean;
  profile: Profile;
  allow: string[];
  deny: string[];
  capacity: number;
  learning: boolean;
  actor: string;
}

/**
 * Configuration precedence: explicit overrides > environment > defaults.
 *   MEM_ROOT            directory holding namespaces (default ~/.agent-memory; set MEM_ROOT=.memory for per-project memory)
 *   MEM_NAMESPACE       default namespace (default "default")
 *   MEM_EMBEDDER        onnx:<model>[@dtype] | ngram[:dim]  (default onnx:all-MiniLM-L6-v2)
 *   MEM_ALLOW_FALLBACK  1 to permit the lexical fallback when the model cannot load (off by default: fail loudly)
 *   MEM_PROFILE         read-only | standard | administrative (default standard)
 *   MEM_ALLOW / MEM_DENY comma-separated tool names
 *   MEM_CAPACITY        max records before compaction (0 = unlimited)
 *   MEM_LEARNING        1 to let recorded outcomes influence ranking
 *   MEM_ACTOR           actor name written to the witness log
 */
export function loadConfig(overrides: Partial<EngineConfig> = {}): EngineConfig {
  const env = process.env;
  const defaultRoot = path.join(os.homedir(), '.agent-memory');
  const profile = (overrides.profile ?? env.MEM_PROFILE ?? 'standard') as Profile;
  if (!['read-only', 'standard', 'administrative'].includes(profile)) throw new Error(`invalid MEM_PROFILE ${profile}`);
  return {
    root: path.resolve(overrides.root ?? env.MEM_ROOT ?? defaultRoot),
    namespace: overrides.namespace ?? env.MEM_NAMESPACE ?? 'default',
    embedder: overrides.embedder ?? env.MEM_EMBEDDER ?? 'onnx:all-MiniLM-L6-v2',
    allowFallback: overrides.allowFallback ?? env.MEM_ALLOW_FALLBACK === '1',
    profile,
    allow: overrides.allow ?? splitList(env.MEM_ALLOW),
    deny: overrides.deny ?? splitList(env.MEM_DENY),
    capacity: overrides.capacity ?? (env.MEM_CAPACITY ? parseInt(env.MEM_CAPACITY, 10) : 0),
    learning: overrides.learning ?? env.MEM_LEARNING === '1',
    actor: overrides.actor ?? env.MEM_ACTOR ?? 'agent',
  };
}

function splitList(s?: string): string[] {
  return (s ?? '')
    .split(',')
    .map((x) => x.trim())
    .filter(Boolean);
}
