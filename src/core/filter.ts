import type { Filter, FilterOp, Metadata } from './types.js';
import { MemError } from './errors.js';

const OPS = new Set(['$eq', '$ne', '$in', '$nin', '$gt', '$gte', '$lt', '$lte', '$prefix', '$exists', '$contains', '$all']);

function isOpObject(v: unknown): v is FilterOp {
  if (!v || typeof v !== 'object' || Array.isArray(v)) return false;
  const keys = Object.keys(v);
  return keys.length > 0 && keys.every((k) => k.startsWith('$'));
}

function eq(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((x, i) => eq(x, b[i]));
  if (a && b && typeof a === 'object') return JSON.stringify(a) === JSON.stringify(b);
  return false;
}

function cmp(a: unknown, b: unknown): number | null {
  if (typeof a === 'number' && typeof b === 'number') return a - b;
  if (typeof a === 'string' && typeof b === 'string') return a < b ? -1 : a > b ? 1 : 0;
  return null;
}

export function validateFilter(filter: Filter | undefined): void {
  if (!filter) return;
  for (const [key, v] of Object.entries(filter)) {
    if (isOpObject(v)) {
      for (const op of Object.keys(v)) {
        if (!OPS.has(op)) throw new MemError('UNSUPPORTED', `unsupported filter operator ${op} on key ${key}`);
      }
    }
  }
}

export function matchesFilter(meta: Metadata, filter: Filter): boolean {
  for (const [key, cond] of Object.entries(filter)) {
    const v = meta[key];
    if (!isOpObject(cond)) {
      if (Array.isArray(v) && !Array.isArray(cond)) {
        if (!v.some((x) => eq(x, cond))) return false;
      } else if (!eq(v, cond)) return false;
      continue;
    }
    const op = cond as FilterOp;
    if ('$exists' in op && (v !== undefined) !== op.$exists) return false;
    if ('$eq' in op && !eq(v, op.$eq)) return false;
    if ('$ne' in op && eq(v, op.$ne)) return false;
    if (op.$in && !op.$in.some((x) => (Array.isArray(v) ? v.some((y) => eq(y, x)) : eq(v, x)))) return false;
    if (op.$nin && op.$nin.some((x) => (Array.isArray(v) ? v.some((y) => eq(y, x)) : eq(v, x)))) return false;
    if (op.$gt !== undefined) { const c = cmp(v, op.$gt); if (c === null || !(c > 0)) return false; }
    if (op.$gte !== undefined) { const c = cmp(v, op.$gte); if (c === null || !(c >= 0)) return false; }
    if (op.$lt !== undefined) { const c = cmp(v, op.$lt); if (c === null || !(c < 0)) return false; }
    if (op.$lte !== undefined) { const c = cmp(v, op.$lte); if (c === null || !(c <= 0)) return false; }
    if (op.$prefix !== undefined && !(typeof v === 'string' && v.startsWith(op.$prefix))) return false;
    if ('$contains' in op && !(Array.isArray(v) && v.some((x) => eq(x, op.$contains)))) return false;
    if (op.$all && !(Array.isArray(v) && op.$all.every((x) => v.some((y) => eq(y, x))))) return false;
  }
  return true;
}

/**
 * Extract the equality-resolvable part of a filter so the metadata index can estimate cardinality.
 * Returns a list of (key, candidate values) pairs: a record must match one value per pair.
 */
export function equalityTerms(filter: Filter): Array<{ key: string; values: unknown[] }> {
  const out: Array<{ key: string; values: unknown[] }> = [];
  for (const [key, cond] of Object.entries(filter)) {
    if (!isOpObject(cond)) {
      if (Array.isArray(cond)) continue; // exact-array equality not indexable
      out.push({ key, values: [cond] });
      continue;
    }
    const op = cond as FilterOp;
    if ('$eq' in op && !Array.isArray(op.$eq)) out.push({ key, values: [op.$eq] });
    else if (op.$in) out.push({ key, values: op.$in });
    else if ('$contains' in op) out.push({ key, values: [op.$contains] });
    else if (op.$all && op.$all.length) out.push({ key, values: [op.$all[0]] });
  }
  return out;
}

export function indexKey(v: unknown): string | null {
  switch (typeof v) {
    case 'string':
      return 's:' + v;
    case 'number':
      return 'n:' + v;
    case 'boolean':
      return 'b:' + v;
    default:
      return null;
  }
}
