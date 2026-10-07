/**
 * Append-only, hash-linked audit log (spec §10). Tamper-evident, not confidential.
 * hash = sha256(prevHash || seq || operation || recordId || payloadHash || timestamp || actor)
 */
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { appendLineSync, readLastLine, withLock } from './fsutil.js';

export interface WitnessEntry {
  seq: number;
  prevHash: string;
  hash: string;
  operation: string;
  recordId: string;
  timestamp: string;
  actor: string;
  payloadHash: string;
}

export const GENESIS = '0'.repeat(64);

export function sha256(s: string | Buffer): string {
  return crypto.createHash('sha256').update(s).digest('hex');
}

export function computeHash(e: Omit<WitnessEntry, 'hash'>): string {
  return sha256(`${e.prevHash}|${e.seq}|${e.operation}|${e.recordId}|${e.payloadHash}|${e.timestamp}|${e.actor}`);
}

export class WitnessLog {
  private head: { seq: number; hash: string } = { seq: 0, hash: GENESIS };
  readonly file: string;
  constructor(readonly dir: string) {
    fs.mkdirSync(dir, { recursive: true });
    this.file = path.join(dir, 'witness.jsonl');
    this.loadHead();
  }

  private loadHead() {
    const last = readLastLine(this.file);
    if (!last) return;
    try {
      const e = JSON.parse(last) as WitnessEntry;
      this.head = { seq: e.seq, hash: e.hash };
    } catch {
      // torn tail line: verify() will report it; keep genesis head so we never chain onto garbage
    }
  }

  headInfo() {
    this.loadHead();
    return { ...this.head };
  }

  append(operation: string, recordId: string, payload: unknown, actor = 'local'): WitnessEntry {
    return this.appendMany([{ operation, recordId, payload }], actor)[0];
  }

  /** Append several entries under one lock and one fsync (one transaction's worth of writes). */
  appendMany(items: Array<{ operation: string; recordId: string; payload?: unknown }>, actor = 'local'): WitnessEntry[] {
    if (!items.length) return [];
    return withLock(this.dir, () => {
      this.loadHead();
      const timestamp = new Date().toISOString();
      const out: WitnessEntry[] = [];
      let lines = '';
      for (const it of items) {
        const base: Omit<WitnessEntry, 'hash'> = {
          seq: this.head.seq + 1,
          prevHash: this.head.hash,
          operation: it.operation,
          recordId: it.recordId,
          timestamp,
          actor,
          payloadHash: sha256(JSON.stringify(it.payload ?? null)),
        };
        const entry: WitnessEntry = { ...base, hash: computeHash(base) };
        lines += JSON.stringify(entry) + '\n';
        this.head = { seq: entry.seq, hash: entry.hash };
        out.push(entry);
      }
      appendLineSync(this.file, lines);
      return out;
    }, { name: 'witness' });
  }

  /** Recompute the chain; report the first break. */
  async verify(): Promise<{ ok: boolean; entries: number; head: string; firstBreak?: { seq: number; reason: string } }> {
    let prev = GENESIS;
    let seq = 0;
    let count = 0;
    if (!fs.existsSync(this.file)) return { ok: true, entries: 0, head: GENESIS };
    const rl = readline.createInterface({ input: fs.createReadStream(this.file), crlfDelay: Infinity });
    for await (const line of rl) {
      if (!line.trim()) continue;
      let e: WitnessEntry;
      try {
        e = JSON.parse(line);
      } catch {
        return { ok: false, entries: count, head: prev, firstBreak: { seq: seq + 1, reason: 'unparseable line' } };
      }
      if (e.seq !== seq + 1) return { ok: false, entries: count, head: prev, firstBreak: { seq: e.seq, reason: `sequence gap (expected ${seq + 1})` } };
      if (e.prevHash !== prev) return { ok: false, entries: count, head: prev, firstBreak: { seq: e.seq, reason: 'prevHash mismatch' } };
      const { hash, ...rest } = e;
      if (computeHash(rest) !== hash) return { ok: false, entries: count, head: prev, firstBreak: { seq: e.seq, reason: 'hash mismatch' } };
      prev = hash;
      seq = e.seq;
      count++;
    }
    return { ok: true, entries: count, head: prev };
  }

  /** Stream entries (optionally only those after `afterSeq`). */
  async *entries(afterSeq = 0): AsyncGenerator<WitnessEntry> {
    if (!fs.existsSync(this.file)) return;
    const rl = readline.createInterface({ input: fs.createReadStream(this.file), crlfDelay: Infinity });
    for await (const line of rl) {
      if (!line.trim()) continue;
      try {
        const e = JSON.parse(line) as WitnessEntry;
        if (e.seq > afterSeq) yield e;
      } catch {
        return;
      }
    }
  }
}
