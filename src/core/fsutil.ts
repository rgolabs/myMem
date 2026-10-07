import fs from 'node:fs';
import path from 'node:path';
import { MemError } from './errors.js';

/** Write a file atomically: write to a temp sibling, fsync, rename over the target. */
export function writeFileAtomic(file: string, data: string | Buffer): void {
  const dir = path.dirname(file);
  fs.mkdirSync(dir, { recursive: true });
  const tmp = path.join(dir, `.${path.basename(file)}.${process.pid}.${Date.now()}.tmp`);
  const fd = fs.openSync(tmp, 'w');
  try {
    fs.writeFileSync(fd, data);
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
  fs.renameSync(tmp, file);
}

/** Append one line to a log file with O_APPEND and fsync. Single write call so the line is all-or-nothing on local filesystems. */
export function appendLineSync(file: string, line: string): void {
  const fd = fs.openSync(file, 'a');
  try {
    fs.writeSync(fd, line.endsWith('\n') ? line : line + '\n');
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
}

export function readJsonIfExists<T>(file: string): T | null {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8')) as T;
  } catch (e: any) {
    if (e?.code === 'ENOENT') return null;
    throw new MemError('CORRUPT', `cannot read ${file}: ${e?.message ?? e}`);
  }
}

/** Identity of a file's current content (inode, size, mtime); changes whenever it is atomically replaced. */
export function fileStamp(file: string): string {
  try {
    const st = fs.statSync(file);
    return `${st.ino}:${st.size}:${st.mtimeMs}`;
  } catch {
    return 'none';
  }
}

export function fileSize(file: string): number {
  try {
    return fs.statSync(file).size;
  } catch {
    return 0;
  }
}

/** Read bytes [from, EOF) of a file. */
export function readFrom(file: string, from: number): Buffer {
  let fd: number;
  try {
    fd = fs.openSync(file, 'r');
  } catch (e: any) {
    if (e?.code === 'ENOENT') return Buffer.alloc(0);
    throw e;
  }
  try {
    const size = fs.fstatSync(fd).size;
    if (size <= from) return Buffer.alloc(0);
    const buf = Buffer.alloc(size - from);
    let off = 0;
    while (off < buf.length) {
      const n = fs.readSync(fd, buf, off, buf.length - off, from + off);
      if (n <= 0) break;
      off += n;
    }
    return buf.subarray(0, off);
  } finally {
    fs.closeSync(fd);
  }
}

/** Read the last line of a file (ignoring a trailing newline). */
export function readLastLine(file: string): string | null {
  const size = fileSize(file);
  if (size === 0) return null;
  let chunk = 4096;
  for (;;) {
    const from = Math.max(0, size - chunk);
    const buf = readFrom(file, from).toString('utf8');
    const trimmed = buf.endsWith('\n') ? buf.slice(0, -1) : buf;
    const idx = trimmed.lastIndexOf('\n');
    if (idx >= 0 || from === 0) return trimmed.slice(idx + 1);
    chunk *= 4;
  }
}

/** Confine a user-supplied path under a root (spec §14 path safety). */
export function confinePath(root: string, p: string): string {
  const r = path.resolve(root);
  const full = path.resolve(r, p);
  if (full !== r && !full.startsWith(r + path.sep)) {
    throw new MemError('PATH_UNSAFE', `path ${p} escapes root ${root}`);
  }
  return full;
}

function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (e: any) {
    return e?.code === 'EPERM';
  }
}

/**
 * Advisory cross-process lock built on atomic `mkdir`. Blocks (spin with backoff) up to `timeoutMs`.
 * A lock whose owner pid is dead, or that is older than `staleMs`, is broken.
 */
export function withLock<T>(dir: string, fn: () => T, opts: { timeoutMs?: number; staleMs?: number; name?: string } = {}): T {
  const lockDir = path.join(dir, opts.name ? `.lock-${opts.name}` : '.lock');
  const ownerFile = path.join(lockDir, 'owner');
  const timeoutMs = opts.timeoutMs ?? 10_000;
  const staleMs = opts.staleMs ?? 60_000;
  const start = Date.now();
  let wait = 2;
  for (;;) {
    try {
      fs.mkdirSync(lockDir);
      fs.writeFileSync(ownerFile, JSON.stringify({ pid: process.pid, at: Date.now() }));
      break;
    } catch (e: any) {
      if (e?.code !== 'EEXIST') throw e;
      let stale = false;
      try {
        const owner = JSON.parse(fs.readFileSync(ownerFile, 'utf8')) as { pid: number; at: number };
        if (!pidAlive(owner.pid) || Date.now() - owner.at > staleMs) stale = true;
      } catch {
        const st = (() => {
          try {
            return fs.statSync(lockDir);
          } catch {
            return null;
          }
        })();
        if (!st || Date.now() - st.mtimeMs > 2_000) stale = true;
      }
      if (stale) {
        try {
          fs.rmSync(lockDir, { recursive: true, force: true });
        } catch {}
        continue;
      }
      if (Date.now() - start > timeoutMs) {
        throw new MemError('LOCKED', `could not acquire lock on ${dir} within ${timeoutMs} ms`);
      }
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, wait);
      wait = Math.min(50, wait * 2);
    }
  }
  try {
    return fn();
  } finally {
    try {
      fs.rmSync(lockDir, { recursive: true, force: true });
    } catch {}
  }
}
