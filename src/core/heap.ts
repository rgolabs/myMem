/** Binary heap keyed by a number. `max=true` keeps the largest at the top. */
export class Heap<T> {
  private keys: number[] = [];
  private vals: T[] = [];
  constructor(private readonly max: boolean) {}
  get size() {
    return this.keys.length;
  }
  peekKey(): number {
    return this.keys[0];
  }
  peek(): T {
    return this.vals[0];
  }
  push(key: number, val: T) {
    this.keys.push(key);
    this.vals.push(val);
    let i = this.keys.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.better(this.keys[i], this.keys[p])) {
        this.swap(i, p);
        i = p;
      } else break;
    }
  }
  pop(): [number, T] {
    const k = this.keys[0];
    const v = this.vals[0];
    const lk = this.keys.pop()!;
    const lv = this.vals.pop()!;
    if (this.keys.length > 0) {
      this.keys[0] = lk;
      this.vals[0] = lv;
      let i = 0;
      const n = this.keys.length;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let b = i;
        if (l < n && this.better(this.keys[l], this.keys[b])) b = l;
        if (r < n && this.better(this.keys[r], this.keys[b])) b = r;
        if (b === i) break;
        this.swap(i, b);
        i = b;
      }
    }
    return [k, v];
  }
  /** Drain in heap order (top first). */
  drain(): Array<[number, T]> {
    const out: Array<[number, T]> = [];
    while (this.size) out.push(this.pop());
    return out;
  }
  entries(): Array<[number, T]> {
    return this.keys.map((k, i) => [k, this.vals[i]]);
  }
  private better(a: number, b: number) {
    return this.max ? a > b : a < b;
  }
  private swap(i: number, j: number) {
    [this.keys[i], this.keys[j]] = [this.keys[j], this.keys[i]];
    [this.vals[i], this.vals[j]] = [this.vals[j], this.vals[i]];
  }
}
