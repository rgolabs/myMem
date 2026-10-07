/**
 * A practical Cypher subset (spec §7). Anything outside the subset raises UNSUPPORTED naming the
 * construct; it never returns an empty result silently.
 *
 * Supported:  MATCH (n:Label {k: v})-[r:TYPE|OTHER]->(m) , (chained patterns, several comma patterns)
 *             WHERE with = <> < <= > >= AND OR NOT, arithmetic, property access, CONTAINS,
 *             STARTS WITH, ENDS WITH, IN, IS NULL, IS NOT NULL
 *             RETURN [DISTINCT] expr [AS alias], count(), collect(), sum(), avg(), min(), max(),
 *             id(), labels(), type(), toLower(), toUpper(), size(), coalesce()
 *             ORDER BY ... [ASC|DESC], SKIP n, LIMIT n  (executed, not just parsed)
 * Unsupported: CREATE SET DELETE MERGE REMOVE WITH UNWIND CALL OPTIONAL MATCH, variable-length
 *             paths [*1..3] (use kHopNeighbors), regular expressions (=~), hyperedge patterns.
 */
import { MemError } from './errors.js';

// ------------------------------------------------------------------ tokenizer

type Tok = { t: 'id' | 'str' | 'num' | 'p'; v: string; pos: number };

const PUNCT = ['<>', '<=', '>=', '<-', '->', '=~', '(', ')', '[', ']', '{', '}', ':', ',', '.', '-', '>', '<', '=', '+', '*', '/', '%', '|'];

function tokenize(src: string): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (/\s/.test(ch)) {
      i++;
      continue;
    }
    if (ch === '/' && src[i + 1] === '/') {
      while (i < src.length && src[i] !== '\n') i++;
      continue;
    }
    if (ch === "'" || ch === '"') {
      let j = i + 1;
      let s = '';
      while (j < src.length && src[j] !== ch) {
        if (src[j] === '\\' && j + 1 < src.length) {
          j++;
          s += src[j] === 'n' ? '\n' : src[j];
        } else s += src[j];
        j++;
      }
      if (j >= src.length) throw new MemError('INVALID_ARGUMENT', `unterminated string at ${i}`);
      out.push({ t: 'str', v: s, pos: i });
      i = j + 1;
      continue;
    }
    if (/[0-9]/.test(ch) || (ch === '.' && /[0-9]/.test(src[i + 1] ?? ''))) {
      let j = i;
      while (j < src.length && /[0-9.eE]/.test(src[j])) j++;
      out.push({ t: 'num', v: src.slice(i, j), pos: i });
      i = j;
      continue;
    }
    if (/[A-Za-z_]/.test(ch)) {
      let j = i;
      while (j < src.length && /[A-Za-z0-9_]/.test(src[j])) j++;
      out.push({ t: 'id', v: src.slice(i, j), pos: i });
      i = j;
      continue;
    }
    if (ch === '`') {
      const j = src.indexOf('`', i + 1);
      if (j < 0) throw new MemError('INVALID_ARGUMENT', `unterminated backtick at ${i}`);
      out.push({ t: 'id', v: src.slice(i + 1, j), pos: i });
      i = j + 1;
      continue;
    }
    const p = PUNCT.find((x) => src.startsWith(x, i));
    if (!p) throw new MemError('INVALID_ARGUMENT', `unexpected character '${ch}' at ${i}`);
    out.push({ t: 'p', v: p, pos: i });
    i += p.length;
  }
  return out;
}

// ------------------------------------------------------------------ AST

export type Expr =
  | { k: 'lit'; v: unknown }
  | { k: 'var'; name: string }
  | { k: 'prop'; obj: Expr; name: string }
  | { k: 'bin'; op: string; l: Expr; r: Expr }
  | { k: 'not'; e: Expr }
  | { k: 'neg'; e: Expr }
  | { k: 'isnull'; e: Expr; not: boolean }
  | { k: 'list'; items: Expr[] }
  | { k: 'fn'; name: string; args: Expr[]; star?: boolean };

interface NodePat {
  v?: string;
  labels: string[];
  props: Record<string, unknown>;
}
interface RelPat {
  v?: string;
  types: string[];
  props: Record<string, unknown>;
  dir: 'out' | 'in' | 'both';
}
interface Pattern {
  nodes: NodePat[];
  rels: RelPat[];
}
interface RetItem {
  e: Expr;
  alias: string;
}
export interface Query {
  patterns: Pattern[];
  where?: Expr;
  distinct: boolean;
  returnAll: boolean;
  items: RetItem[];
  orderBy: Array<{ e: Expr; desc: boolean }>;
  skip?: number;
  limit?: number;
}

const AGG = new Set(['count', 'collect', 'sum', 'avg', 'min', 'max']);
const UNSUPPORTED_CLAUSES = new Set(['CREATE', 'SET', 'DELETE', 'MERGE', 'REMOVE', 'WITH', 'UNWIND', 'CALL', 'OPTIONAL', 'DETACH', 'FOREACH', 'UNION']);

class Parser {
  private i = 0;
  constructor(private toks: Tok[]) {}

  private peek(o = 0) {
    return this.toks[this.i + o];
  }
  private isKw(kw: string, o = 0) {
    const t = this.peek(o);
    return !!t && t.t === 'id' && t.v.toUpperCase() === kw;
  }
  private isP(p: string, o = 0) {
    const t = this.peek(o);
    return !!t && t.t === 'p' && t.v === p;
  }
  private kw(kw: string) {
    if (!this.isKw(kw)) this.fail(`expected ${kw}`);
    this.i++;
  }
  private p(p: string) {
    if (!this.isP(p)) this.fail(`expected '${p}'`);
    this.i++;
  }
  private fail(msg: string): never {
    const t = this.peek();
    throw new MemError('INVALID_ARGUMENT', `cypher parse error: ${msg} ${t ? `near '${t.v}' at ${t.pos}` : 'at end of query'}`);
  }
  private unsupported(what: string): never {
    throw new MemError('UNSUPPORTED', `cypher construct not supported: ${what}`);
  }

  parse(): Query {
    if (this.peek() && this.peek().t === 'id' && UNSUPPORTED_CLAUSES.has(this.peek().v.toUpperCase())) this.unsupported(this.peek().v.toUpperCase());
    this.kw('MATCH');
    const patterns: Pattern[] = [this.pattern()];
    while (this.isP(',')) {
      this.i++;
      patterns.push(this.pattern());
    }
    if (this.isKw('MATCH')) {
      this.i++;
      patterns.push(this.pattern());
      while (this.isP(',')) {
        this.i++;
        patterns.push(this.pattern());
      }
    }
    let where: Expr | undefined;
    if (this.isKw('WHERE')) {
      this.i++;
      where = this.expr();
    }
    if (this.peek() && this.peek().t === 'id' && UNSUPPORTED_CLAUSES.has(this.peek().v.toUpperCase())) this.unsupported(this.peek().v.toUpperCase());
    this.kw('RETURN');
    let distinct = false;
    if (this.isKw('DISTINCT')) {
      this.i++;
      distinct = true;
    }
    const items: RetItem[] = [];
    let returnAll = false;
    if (this.isP('*')) {
      this.i++;
      returnAll = true;
    } else {
      items.push(this.retItem());
      while (this.isP(',')) {
        this.i++;
        items.push(this.retItem());
      }
    }
    const orderBy: Array<{ e: Expr; desc: boolean }> = [];
    if (this.isKw('ORDER')) {
      this.i++;
      this.kw('BY');
      do {
        if (this.isP(',')) this.i++;
        const e = this.expr();
        let desc = false;
        if (this.isKw('DESC') || this.isKw('DESCENDING')) {
          this.i++;
          desc = true;
        } else if (this.isKw('ASC') || this.isKw('ASCENDING')) this.i++;
        orderBy.push({ e, desc });
      } while (this.isP(','));
    }
    let skip: number | undefined, limit: number | undefined;
    if (this.isKw('SKIP')) {
      this.i++;
      skip = this.int();
    }
    if (this.isKw('LIMIT')) {
      this.i++;
      limit = this.int();
    }
    if (this.isKw('UNION')) this.unsupported('UNION');
    if (this.peek()) this.fail('unexpected trailing input');
    return { patterns, where, distinct, returnAll, items, orderBy, skip, limit };
  }

  private int(): number {
    const t = this.peek();
    if (!t || t.t !== 'num') this.fail('expected integer');
    this.i++;
    return parseInt(t.v, 10);
  }

  private retItem(): RetItem {
    const start = this.peek()?.pos ?? 0;
    const e = this.expr();
    let alias: string;
    if (this.isKw('AS')) {
      this.i++;
      const t = this.peek();
      if (!t || t.t !== 'id') this.fail('expected alias');
      alias = t.v;
      this.i++;
    } else {
      const end = this.peek()?.pos ?? Infinity;
      alias = this.toks
        .filter((t) => t.pos >= start && t.pos < end)
        .map((t) => (t.t === 'str' ? `'${t.v}'` : t.v))
        .join('');
    }
    return { e, alias };
  }

  private pattern(): Pattern {
    const nodes: NodePat[] = [this.node()];
    const rels: RelPat[] = [];
    while (this.isP('-') || this.isP('<-')) {
      rels.push(this.rel());
      nodes.push(this.node());
    }
    return { nodes, rels };
  }

  private node(): NodePat {
    this.p('(');
    const n: NodePat = { labels: [], props: {} };
    if (this.peek()?.t === 'id') {
      n.v = this.peek().v;
      this.i++;
    }
    while (this.isP(':')) {
      this.i++;
      const t = this.peek();
      if (!t || t.t !== 'id') this.fail('expected label');
      n.labels.push(t.v);
      this.i++;
    }
    if (this.isP('{')) n.props = this.propMap();
    this.p(')');
    return n;
  }

  private rel(): RelPat {
    const r: RelPat = { types: [], props: {}, dir: 'both' };
    let incoming = false;
    if (this.isP('<-')) {
      incoming = true;
      this.i++;
    } else this.p('-');
    if (this.isP('[')) {
      this.i++;
      if (this.peek()?.t === 'id') {
        r.v = this.peek().v;
        this.i++;
      }
      if (this.isP(':')) {
        this.i++;
        const t = this.peek();
        if (!t || t.t !== 'id') this.fail('expected relationship type');
        r.types.push(t.v);
        this.i++;
        while (this.isP('|')) {
          this.i++;
          if (this.isP(':')) this.i++;
          const t2 = this.peek();
          if (!t2 || t2.t !== 'id') this.fail('expected relationship type');
          r.types.push(t2.v);
          this.i++;
        }
      }
      if (this.isP('*')) this.unsupported('variable-length path [*] (use graph_k_hop instead)');
      if (this.isP('{')) r.props = this.propMap();
      this.p(']');
    }
    if (this.isP('->')) {
      this.i++;
      if (incoming) this.unsupported('bidirectional arrow <-[]->');
      r.dir = 'out';
    } else {
      this.p('-');
      r.dir = incoming ? 'in' : 'both';
    }
    return r;
  }

  private propMap(): Record<string, unknown> {
    this.p('{');
    const out: Record<string, unknown> = {};
    while (!this.isP('}')) {
      const k = this.peek();
      if (!k || k.t !== 'id') this.fail('expected property key');
      this.i++;
      this.p(':');
      const e = this.expr();
      if (e.k !== 'lit' && e.k !== 'list') this.unsupported('non-literal inline property value');
      out[k.v] = evalExpr(e, {});
      if (this.isP(',')) this.i++;
    }
    this.p('}');
    return out;
  }

  // expressions
  expr(): Expr {
    return this.or();
  }
  private or(): Expr {
    let l = this.and();
    while (this.isKw('OR')) {
      this.i++;
      l = { k: 'bin', op: 'OR', l, r: this.and() };
    }
    return l;
  }
  private and(): Expr {
    let l = this.not();
    while (this.isKw('AND')) {
      this.i++;
      l = { k: 'bin', op: 'AND', l, r: this.not() };
    }
    return l;
  }
  private not(): Expr {
    if (this.isKw('NOT')) {
      this.i++;
      return { k: 'not', e: this.not() };
    }
    return this.cmp();
  }
  private cmp(): Expr {
    const l = this.add();
    const t = this.peek();
    if (!t) return l;
    if (t.t === 'p' && ['=', '<>', '<', '<=', '>', '>='].includes(t.v)) {
      this.i++;
      return { k: 'bin', op: t.v, l, r: this.add() };
    }
    if (t.t === 'p' && t.v === '=~') this.unsupported('regular expression match =~');
    if (this.isKw('CONTAINS')) {
      this.i++;
      return { k: 'bin', op: 'CONTAINS', l, r: this.add() };
    }
    if (this.isKw('STARTS') && this.isKw('WITH', 1)) {
      this.i += 2;
      return { k: 'bin', op: 'STARTS WITH', l, r: this.add() };
    }
    if (this.isKw('ENDS') && this.isKw('WITH', 1)) {
      this.i += 2;
      return { k: 'bin', op: 'ENDS WITH', l, r: this.add() };
    }
    if (this.isKw('IN')) {
      this.i++;
      return { k: 'bin', op: 'IN', l, r: this.add() };
    }
    if (this.isKw('IS')) {
      this.i++;
      let not = false;
      if (this.isKw('NOT')) {
        this.i++;
        not = true;
      }
      this.kw('NULL');
      return { k: 'isnull', e: l, not };
    }
    return l;
  }
  private add(): Expr {
    let l = this.mul();
    while (this.isP('+') || this.isP('-')) {
      const op = this.peek().v;
      this.i++;
      l = { k: 'bin', op, l, r: this.mul() };
    }
    return l;
  }
  private mul(): Expr {
    let l = this.unary();
    while (this.isP('*') || this.isP('/') || this.isP('%')) {
      const op = this.peek().v;
      this.i++;
      l = { k: 'bin', op, l, r: this.unary() };
    }
    return l;
  }
  private unary(): Expr {
    if (this.isP('-')) {
      this.i++;
      return { k: 'neg', e: this.unary() };
    }
    return this.postfix();
  }
  private postfix(): Expr {
    let e = this.primary();
    while (this.isP('.')) {
      this.i++;
      const t = this.peek();
      if (!t || t.t !== 'id') this.fail('expected property name');
      this.i++;
      e = { k: 'prop', obj: e, name: t.v };
    }
    return e;
  }
  private primary(): Expr {
    const t = this.peek();
    if (!t) this.fail('unexpected end of expression');
    if (t.t === 'str') {
      this.i++;
      return { k: 'lit', v: t.v };
    }
    if (t.t === 'num') {
      this.i++;
      return { k: 'lit', v: Number(t.v) };
    }
    if (t.t === 'p' && t.v === '(') {
      this.i++;
      const e = this.expr();
      this.p(')');
      return e;
    }
    if (t.t === 'p' && t.v === '[') {
      this.i++;
      const items: Expr[] = [];
      while (!this.isP(']')) {
        items.push(this.expr());
        if (this.isP(',')) this.i++;
      }
      this.p(']');
      return { k: 'list', items };
    }
    if (t.t === 'id') {
      const up = t.v.toUpperCase();
      if (up === 'TRUE' || up === 'FALSE' || up === 'NULL') {
        this.i++;
        return { k: 'lit', v: up === 'TRUE' ? true : up === 'FALSE' ? false : null };
      }
      if (this.isP('(', 1)) {
        this.i += 2;
        const name = t.v.toLowerCase();
        const args: Expr[] = [];
        let star = false;
        if (this.isP('*')) {
          this.i++;
          star = true;
        } else if (this.isKw('DISTINCT')) this.unsupported('aggregate DISTINCT');
        while (!this.isP(')')) {
          args.push(this.expr());
          if (this.isP(',')) this.i++;
        }
        this.p(')');
        return { k: 'fn', name, args, star };
      }
      this.i++;
      return { k: 'var', name: t.v };
    }
    this.fail('unexpected token');
  }
}

export function parseCypher(src: string): Query {
  return new Parser(tokenize(src)).parse();
}

// ------------------------------------------------------------------ evaluation

export interface GraphElement {
  id: string;
  labels?: string[];
  type?: string;
  properties: Record<string, unknown>;
  from?: string;
  to?: string;
  confidence?: number;
}
export type Bindings = Record<string, GraphElement>;

function cmpVals(a: unknown, b: unknown): number | null {
  if (a === null || a === undefined || b === null || b === undefined) return null;
  if (typeof a === 'number' && typeof b === 'number') return a - b;
  if (typeof a === 'string' && typeof b === 'string') return a < b ? -1 : a > b ? 1 : 0;
  if (typeof a === 'boolean' && typeof b === 'boolean') return Number(a) - Number(b);
  return null;
}

function eqVals(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((x, i) => eqVals(x, b[i]));
  return false;
}

export function evalExpr(e: Expr, b: Bindings): unknown {
  switch (e.k) {
    case 'lit':
      return e.v;
    case 'var': {
      if (!(e.name in b)) throw new MemError('INVALID_ARGUMENT', `cypher: variable ${e.name} not defined`);
      return b[e.name];
    }
    case 'prop': {
      const o = evalExpr(e.obj, b);
      if (o === null || o === undefined) return null;
      if (typeof o === 'object') {
        const el = o as GraphElement;
        if (el.properties && typeof el.properties === 'object') {
          if (e.name in el.properties) return el.properties[e.name];
          if (e.name === 'id') return el.id;
          if (e.name === 'confidence') return el.confidence ?? null;
          if (e.name === 'type') return el.type ?? null;
          return null;
        }
        return (o as Record<string, unknown>)[e.name] ?? null;
      }
      return null;
    }
    case 'list':
      return e.items.map((x) => evalExpr(x, b));
    case 'not':
      return !truthy(evalExpr(e.e, b));
    case 'neg': {
      const v = evalExpr(e.e, b);
      return typeof v === 'number' ? -v : null;
    }
    case 'isnull': {
      const v = evalExpr(e.e, b);
      const isNull = v === null || v === undefined;
      return e.not ? !isNull : isNull;
    }
    case 'bin': {
      if (e.op === 'AND') return truthy(evalExpr(e.l, b)) && truthy(evalExpr(e.r, b));
      if (e.op === 'OR') return truthy(evalExpr(e.l, b)) || truthy(evalExpr(e.r, b));
      const l = evalExpr(e.l, b);
      const r = evalExpr(e.r, b);
      switch (e.op) {
        case '=':
          return eqVals(l, r);
        case '<>':
          return !eqVals(l, r);
        case '<': {
          const c = cmpVals(l, r);
          return c === null ? null : c < 0;
        }
        case '<=': {
          const c = cmpVals(l, r);
          return c === null ? null : c <= 0;
        }
        case '>': {
          const c = cmpVals(l, r);
          return c === null ? null : c > 0;
        }
        case '>=': {
          const c = cmpVals(l, r);
          return c === null ? null : c >= 0;
        }
        case '+':
          if (typeof l === 'number' && typeof r === 'number') return l + r;
          if (typeof l === 'string' || typeof r === 'string') return String(l ?? '') + String(r ?? '');
          if (Array.isArray(l)) return l.concat(r as unknown[]);
          return null;
        case '-':
          return typeof l === 'number' && typeof r === 'number' ? l - r : null;
        case '*':
          return typeof l === 'number' && typeof r === 'number' ? l * r : null;
        case '/':
          return typeof l === 'number' && typeof r === 'number' ? l / r : null;
        case '%':
          return typeof l === 'number' && typeof r === 'number' ? l % r : null;
        case 'CONTAINS':
          return typeof l === 'string' && typeof r === 'string' ? l.includes(r) : null;
        case 'STARTS WITH':
          return typeof l === 'string' && typeof r === 'string' ? l.startsWith(r) : null;
        case 'ENDS WITH':
          return typeof l === 'string' && typeof r === 'string' ? l.endsWith(r) : null;
        case 'IN':
          return Array.isArray(r) ? r.some((x) => eqVals(x, l)) : null;
      }
      throw new MemError('UNSUPPORTED', `cypher operator ${e.op}`);
    }
    case 'fn': {
      if (AGG.has(e.name)) throw new MemError('INVALID_ARGUMENT', `aggregate ${e.name}() is only allowed in RETURN`);
      const args = e.args.map((a) => evalExpr(a, b));
      switch (e.name) {
        case 'id':
          return (args[0] as GraphElement)?.id ?? null;
        case 'labels':
          return (args[0] as GraphElement)?.labels ?? [];
        case 'type':
          return (args[0] as GraphElement)?.type ?? null;
        case 'tolower':
          return typeof args[0] === 'string' ? args[0].toLowerCase() : null;
        case 'toupper':
          return typeof args[0] === 'string' ? args[0].toUpperCase() : null;
        case 'tostring':
          return args[0] === null || args[0] === undefined ? null : String(args[0]);
        case 'tointeger':
          return args[0] === null ? null : Math.trunc(Number(args[0]));
        case 'tofloat':
          return args[0] === null ? null : Number(args[0]);
        case 'size':
          return Array.isArray(args[0]) || typeof args[0] === 'string' ? args[0].length : null;
        case 'coalesce':
          return args.find((a) => a !== null && a !== undefined) ?? null;
        case 'keys':
          return Object.keys((args[0] as GraphElement)?.properties ?? {});
        case 'properties':
          return (args[0] as GraphElement)?.properties ?? null;
        case 'abs':
          return typeof args[0] === 'number' ? Math.abs(args[0]) : null;
        case 'round':
          return typeof args[0] === 'number' ? Math.round(args[0]) : null;
      }
      throw new MemError('UNSUPPORTED', `cypher function ${e.name}()`);
    }
  }
}

function truthy(v: unknown): boolean {
  return v === true;
}

export function hasAggregate(e: Expr): boolean {
  switch (e.k) {
    case 'fn':
      return AGG.has(e.name) || e.args.some(hasAggregate);
    case 'bin':
      return hasAggregate(e.l) || hasAggregate(e.r);
    case 'not':
    case 'neg':
      return hasAggregate(e.e);
    case 'isnull':
      return hasAggregate(e.e);
    case 'prop':
      return hasAggregate(e.obj);
    case 'list':
      return e.items.some(hasAggregate);
    default:
      return false;
  }
}

/** Evaluate an expression that may contain aggregates over a group of bindings. */
export function evalAggregate(e: Expr, group: Bindings[]): unknown {
  if (e.k === 'fn' && AGG.has(e.name)) {
    if (e.name === 'count') {
      if (e.star || !e.args.length) return group.length;
      return group.filter((b) => {
        const v = evalExpr(e.args[0], b);
        return v !== null && v !== undefined;
      }).length;
    }
    const vals = group.map((b) => evalExpr(e.args[0], b)).filter((v) => v !== null && v !== undefined);
    switch (e.name) {
      case 'collect':
        return vals;
      case 'sum':
        return vals.reduce<number>((a, v) => a + Number(v), 0);
      case 'avg':
        return vals.length ? vals.reduce<number>((a, v) => a + Number(v), 0) / vals.length : null;
      case 'min':
        return vals.length ? vals.reduce((a, v) => ((cmpVals(v, a) ?? 1) < 0 ? v : a)) : null;
      case 'max':
        return vals.length ? vals.reduce((a, v) => ((cmpVals(v, a) ?? -1) > 0 ? v : a)) : null;
    }
  }
  if (!hasAggregate(e)) return evalExpr(e, group[0] ?? {});
  switch (e.k) {
    case 'bin': {
      const sub: Expr = { k: 'bin', op: e.op, l: { k: 'lit', v: evalAggregate(e.l, group) }, r: { k: 'lit', v: evalAggregate(e.r, group) } };
      return evalExpr(sub, {});
    }
    case 'neg':
      return evalExpr({ k: 'neg', e: { k: 'lit', v: evalAggregate(e.e, group) } }, {});
    case 'not':
      return evalExpr({ k: 'not', e: { k: 'lit', v: evalAggregate(e.e, group) } }, {});
    case 'fn':
      return evalExpr({ k: 'fn', name: e.name, args: e.args.map((a) => ({ k: 'lit', v: evalAggregate(a, group) }) as Expr) }, {});
    default:
      throw new MemError('UNSUPPORTED', 'aggregate nested in this expression form');
  }
}

/** Find `var.id = 'literal'` conjuncts usable as seeds. */
export function idSeeds(where: Expr | undefined): Map<string, string> {
  const out = new Map<string, string>();
  const walk = (e: Expr) => {
    if (e.k === 'bin' && e.op === 'AND') {
      walk(e.l);
      walk(e.r);
      return;
    }
    if (e.k === 'bin' && e.op === '=') {
      const sides = [
        [e.l, e.r],
        [e.r, e.l],
      ];
      for (const [a, c] of sides) {
        if (a.k === 'prop' && a.name === 'id' && a.obj.k === 'var' && c.k === 'lit' && typeof c.v === 'string') out.set(a.obj.name, c.v);
        if (a.k === 'fn' && a.name === 'id' && a.args[0]?.k === 'var' && c.k === 'lit' && typeof c.v === 'string') out.set(a.args[0].name, c.v);
      }
    }
  };
  if (where) walk(where);
  return out;
}

export type { NodePat, RelPat, Pattern };
