/**
 * HTTP front for the MCP server, for agents that cannot spawn a stdio process.
 *
 *   GET  /sse               legacy HTTP+SSE transport (events stream; client posts to /messages)
 *   POST /messages?sessionId=...
 *   ALL  /mcp               Streamable HTTP transport (current MCP spec; SSE for server->client)
 *   GET  /health            liveness + live tool count
 *
 * One McpServer instance is created per client session; all sessions share one Engine (same files,
 * same embedder). Binds to 127.0.0.1 by default. Optional bearer token (MEM_HTTP_TOKEN / --token):
 * header `Authorization: Bearer <token>`, or `?token=` on GET /sse for EventSource clients that
 * cannot set headers. DNS-rebinding protection is on when bound to a loopback address.
 */
import http from 'node:http';
import crypto from 'node:crypto';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { buildServer, type Engine } from './server.js';

export interface HttpOptions {
  host?: string;
  port?: number;
  token?: string;
  /** Allowed browser origins for CORS; empty disables CORS headers. */
  cors?: string[];
  log?: (m: string) => void;
}

export interface RunningHttpServer {
  host: string;
  port: number;
  urls: { sse: string; messages: string; mcp: string; health: string };
  sessions: () => number;
  close: () => Promise<void>;
}

function isLoopback(host: string) {
  return host === '127.0.0.1' || host === 'localhost' || host === '::1' || host === '[::1]';
}

export async function startHttpServer(engine: Engine, opts: HttpOptions = {}): Promise<RunningHttpServer> {
  const host = opts.host ?? process.env.MEM_HTTP_HOST ?? '127.0.0.1';
  const requestedPort = opts.port ?? (process.env.MEM_HTTP_PORT ? parseInt(process.env.MEM_HTTP_PORT, 10) : 3939);
  const token = opts.token ?? process.env.MEM_HTTP_TOKEN ?? '';
  const cors = opts.cors ?? (process.env.MEM_HTTP_CORS ? process.env.MEM_HTTP_CORS.split(',').map((s) => s.trim()).filter(Boolean) : []);
  const log = opts.log ?? ((m: string) => console.error(m));

  const sse = new Map<string, { transport: SSEServerTransport; server: McpServer }>();
  const streamable = new Map<string, { transport: StreamableHTTPServerTransport; server: McpServer }>();
  let liveTools = 0;

  const protection = (port: number) =>
    isLoopback(host) ? { enableDnsRebindingProtection: true, allowedHosts: [`127.0.0.1:${port}`, `localhost:${port}`, `[::1]:${port}`, '127.0.0.1', 'localhost'] } : {};

  const authorized = (req: http.IncomingMessage, url: URL): boolean => {
    if (!token) return true;
    const h = req.headers.authorization;
    if (h && h.startsWith('Bearer ') && safeEqual(h.slice(7), token)) return true;
    if (url.pathname === '/sse' && url.searchParams.get('token') && safeEqual(url.searchParams.get('token')!, token)) return true;
    return false;
  };

  const applyCors = (req: http.IncomingMessage, res: http.ServerResponse) => {
    if (!cors.length) return;
    const origin = req.headers.origin;
    if (origin && (cors.includes('*') || cors.includes(origin))) {
      res.setHeader('Access-Control-Allow-Origin', cors.includes('*') ? '*' : origin);
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, Mcp-Session-Id, Last-Event-ID, MCP-Protocol-Version');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS');
      res.setHeader('Access-Control-Expose-Headers', 'Mcp-Session-Id');
    }
  };

  const json = (res: http.ServerResponse, status: number, body: unknown) => {
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(body));
  };

  let boundPort = requestedPort;
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url ?? '/', `http://${req.headers.host ?? `${host}:${boundPort}`}`);
    applyCors(req, res);
    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      res.end();
      return;
    }
    try {
      if (url.pathname === '/health') {
        json(res, 200, { ok: true, transport: ['sse', 'streamable-http'], sessions: sse.size + streamable.size, tools: liveTools, embedder: engine.embedder?.name ?? null, embedderError: engine.embedderError });
        return;
      }
      if (url.pathname === '/' && req.method === 'GET') {
        json(res, 200, { name: 'agent-memory-engine', endpoints: { sse: '/sse', messages: '/messages', mcp: '/mcp', health: '/health' }, auth: token ? 'bearer' : 'none' });
        return;
      }
      if (!authorized(req, url)) {
        json(res, 401, { error: 'UNAUTHORIZED', message: 'missing or invalid bearer token' });
        return;
      }

      // ---- legacy HTTP+SSE
      if (url.pathname === '/sse' && req.method === 'GET') {
        const transport = new SSEServerTransport('/messages', res, protection(boundPort));
        const { server: mcp, live } = buildServer(engine);
        liveTools = live.length;
        const id = transport.sessionId;
        sse.set(id, { transport, server: mcp });
        transport.onclose = () => {
          sse.delete(id);
          mcp.close().catch(() => {});
        };
        await mcp.connect(transport);
        log(`[mem] sse session ${id.slice(0, 8)} opened (${sse.size} sse, ${streamable.size} http)`);
        return;
      }
      if (url.pathname === '/messages' && req.method === 'POST') {
        const id = url.searchParams.get('sessionId') ?? '';
        const s = sse.get(id);
        if (!s) {
          json(res, 404, { error: 'NO_SESSION', message: 'unknown or expired sessionId; reconnect to /sse' });
          return;
        }
        await s.transport.handlePostMessage(req, res);
        return;
      }

      // ---- streamable HTTP
      if (url.pathname === '/mcp') {
        const sid = req.headers['mcp-session-id'];
        const sessionId = Array.isArray(sid) ? sid[0] : sid;
        if (sessionId && streamable.has(sessionId)) {
          await streamable.get(sessionId)!.transport.handleRequest(req, res);
          return;
        }
        if (sessionId && !streamable.has(sessionId)) {
          json(res, 404, { error: 'NO_SESSION', message: 'unknown or expired Mcp-Session-Id; initialize again' });
          return;
        }
        if (req.method !== 'POST') {
          json(res, 400, { error: 'BAD_REQUEST', message: 'open a session with an initialize POST first' });
          return;
        }
        const { server: mcp, live } = buildServer(engine);
        liveTools = live.length;
        const transport = new StreamableHTTPServerTransport({
          sessionIdGenerator: () => crypto.randomUUID(),
          onsessioninitialized: (id) => {
            streamable.set(id, { transport, server: mcp });
            log(`[mem] http session ${id.slice(0, 8)} opened (${sse.size} sse, ${streamable.size} http)`);
          },
          onsessionclosed: (id) => {
            streamable.delete(id);
          },
          ...protection(boundPort),
        });
        transport.onclose = () => {
          if (transport.sessionId) streamable.delete(transport.sessionId);
          mcp.close().catch(() => {});
        };
        await mcp.connect(transport);
        await transport.handleRequest(req, res);
        return;
      }
      json(res, 404, { error: 'NOT_FOUND', message: `no route for ${req.method} ${url.pathname}` });
    } catch (e: any) {
      log(`[mem] http error: ${e?.message ?? e}`);
      if (!res.headersSent) json(res, 500, { error: 'INTERNAL', message: e?.message ?? String(e) });
      else res.end();
    }
  });

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(requestedPort, host, () => resolve());
  });
  const addr = server.address();
  boundPort = typeof addr === 'object' && addr ? addr.port : requestedPort;
  const base = `http://${host.includes(':') ? `[${host}]` : host}:${boundPort}`;
  const running: RunningHttpServer = {
    host,
    port: boundPort,
    urls: { sse: `${base}/sse`, messages: `${base}/messages`, mcp: `${base}/mcp`, health: `${base}/health` },
    sessions: () => sse.size + streamable.size,
    close: async () => {
      for (const s of sse.values()) await s.transport.close().catch(() => {});
      for (const s of streamable.values()) await s.transport.close().catch(() => {});
      await new Promise<void>((r) => server.close(() => r()));
    },
  };
  log(`[mem] HTTP MCP listening on ${base}  (SSE: ${running.urls.sse}  Streamable HTTP: ${running.urls.mcp}  auth: ${token ? 'bearer token' : 'none'}${isLoopback(host) ? '' : '  WARNING: bound to a non-loopback address'})`);
  return running;
}

function safeEqual(a: string, b: string) {
  const ab = Buffer.from(a), bb = Buffer.from(b);
  return ab.length === bb.length && crypto.timingSafeEqual(ab, bb);
}
