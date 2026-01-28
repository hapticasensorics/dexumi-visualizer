# Web Viewer Collaboration and Cloud Architecture Guide

This guide covers how to enable remote collaboration for dataset visualization with Rerun's Web Viewer and a Reflex-based web UI (HapticaGUI), plus an architecture proposal for a cloud visualization platform.

## 1) Rerun Web Viewer architecture and limitations

- The Rerun Viewer (native and web) includes an in-memory Chunk Store and a gRPC endpoint that accepts streamed data from the SDK. The Logging SDK can stream data directly to the Viewer or write `.rrd` files for later viewing. The Web Viewer can also be embedded in web apps or notebooks. The CLI bundles the Web Viewer and the OSS Data Platform server. The hosted Web Viewer is available at `rerun.io/viewer`. 
- The Web Viewer runs as 32-bit Wasm and is limited to roughly 2 GiB of memory in practice. It is single-threaded, so it is slower than the native viewer for heavy workloads.
- The Data Platform provides persistent storage and indexing for large-scale data, organizing datasets into named collections with segments that map to individual `.rrd` files. It is available as an OSS server (`rerun server`) and as a managed offering.

## 2) Streaming large recordings efficiently over HTTP

### 2.1 Static recordings (.rrd)

- The Rerun CLI accepts HTTP(S) URLs to `.rrd` or `.rbl` files, which makes it practical to host recordings on object storage or a CDN.
- Embedding `app.rerun.io` uses `RERUN_VERSION` and `RRD_URL`. The `RRD_URL` can be an HTTP file or a live connection using the `rerun+http` scheme (serve API). The JS Web Viewer API can open and close recordings; it also accepts an array of URLs to open multiple recordings.
- Version compatibility matters: the web-viewer package version corresponds to the supported SDK version, and `.rrd` compatibility is only partially stable across versions. For example, a web viewer version can load the previous minor version, but not arbitrarily older files.

### 2.2 Live streaming (serve API)

- `serve_grpc` starts a gRPC server that buffers log data in memory so late-connecting viewers can replay data. You can set a `server_memory_limit`; once reached, the earliest data is dropped, while static data is never dropped.
- `serve_web` is deprecated. The recommended pattern is `serve_grpc` plus `serve_web_viewer`. `serve_web` is equivalent to calling those two functions.
- The gRPC server is reachable via `rerun+http://{host}:{port}/proxy`, which the Web Viewer can connect to.

### 2.3 HTTP transport optimizations (recommended patterns)

- Support HTTP Range requests for large `.rrd` files. Range requests use the `Range` header, `206 Partial Content`, and `Content-Range`, and allow partial downloads and resume. Many object stores (including S3) support retrieving object parts by Range.
- For very large datasets, split recordings into smaller segments. The Data Platform already models datasets as multiple `.rrd` segments, and the Web Viewer can open multiple recordings (array of URLs). This provides a practical way to load data incrementally and keep total in-memory size under the Web Viewer's 2 GiB limit.

## 3) Multi-user viewing sessions

- The Web Viewer JS API supports callbacks (for example `selection_change`) and can open or close recordings. These hooks can be used to capture user interactions for collaboration.
- Reflex uses WebSockets for event delivery and state updates, and each user gets an isolated state instance by default.
- Reflex `SharedState` lets multiple clients share state when they link to the same token; updates propagate to all linked clients. Best practices include keeping shared state minimal, using backend-only vars for sensitive data, and using `client_token` for identity rather than trusting client-supplied parameters.
- Use WebSockets for low-latency bidirectional collaboration (presence, cursor, time slider, selection). Use Server-Sent Events (EventSource) when you only need server -> client updates.

## 4) Cloud deployment patterns

### AWS pattern

- Compute: ECS (fully managed container orchestration) with Fargate (serverless compute for containers) or EKS (managed Kubernetes control plane that can use EC2 or Fargate).
- Storage: S3 for `.rrd` files; issue presigned URLs for time-limited access.
- CDN and access control: CloudFront signed URLs or signed cookies to control access to private content at the edge.

### GCP pattern

- Compute: Cloud Run (fully managed platform for running containers) with HTTPS endpoints and support for WebSockets and gRPC.
- Storage: Cloud Storage with signed URLs for time-limited access.
- CDN and access control: Cloud CDN signed URLs or signed cookies for private content.

### Fly.io pattern

- Compute: Fly.io runs apps in regions worldwide; users connect through a global Anycast network to the nearest region.
- Multi-region: Fly Proxy routes to the closest healthy region. Volumes do not replicate across regions, so multi-region setups should use a primary write region and read replicas where applicable.
- Storage: Fly Volumes are local to a single server in a single region and do not replicate automatically; apps must handle replication if needed.

## 5) Integration with HapticaGUI (Reflex-based web UI)

**Embedding choices**
- Iframe (quickest): Embed `app.rerun.io` with `RERUN_VERSION` and `RRD_URL`. This is simple but has no programmable control.
- JS package (recommended for collaboration): Use `@rerun-io/web-viewer` in a custom Reflex component to embed the viewer, then use the Viewer API (`open`, `close`, `on('selection_change', ...)`) to sync state.

**Self-hosting Reflex**
- Set `api_url` to the external host so the frontend can reach the backend state service.
- Production mode serves the frontend on port 3000 and backend on port 8000. Reverse proxies should be configured to pass WebSocket traffic for Reflex interactivity.

**Integration flow**
1. HapticaGUI lists datasets and sessions (from metadata service or Data Platform catalog).
2. When a dataset is selected, the backend issues a signed URL (or session-bound `rerun+http` URL for live streams).
3. The Web Viewer opens the recording and emits selection/viewport events.
4. Reflex `SharedState` records shared view state (time cursor, selection, layout) and broadcasts it to collaborators.
5. Other clients apply state updates to their Web Viewer instance via the Viewer API.

## 6) Authentication and access control

- Use OAuth 2.0 to obtain scoped, time-limited access tokens from an authorization server. Resource servers verify tokens and enforce scopes.
- Enforce access control server-side, deny by default, validate permissions on every request, and log failures. Keep tokens short-lived.
- Protect recordings with signed URLs in object storage (S3, Cloud Storage). These are time-limited and can be issued per dataset or per session.
- If using a CDN, use CloudFront or Cloud CDN signed URLs/cookies to restrict access at the edge.
- Never trust client-supplied identifiers for authorization. Use server-side identity (for Reflex, `client_token`) and server-side ACL checks.

## Proposed architecture for Haptica's cloud visualization platform

Below is a reference architecture that combines Rerun, Reflex, and collaboration features into a scalable cloud platform. This is a proposed design, built from Rerun and Reflex capabilities and standard cloud patterns.

```
+--------------------+       +---------------------+      +----------------------+
| Data Producers     |       | Live Stream Service |      | Object Storage       |
| (SDK + logging)    |-----> | (serve_grpc)        |----> | (S3/GCS + CDN)       |
+--------------------+       +---------------------+      +----------------------+
          |                               |                         |
          | save .rrd segments            | rerun+http               | signed URLs
          v                               v                         v
+---------------------------+     +---------------------+    +----------------------+
| Metadata / Catalog        |     | Web Viewer Service  |    | HapticaGUI (Reflex)  |
| (Rerun Data Platform or   |<--->| (embed JS viewer)   |<-->| UI + Collaboration   |
| custom DB)                |     +---------------------+    +----------------------+
          ^                               ^                         |
          |                               |                         |
          |                               | WebSocket/SSE            |
          |                               v                         v
+---------------------------+     +---------------------+    +----------------------+
| AuthN/AuthZ Service       |<--->| Collaboration svc   |<-->| SharedState tokens   |
| (OAuth2, ACLs, audit)     |     | (session state)     |    | and viewer state     |
+---------------------------+     +---------------------+    +----------------------+
```

### Data plane
- **Offline datasets:** Log to `.rrd` segments and store in object storage. Index segments in the Rerun Data Platform (datasets + segments) or a custom catalog. The UI asks the backend for a dataset, then gets signed URLs and opens them in the Web Viewer.
- **Live streams:** Use `serve_grpc` per live session. The Web Viewer connects via `rerun+http://.../proxy`. Set `server_memory_limit` based on expected session size and retention.

### Control and collaboration plane
- **Session service:** Creates collaboration rooms, issues tokens, and stores shared view state.
- **Reflex SharedState:** Links to the session token and broadcasts state changes (time cursor, selection, camera presets) to all clients.
- **Viewer sync:** Use the Web Viewer JS API callbacks to capture selection and navigation and apply updates to local viewer instances.

### Compatibility and scaling notes
- **Viewer versioning:** Match `RERUN_VERSION` (or the JS package version) to the SDK version that generated each `.rrd` file. Do not assume broad cross-version compatibility.
- **Memory limits:** Keep total concurrently loaded data under the Web Viewer's ~2 GiB limit. Use segmentation and progressive loading to stay within bounds.
- **Edge delivery:** For static datasets, push `.rrd` files through a CDN with signed URLs or signed cookies to reduce latency and protect access.

## References

Rerun
- https://rerun.io/docs/concepts/app-model
- https://rerun.io/docs/howto/integrations/embed-web
- https://ref.rerun.io/docs/js/0.24.0/web-viewer/
- https://rerun.io/docs/reference/cli
- https://ref.rerun.io/docs/python/0.26.1/common/initialization_functions/

Reflex
- https://reflex.dev/docs/advanced-onboarding/how-reflex-works/
- https://reflex.dev/docs/state/overview
- https://reflex.dev/docs/state-structure/shared-state
- https://reflex.dev/docs/hosting/self-hosting/

Web standards
- https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Range
- https://developer.mozilla.org/docs/Web/HTTP/Reference/Headers/Content-Range
- https://developer.mozilla.org/id/docs/Web/HTTP/Range_requests
- https://developer.mozilla.org/docs/Web/API/WebSocket
- https://developer.mozilla.org/en-US/docs/Web/API/EventSource

AWS
- https://aws.amazon.com/ecs/
- https://aws.amazon.com/fargate/features/
- https://aws.amazon.com/eks/features/
- https://docs.aws.amazon.com/AmazonS3/latest/dev/ShareObjectPreSignedURL.html
- https://docs.aws.amazon.com/AmazonS3/latest/API/API_GetObject.html
- https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-overview.html

GCP
- https://docs.cloud.google.com/run/docs/overview/what-is-cloud-run
- https://cloud.google.com/run/docs/triggering/websockets
- https://cloud.google.com/storage/docs/access-control/signed-urls
- https://cloud.google.com/cdn/docs/using-signed-urls
- https://cloud.google.com/cdn/docs/authenticate-content

Fly.io
- https://fly.io/docs/reference/regions/
- https://fly.io/docs/volumes/overview/
- https://fly.io/docs/blueprints/resilient-apps-multiple-machines/

Security
- https://datatracker.ietf.org/doc/html/rfc6749
- https://owasp.org/Top10/2025/A01_2025-Broken_Access_Control/
