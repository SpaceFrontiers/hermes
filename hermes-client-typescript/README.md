# hermes-client-typescript

TypeScript client for [Hermes](https://github.com/SpaceFrontiers/hermes) search server.

## Installation

```bash
pnpm add hermes-client-typescript
```

## Usage

```typescript
import { HermesClient } from "hermes-client-typescript";

const client = new HermesClient("localhost:50051");
client.connect();

// Create index
await client.createIndex(
  "articles",
  `
  index articles {
    field title: text [indexed, stored]
    field body: text [indexed, stored]
  }
`,
);

// Index documents
await client.indexDocuments("articles", [
  { title: "Hello", body: "World" },
  { title: "Foo", body: "Bar" },
]);
await client.commit("articles");

// Search
const results = await client.search("articles", {
  query: { match: { field: "title", text: "hello" } },
  fieldsToLoad: ["title", "body"],
});
for (const hit of results.hits) {
  console.log(hit.address, hit.score, hit.fields);
}

// Clean up
client.close();
```

## API

### Index Management

- `createIndex(indexName, schema)` — Create a new index with SDL or JSON schema
- `deleteIndex(indexName)` — Delete an index
- `listIndexes()` — List all index names
- `getIndexInfo(indexName)` — Get index metadata (doc count, segments, schema)

### Document Indexing

- `indexDocuments(indexName, docs)` — Batch index documents, returns `[indexedCount, errorCount, errors]`
- `indexDocument(indexName, doc)` — Index a single document
- `indexDocumentsStream(indexName, asyncIterable)` — Stream documents for indexing
- `commit(indexName)` — Commit pending changes, returns total doc count
- `forceMerge(indexName)` — Foreground compaction, returns the resulting segment count
- `reorder(indexName)` — BP-reorder eligible BMP fields, returns the segment count

### Search

- `search(indexName, request)` — Search with the proto-shaped query union
- `getDocument(indexName, address)` — Load a document using `SearchHit.address`

### Search Options

```typescript
await client.search("docs", {
  // Exact term query
  query: { term: { field: "field", term: "value" } },

  limit: 10,
  offset: 0,
  fieldsToLoad: ["title", "body"],
});

await client.search("docs", {
  // Boolean query with server-side text tokenization
  query: {
    boolean: {
      must: [{ match: { field: "title", text: "hello" } }],
      should: [{ match: { field: "body", text: "world" } }],
    },
  },
});

await client.search("docs", {
  // Sparse text; use indices + values instead of text for a precomputed query
  query: {
    sparseVector: {
      field: "embedding",
      text: "what is machine learning?",
      pruning: 0.5,
    },
  },
});

await client.search("docs", {
  // Dense retrieval plus exact L2 reranking
  query: {
    denseVector: {
      field: "embedding",
      vector: [0.1, 0.2, 0.3],
      nprobe: 10,
    },
  },
  reranker: { field: "embedding", vector: [0.1, 0.2, 0.3] },
  candidateLimit: 20,
});
```

## Development

```bash
pnpm install
pnpm run generate  # regenerate proto stubs
pnpm run build     # compile TypeScript
```

## Timeouts / deadlines

Every RPC accepts an optional trailing `timeoutMs` which sets a real gRPC
deadline (propagated to the server via `grpc-timeout`); a client-wide default
can be set in the constructor. On expiry the call rejects with
`DEADLINE_EXCEEDED`.

```ts
const client = new HermesClient("localhost:50051", { defaultTimeoutMs: 5000 });
await client.search("articles", { query: { ... } }, 500); // per-call override
await client.forceMerge("articles", 3_600_000); // long op, long deadline
```
