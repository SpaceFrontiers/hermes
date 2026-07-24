# Hermes Server

A high-performance gRPC search server for Hermes indexes.

## Features

- **Index Management**: Create, delete, and manage search indexes
- **Document Indexing**: Stream or batch index documents
- **Full-Text Search**: Term queries, boolean queries, and boosting
- **Document Retrieval**: Get documents by ID
- **Segment Management**: Commit changes and force merge segments

## Installation

```bash
cargo install hermes-server
```

Or build from source:

```bash
cargo build --release -p hermes-server
```

## Usage

### Starting the Server

```bash
hermes-server --addr 0.0.0.0:50051 --data-dir ./data
```

Options:

- `-a, --addr`: Address to bind to (default: `0.0.0.0:50051`)
- `-d, --data-dir`: Directory for storing indexes (default: `./data`)

### Search resource controls

`--search-threads` sets the width of a bounded Rayon pool shared by CPU-bound
search work across every open index. When omitted, it defaults to one thread per
four detected CPUs (minimum one). Nested parallel work, including fused queries,
segment fan-out, phrase loading, and vector search, stays inside this same pool;
Hermes does not create a pool per index or request.

`--max-concurrent-searches` bounds expensive search pipelines across all HTTP/2
connections; document lookup and metadata RPCs do not consume these permits.
When omitted, Hermes allows one concurrent search per eight detected CPUs,
clamped to `1..=8` (six searches on a 48-core host). Requests above that
capacity fail promptly with gRPC `RESOURCE_EXHAUSTED`; clients should retry with
bounded exponential backoff. This keeps overload from accumulating an
unbounded in-process request queue. Completed or cancelled searches release
their permit automatically.

The server also rejects request sizes that could otherwise multiply into large
per-segment heaps:

| Request component                           |            Limit |
| ------------------------------------------- | ---------------: |
| Final search results                        |          `10000` |
| Pagination window (`offset + limit`)        |          `50000` |
| L1 reranker candidates                      |          `50000` |
| Fusion candidates fetched per sub-query     |          `50000` |
| Fusion sub-queries                          |             `16` |
| Fusion fetch depth x number of sub-queries  |         `200000` |
| Query nesting depth                         |             `32` |
| Query nodes / aggregate clauses             |    `256` / `512` |
| Clauses in one Boolean query                |            `128` |
| Aggregate query text                        |         `64 KiB` |
| Aggregate query vector payload              |          `1 MiB` |
| Dense dimensions / sparse input dimensions  | `65536` / `4096` |
| Binary query bytes                          |        `256 KiB` |
| Stored fields requested                     |             `64` |
| Aggregate requested-field name bytes        |         `16 KiB` |
| Retained response / encoded response (each) |         `48 MiB` |

Zero-valued defaults remain supported. Derived reranker and fusion defaults are
checked and capped at the corresponding limit; explicit values over a limit
return gRPC `INVALID_ARGUMENT`.

The structural limits are checked iteratively before query conversion and
before a search permit is acquired. Requested stored fields are resolved and
deduplicated once, and response hydration is charged before field values are
cloned into protobuf objects. A response that would exceed its memory/encoding
budget fails with `RESOURCE_EXHAUSTED`; request fewer hits or fields.

### Background merge and reorder

The server uses one application-wide automatic-merge gate, one BP CPU pool,
and one whole-pass BP gate across all indexes. These are deliberately separate
controls:

| Option                                   |   Default | Meaning                                                                                                                                           |
| ---------------------------------------- | --------: | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--max-concurrent-merges`                |       `4` | Automatic merge tasks admitted per index and across the process.                                                                                  |
| `--segments-per-tier`                    |      `10` | Segment budget per size tier. Automatic compaction begins only above the computed tier budget.                                                    |
| `--max-merge-at-once`                    |      `24` | Maximum inputs in one automatic merge; wider fan-in drains flush-segment bursts in fewer passes.                                                  |
| `--max-merged-docs`                      | `5000000` | Maximum documents written by one automatic merge.                                                                                                 |
| `--max-segment-docs`                     | `5000000` | Absolute output-size cap for automatic and explicit force merges.                                                                                 |
| `--optimizer-threads`                    |       `0` | Threads in the shared BP pool. `0` disables periodic optimizer scans; merge-time and manual BP still use the process-wide fallback pool.          |
| `--optimizer-concurrent-passes`          |       `2` | Simultaneous whole-segment BP passes across optimizer, merge-time, and manual reorder. Clamped to `1..=2`; automatic merges use at most one slot. |
| `--optimizer-scan-interval-secs`         |      `60` | Interval between background scans.                                                                                                                |
| `--optimizer-large-segment-docs`         | `5000000` | Document threshold for partial/budgeted first passes.                                                                                             |
| `--optimizer-time-budget-secs`           |     `600` | Wall-clock budget for an optimizer pass on a large segment.                                                                                       |
| `--optimizer-partial-min-partition-docs` |     `256` | Initial depth floor for large segments (one default LSP superblock).                                                                              |
| `--optimizer-unconverged-cooldown-secs`  |     `600` | Delay after a rewrite finishes before another deepening pass.                                                                                     |
| `--optimizer-max-unconverged-passes`     |       `3` | Optimizer follow-up limit per truncated lineage, including the initial partial pass. `0` disables follow-up deepening.                            |
| `--merge-bp-budget-secs`                 |     `600` | Wall-clock budget for BP performed inside a merge; `0` explicitly selects an unbudgeted pass.                                                     |
| `--bp-memory-budget-mb`                  |   `24576` | Per-pass algorithmic working-set bound; not a reservation or a total-process RSS limit.                                                           |

An active BP pass is CPU-bound and is expected to occupy up to
`--optimizer-threads` cores. Concurrent passes share that same pool, so a
second pass primarily raises simultaneous working sets and outstanding IO; it
does not create another pool per pass or per index. Explicit force merges pause
new background BP admission and reserve foreground capacity after existing
merges drain. With two-pass admission, automatic merge BP uses at most one
slot; the other remains available to retire short fresh-segment optimizer
passes instead of queueing them behind multi-minute merges. For predictable
service latency, start with one pass and a CPU width that leaves capacity for
query and indexing work.

BP itself is level-synchronized. Same-depth partitions dynamically share the
memory-bounded degree lanes printed at pass start. Early levels have too few
partitions to fill a pool, so coarse gain, degree, radix-selection, and movement
work parallelizes within those partitions; later levels distribute independent
partitions across lanes. A progress line such as
`field=sparse_vectors entity_kind=records` is sparse BP—not ANN training—even
when the containing force merge also carries ANN fields.

In BP progress logs:

- `active` is the number of partitions currently being bisected, not a CPU
  count.
- `depth=x/y` is the deepest started level versus the expected level count.
- `entity_passes` is the cumulative number of entities visited by refinement
  iterations; it can be billions even for a much smaller corpus.
- `degree_lanes` in the start line is the maximum concurrent vocabulary
  workspace count allowed by the BP memory budget. A low value can constrain
  utilization even when CPU is free.
- `stop_reason=objective`, `time_budget`, `memory_budget`, `shutdown`, or
  `complete` distinguishes convergence from a bounded early stop.

Low CPU outside `[reorder][bp]` lines can be expected during source reads,
kernel-assisted block copies, output serialization, fsync, reader reload, or
primary-key snapshot refresh. During BP, sustained low utilization with
`degree_lanes=1` indicates a memory-derived concurrency limit; low `active`
alone does not, because coarse partitions can use inner parallelism.

The dominant graph representation is roughly `4 bytes/posting + 32
bytes/document`. The limit also accounts for record maps, vocabulary-sized
degree arrays, and record-rewrite grid/encode windows. If the record
representation cannot fit, Hermes performs a valid blockwise rewrite and marks
it unconverged; if only the graph is too large, it retains a bounded set of
low-frequency dimensions. Stored postings are never truncated. At the process
level, still budget for up to `concurrent-passes * bp-memory-budget`, plus
indexing builders, merge state, mmap/page-cache residency, output buffering,
and open readers.

Automatic compaction is deliberately more aggressive under ingestion bursts.
The normal large-scale policy uses budget-aware, balanced selection and up to
24 inputs per merge. If eligible live segments exceed twice the computed tier
budget, merge-time BP is deferred: automatic merges block-copy their BMP
payloads, mark the output unreordered, and let the optimizer perform one BP
pass after the compaction wave. This prevents every merge slot from waiting on
long sparse BP while hundreds of fresh segments accumulate.

Explicit force merge is a foreground compaction plan, not “always one output”:

- Segments are packed into maximal output groups under
  `--max-segment-docs`; the response can therefore report multiple segments.
- A group larger than 64 inputs is reduced with a shallow, balanced 64-way
  hierarchy. Intermediate levels are streaming block-copy merges; only the
  final level pays BP.
- Search and primary-key snapshots refresh after every durable replacement so
  retired source files can be deleted during a long merge rather than all at
  the end.
- The target index's writer lock is held for the RPC, so indexing that same
  index waits. Other indexes use different writer locks and remain admissible,
  but may run more slowly because merge permits, the BP pool, CPU, and storage
  are process-wide resources.

Merge failures use exponential retry backoff (30 seconds through 30 minutes).
A deterministic missing/corrupt source is quarantined for the process lifetime
so the same candidate cannot consume all cores in an immediate loop. The
metadata entry remains visible—Hermes never silently removes documents. To
explicitly remove corrupt entries and their files, stop normal traffic and run:

```bash
hermes-server --data-dir ./data --doctor
```

`--doctor` is destructive recovery: it validates every metadata-live segment
and removes entries that cannot be opened. Normal startup/writer-open cleanup
only deletes true unowned files and cannot delete metadata-live, actively
written, or reader-retained segments. Standalone reorder failures are also
backed off per source from pass completion, so a pass that outlasts the scan
interval cannot restart continuously. Successful but budget-truncated outputs
carry a retry count across replacement IDs and stop being optimizer candidates
at `--optimizer-max-unconverged-passes`, so a lineage that never converges
cannot consume the optimizer pool forever. Explicitly configured merge-time BP
still runs when that lineage later participates in a real merge. Details and
invariants are in [Segment lifecycle and recovery](../docs/segment-lifecycle.md).

### Graceful shutdown

SIGTERM and Ctrl-C initiate an ordered drain:

1. Reject new registry/lifecycle work and notify the optimizer.
2. Stop gRPC admission and drain requests already accepted by tonic.
3. Stop optimizer scheduling; running BP observes shutdown between bounded
   refinement/copy stages and exits without publishing a partial replacement.
4. Signal and join every index's native indexing workers, waiting first for any
   cancellation-safe commit finalizer.
5. Drain merge/reorder tasks, metadata transactions, and tracked deletions.

`[shutdown] gRPC server drained; waiting for background work` means the
transport is down but lifecycle work is still being joined. Per-index
`[shutdown] index '…' drained` lines confirm individual completion. Only
`Hermes server shut down gracefully` is the final success marker. Shutdown can
therefore take longer than an orchestrator's default grace period when an
already-running filesystem or CPU stage needs time to reach a cancellation
check; size the grace period accordingly and avoid SIGKILL when durability is
required.

## gRPC API

The server exposes two services: `SearchService` and `IndexService`.

### IndexService

#### CreateIndex

Create a new index with a schema definition (SDL or JSON format).

```protobuf
rpc CreateIndex(CreateIndexRequest) returns (CreateIndexResponse);

message CreateIndexRequest {
  string index_name = 1;
  string schema = 2;  // SDL or JSON schema
}
```

**SDL Schema Example:**

```sdl
index articles {
    field title: text [indexed, stored]
    field body: text [indexed, stored]
    field author: text [indexed, stored]
    field published_at: i64 [indexed, stored]
    field tags: text [indexed, stored<multi>]
}
```

**JSON Schema Example:**

```json
{
  "fields": [
    { "name": "title", "type": "text", "indexed": true, "stored": true },
    { "name": "body", "type": "text", "indexed": true, "stored": true },
    { "name": "score", "type": "f64", "indexed": false, "stored": true }
  ]
}
```

#### BatchIndexDocuments

Index multiple documents in a single request.

```protobuf
rpc BatchIndexDocuments(BatchIndexDocumentsRequest) returns (BatchIndexDocumentsResponse);

message BatchIndexDocumentsRequest {
  string index_name = 1;
  repeated NamedDocument documents = 2;
}

message FieldEntry {
  string name = 1;
  FieldValue value = 2;
}

message NamedDocument {
  repeated FieldEntry fields = 1; // repeated names preserve multi-value fields
}
```

#### IndexDocuments (Streaming)

Stream documents for indexing.

```protobuf
rpc IndexDocuments(stream IndexDocumentRequest) returns (IndexDocumentsResponse);
```

#### Commit

Commit pending changes to make them searchable.

```protobuf
rpc Commit(CommitRequest) returns (CommitResponse);
```

#### ForceMerge

Compact the index into a deterministic near-minimal set of maximal outputs
allowed by the configured `--max-segment-docs` cap. Small indexes normally
become one segment; larger indexes intentionally remain multi-segment. The RPC
returns only after the foreground plan completes.

```protobuf
rpc ForceMerge(ForceMergeRequest) returns (ForceMergeResponse);
```

#### Reorder

Run standalone BP reorder for the index's `reorder`-attributed BMP fields. The
RPC returns after all eligible segment replacements complete.

```protobuf
rpc Reorder(ReorderRequest) returns (ReorderResponse);
```

#### RetrainVectorIndex

Train new global IVF-PQ or binary-IVF codebooks from the current corpus and
rebuild every ANN segment. The complete segment/codebook generation is
published atomically; existing readers keep the previous generation.

Training reads a deterministic corpus-wide sample for one field at a time.
`--vector-training-max-samples` (default 10,000,000) and
`--vector-training-memory-mb` (default 4096) are simultaneous per-field
bounds; the smaller limit wins. Unselected corpus vectors are never loaded
into the training sample.

```protobuf
rpc RetrainVectorIndex(RetrainVectorIndexRequest) returns (RetrainVectorIndexResponse);
```

#### DeleteIndex

Delete an index and all its data.

```protobuf
rpc DeleteIndex(DeleteIndexRequest) returns (DeleteIndexResponse);
```

#### ListIndexes

```protobuf
rpc ListIndexes(ListIndexesRequest) returns (ListIndexesResponse);
```

### SearchService

#### Search

Search for documents matching a query.

```protobuf
rpc Search(SearchRequest) returns (SearchResponse);

message SearchRequest {
  string index_name = 1;
  Query query = 2;
  uint32 limit = 3;
  uint32 offset = 4;
  repeated string fields_to_load = 5;
  Reranker reranker = 6;
  uint32 candidate_limit = 7;
}
```

**Query Types:**

- **TermQuery**: match one already-tokenized term
- **MatchQuery**: tokenize natural-language text server-side
- **BooleanQuery**: combine queries with must/should/must_not
- **BoostQuery**: multiply a nested query's score
- **AllQuery**: match every document
- **RangeQuery**: inclusive range over a fast numeric field
- **PrefixQuery**: filter-style term-prefix expansion
- **SparseVectorQuery**: learned sparse retrieval from text or explicit weights
- **DenseVectorQuery**: dense TQ/IVF-TQ retrieval
- **BinaryDenseVectorQuery**: packed-bit Hamming retrieval
- **FusionQuery**: top-level weighted union using RRF or normalized weighted sum

#### GetDocument

Retrieve a document by the segment-local address returned in `SearchHit`.

```protobuf
rpc GetDocument(GetDocumentRequest) returns (GetDocumentResponse);

message GetDocumentRequest {
  string index_name = 1;
  DocAddress address = 2; // segment_id + segment-local doc_id
}
```

#### GetIndexInfo

Get information about an index (document count, segments, schema).

```protobuf
rpc GetIndexInfo(GetIndexInfoRequest) returns (GetIndexInfoResponse);
```

## Field Types

| Type                  | Description                          |
| --------------------- | ------------------------------------ |
| `text`                | Full-text searchable string          |
| `u64`                 | Unsigned 64-bit integer              |
| `i64`                 | Signed 64-bit integer                |
| `f64`                 | 64-bit floating point                |
| `bytes`               | Binary data                          |
| `json`                | JSON object (stored as string)       |
| `sparse_vector`       | Sparse vector for semantic search    |
| `dense_vector`        | Dense vector for semantic search     |
| `binary_dense_vector` | Packed-bit vector for Hamming search |

## Example: Python Client

```python
from hermes_client_python import HermesClient

async with HermesClient("localhost:50051") as client:
    await client.create_index(
        "articles",
        """
        index articles {
            field title: text [indexed, stored]
            field body: text [indexed, stored]
        }
        """,
    )
    await client.index_documents(
        "articles",
        [{"title": "Hello World", "body": "This is my first article"}],
    )
    await client.commit("articles")
    response = await client.search(
        "articles",
        query={"match": {"field": "title", "text": "hello"}},
        fields_to_load=["title", "body"],
    )
    for hit in response.hits:
        print(hit.address, hit.score, hit.fields)
```

## Docker

Build and run with Docker:

```bash
docker build -t hermes-server -f hermes-server/Dockerfile .
docker run -p 50051:50051 -v ./data:/data hermes-server --data-dir /data
```

Or pull from GitHub Container Registry:

```bash
docker pull ghcr.io/spacefrontiers/hermes/hermes-server:latest
docker run -p 50051:50051 -v ./data:/data ghcr.io/spacefrontiers/hermes/hermes-server:latest --data-dir /data
```

## License

MIT
