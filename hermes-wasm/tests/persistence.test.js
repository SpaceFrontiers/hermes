import { test, expect } from "vitest";

import init, { LocalIndex } from "../pkg/hermes_wasm";

export class InMemoryFS {
	constructor() {
		this.storage = {};
	}

	write = async (id, buffer) => {
		this.storage[id] = new Uint8Array(buffer).buffer;
	}

	get = async (id) => {
		return this.storage[id];
	}
	delete = async (ids) => {
		paths.forEach((id) => {
			delete this.storage[id];
		});
	}
	list = async () => {
		return Object.keys(this.storage);
	}
}

const sharedStorage = new InMemoryFS();

test("Fill the index with the data", async () => {
	await init();

	// Define schema using SDL
	const index = await LocalIndex.withStorage(sharedStorage, `
		index articles {
			field title: text<en_stem> [indexed, stored]
			field body:  text<en_stem> [indexed, stored]
			field views: u64 [indexed, stored]
		}
	`);

	// Add documents
	await index.addDocuments([
		{
			title: "Rust Programming",
			body: "Rust is a systems language.",
			views: 1500,
		},
		{
			title: "Search Engines",
			body: "BM25 is a ranking function.",
			views: 800,
		},
	]);

	// Commit (builds the segment)
	await index.commit();

	// Search
	const results = await index.search("rust", 10);
	// { hits: [{ address: { segment_id, doc_id }, score }], total_hits: 1 }

	// Get document
	const doc = await index.getDocument(
		results.hits[0].address.segment_id,
		results.hits[0].address.doc_id,
	);

	expect(doc).toEqual({
		title: "Rust Programming",
		body: "Rust is a systems language.",
		views: 1500,
	});
});

test("Each instance of Index have its own state", async () => {
	await init();

	// Define schema using SDL
	const index = await LocalIndex.withStorage(new InMemoryFS(), `
		index articles {
			field title: text<en_stem> [indexed, stored]
			field body:  text<en_stem> [indexed, stored]
			field views: u64 [indexed, stored]
		}
	`);

	// No data in this instance
	await expect(index.search("rust", 10)).rejects.toThrow("No committed data");
});

test("Load data in new instance", async () => {
	await init();

	// Define schema using SDL
	const index = await LocalIndex.withStorage(sharedStorage, `
		index articles {
			field title: text<en_stem> [indexed, stored]
			field body:  text<en_stem> [indexed, stored]
			field views: u64 [indexed, stored]
		}
	`);

	// Search
	const results = await index.search("rust", 10);
	// { hits: [{ address: { segment_id, doc_id }, score }], total_hits: 1 }

	// Get document
	const doc = await index.getDocument(
		results.hits[0].address.segment_id,
		results.hits[0].address.doc_id,
	);

	expect(doc).toEqual({
		title: "Rust Programming",
		body: "Rust is a systems language.",
		views: 1500,
	});
});