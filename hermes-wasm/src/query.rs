//! Structured query builder: converts JS query objects → core `Box<dyn Query>`.
//!
//! ```js
//! // Term query (exact token match after tokenization)
//! { term: { field: "title", value: "rust" } }
//!
//! // Match query (tokenized, multi-token OR)
//! { match: { field: "body", text: "search engine" } }
//!
//! // Boolean query
//! { boolean: { must: [...], should: [...], mustNot: [...] } }
//!
//! // Prefix query
//! { prefix: { field: "title", value: "rus" } }
//!
//! // Sparse vector query
//! { sparseVector: { field: "emb", indices: [1, 5], values: [0.5, 0.3] } }
//!
//! // Dense vector query
//! { denseVector: { field: "emb", vector: [0.1, 0.2, 0.3] } }
//! ```

use hermes_core::Schema;
use hermes_core::query::{
    BooleanQuery, DenseVectorQuery, PrefixQuery, Query, SparseVectorQuery, TermQuery,
};
use hermes_core::tokenizer::TokenizerRegistry;
use serde::Deserialize;
use std::sync::Arc;
use wasm_bindgen::JsValue;

/// Top-level query object deserialized from JS.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JsQuery {
    #[serde(default)]
    term: Option<JsTermQuery>,
    #[serde(default, rename = "match")]
    match_: Option<JsMatchQuery>,
    #[serde(default)]
    boolean: Option<JsBooleanQuery>,
    #[serde(default)]
    prefix: Option<JsPrefixQuery>,
    #[serde(default)]
    sparse_vector: Option<JsSparseVectorQuery>,
    #[serde(default)]
    dense_vector: Option<JsDenseVectorQuery>,
}

#[derive(Deserialize)]
pub(crate) struct JsTermQuery {
    field: String,
    value: String,
}

#[derive(Deserialize)]
pub(crate) struct JsMatchQuery {
    field: String,
    text: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JsBooleanQuery {
    #[serde(default)]
    must: Vec<JsQuery>,
    #[serde(default)]
    should: Vec<JsQuery>,
    #[serde(default)]
    must_not: Vec<JsQuery>,
}

#[derive(Deserialize)]
pub(crate) struct JsPrefixQuery {
    field: String,
    value: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JsSparseVectorQuery {
    field: String,
    indices: Vec<u32>,
    values: Vec<f32>,
    #[serde(default)]
    heap_factor: Option<f32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JsDenseVectorQuery {
    field: String,
    vector: Vec<f32>,
    #[serde(default)]
    nprobe: Option<usize>,
    #[serde(default)]
    rerank_factor: Option<f32>,
}

/// Search request with optional parameters.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JsSearchRequest {
    pub query: JsQuery,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub fields_to_load: Option<Vec<String>>,
}

fn default_limit() -> usize {
    10
}

/// Convert a JS query object into a core `Box<dyn Query>`.
pub(crate) fn convert_query(
    js: &JsQuery,
    schema: &Arc<Schema>,
    tokenizers: &TokenizerRegistry,
) -> Result<Box<dyn Query>, JsValue> {
    if let Some(ref tq) = js.term {
        let field = schema
            .get_field(&tq.field)
            .ok_or_else(|| JsValue::from_str(&format!("Unknown field: '{}'", tq.field)))?;
        let tokenizer_name = schema
            .get_field_entry(field)
            .and_then(|e| e.tokenizer.as_deref())
            .unwrap_or("simple");
        let tok = tokenizers
            .get(tokenizer_name)
            .unwrap_or_else(|| Box::new(hermes_core::SimpleTokenizer));
        let tokens: Vec<String> = tok
            .tokenize(&tq.value)
            .into_iter()
            .map(|t| t.text)
            .collect();
        if tokens.is_empty() {
            return Err(JsValue::from_str("No tokens in term query"));
        }
        if tokens.len() == 1 {
            return Ok(Box::new(TermQuery::text(field, &tokens[0])));
        }
        let mut bq = BooleanQuery::new();
        for token in tokens {
            bq = bq.must(TermQuery::text(field, &token));
        }
        return Ok(Box::new(bq));
    }

    if let Some(ref mq) = js.match_ {
        let field = schema
            .get_field(&mq.field)
            .ok_or_else(|| JsValue::from_str(&format!("Unknown field: '{}'", mq.field)))?;
        let tokenizer_name = schema
            .get_field_entry(field)
            .and_then(|e| e.tokenizer.as_deref())
            .unwrap_or("simple");
        let tok = tokenizers
            .get(tokenizer_name)
            .unwrap_or_else(|| Box::new(hermes_core::SimpleTokenizer));
        let tokens: Vec<String> = tok.tokenize(&mq.text).into_iter().map(|t| t.text).collect();
        if tokens.is_empty() {
            return Err(JsValue::from_str("No tokens in match query"));
        }
        if tokens.len() == 1 {
            return Ok(Box::new(TermQuery::text(field, &tokens[0])));
        }
        let mut bq = BooleanQuery::new();
        for token in tokens {
            bq = bq.should(TermQuery::text(field, &token));
        }
        return Ok(Box::new(bq));
    }

    if let Some(ref bq) = js.boolean {
        let mut query = BooleanQuery::new();
        for q in &bq.must {
            let inner = convert_query(q, schema, tokenizers)?;
            query.must.push(inner.into());
        }
        for q in &bq.should {
            let inner = convert_query(q, schema, tokenizers)?;
            query.should.push(inner.into());
        }
        for q in &bq.must_not {
            let inner = convert_query(q, schema, tokenizers)?;
            query.must_not.push(inner.into());
        }
        return Ok(Box::new(query));
    }

    if let Some(ref pq) = js.prefix {
        let field = schema
            .get_field(&pq.field)
            .ok_or_else(|| JsValue::from_str(&format!("Unknown field: '{}'", pq.field)))?;
        return Ok(Box::new(PrefixQuery::text(field, &pq.value)));
    }

    if let Some(ref sq) = js.sparse_vector {
        let field = schema
            .get_field(&sq.field)
            .ok_or_else(|| JsValue::from_str(&format!("Unknown field: '{}'", sq.field)))?;
        if sq.indices.len() != sq.values.len() {
            return Err(JsValue::from_str(
                "sparseVector: indices and values must have the same length",
            ));
        }
        let vector: Vec<(u32, f32)> = sq
            .indices
            .iter()
            .zip(sq.values.iter())
            .map(|(&i, &v)| (i, v))
            .collect();
        let mut query = SparseVectorQuery::new(field, vector);
        if let Some(hf) = sq.heap_factor {
            query = query.with_heap_factor(hf);
        }
        return Ok(Box::new(query));
    }

    if let Some(ref dq) = js.dense_vector {
        let field = schema
            .get_field(&dq.field)
            .ok_or_else(|| JsValue::from_str(&format!("Unknown field: '{}'", dq.field)))?;
        let mut query = DenseVectorQuery::new(field, dq.vector.clone());
        if let Some(np) = dq.nprobe {
            query = query.with_nprobe(np);
        }
        if let Some(rf) = dq.rerank_factor {
            query = query.with_rerank_factor(rf);
        }
        return Ok(Box::new(query));
    }

    Err(JsValue::from_str(
        "Invalid query: must contain exactly one of: term, match, boolean, prefix, sparseVector, denseVector",
    ))
}
