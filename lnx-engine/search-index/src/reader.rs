use std::borrow::Cow;
use std::cmp::Reverse;
use std::sync::Arc;
use std::time::Instant;

use aexecutor::SearcherExecutorPool;
use anyhow::{anyhow, Error, Result};
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use tantivy::collector::{Count, TopDocs};
use tantivy::query::{Query, TermQuery};
use tantivy::schema::{Field, FieldType, IndexRecordOption, Schema, Value};
use tantivy::{DateTime, DocAddress, DocId, Executor, IndexReader, ReloadPolicy, Score, Searcher, SegmentReader, SnippetGenerator, Term};

use crate::helpers::{AsScore, Validate};
use crate::query::{DocumentId, QueryBuilder, QuerySelector};
use crate::schema::SchemaContext;
use crate::structures::{CompliantDocumentValue, DocumentHit, IndexContext};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ReaderContext {
    /// The number of reader threads to use.
    ///
    /// The current implementation is rather naive:
    /// multithreading is by splitting search into as many task as there are segments.
    /// It is powerless at making search faster if your index consists in one large segment.
    /// Also, keep in my multithreading a single query on several threads will not improve
    /// your throughput. It can actually hurt it.
    /// It will however, decrease the average response time.
    #[serde(default = "ReaderContext::default_reader_threads")]
    reader_threads: usize,

    /// The maximum searches that can be done at any one time.
    max_concurrency: usize,
}

impl Validate for ReaderContext {
    fn validate(&self) -> Result<()> {
        if self.max_concurrency == 0 {
            return Err(Error::msg("max concurrency must be at least 1."));
        }

        Ok(())
    }
}

impl ReaderContext {
    fn default_reader_threads() -> usize {
        1
    }
}

/// A given query payload that describes how the reader should
/// search the index.
#[derive(Debug, Deserialize)]
pub struct QueryPayload {
    /// The query(s) itself.
    pub(crate) query: QuerySelector,

    /// The amount of results to limit by.
    #[serde(default = "QueryPayload::default_limit")]
    pub(crate) limit: usize,

    /// The amount of documents to skip before getting the results.
    #[serde(default)]
    pub(crate) offset: usize,

    /// A specified field to order results by, this defaults to the
    /// score of the indexed documents (relevancy).
    pub(crate) order_by: Option<String>,

    /// How to sort the data (asc/desc).
    #[serde(default)]
    pub(crate) sort: Sort,

    /// If you want to generate snippets you need to send the size of it
    #[serde(default)]
    pub(crate) snippet_size: Option<usize>,

    /// If you want to generate snippet you need to set names of the fields that will be snipped
    #[serde(default)]
    pub(crate) snippet_fields: Vec<String>
}

impl QueryPayload {
    fn default_limit() -> usize {
        20
    }
}

/// What order to sort the returned data.
#[derive(Debug, Copy, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Sort {
    /// Sort the data in ascending order.
    Asc,

    /// Sort the data in descending order. (Default)
    Desc,
}

impl Default for Sort {
    fn default() -> Self {
        Self::Desc
    }
}

#[derive(Debug, Serialize)]
pub struct QueryResults {
    /// The retrieved documents.
    pub(crate) hits: Vec<DocumentHit>,

    /// The total amount of documents matching the search
    count: usize,

    /// The amount of time taken to search in seconds.
    time_taken: f32,
}

impl QueryResults {
    #[inline]
    pub fn len(&self) -> usize {
        self.hits.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.hits.len() == 0
    }
}

/// Attaches an order by clause to the collector.
///
/// This collected the values with be returned in the order according to the
/// given field value.
fn order_and_search<R: AsScore + tantivy::fastfield::FastValue>(
    searcher: &Searcher,
    field: Field,
    query: &Box<dyn Query>,
    limit: usize,
    offset: usize,
    executor: &Executor,
) -> Result<(Vec<(R, DocAddress)>, usize)> {
    let collector = TopDocs::with_limit(limit + offset);
    let collector = collector.order_by_fast_field(field);
    let collector = (collector, Count);

    let (result_addresses, count) = searcher
        .search_with_executor(query, &collector, executor)
        .map_err(Error::from)?;

    let results = result_addresses
        .into_iter()
        .skip(offset)
        .take(limit)
        .collect::<Vec<_>>();

    Ok((results, count))
}

macro_rules! execute_staged_search {
    ($query:expr, $searcher:expr, $collector:expr, $executor:expr, $limit:expr, $offset:expr) => {{
        let collector = ($collector, Count);

        let (results, count) = $searcher
            .search_with_executor(&$query, &collector, $executor)
            .map_err(Error::from)?;

        let results = results
            .into_iter()
            .skip($offset)
            .take($limit)
            .collect::<Vec<_>>();

        Ok::<_, anyhow::Error>((results, count))
    }};
}

/// Performs the search operation and processes the returned results.
fn process_search<S: AsScore>(
    ctx: &SchemaContext,
    searcher: &Searcher,
    schema: &Schema,
    top_docs: Vec<(S, DocAddress)>,
    query: &Box<dyn Query>,
    snippet_size: Option<usize>,
    snippet_fields: Vec<String>,
) -> Result<Vec<DocumentHit>> {
    println!("Process search, snippet_size = {:?}, snippet_fields = {:?}, query:\n{:#?}", &snippet_size, &snippet_fields, &query);
    let mut hits = Vec::with_capacity(top_docs.len());
    let mut ms = 0;
    //let start = Instant::now();
    let mut snippet_generators = Vec::new();
    for field_name in &snippet_fields {
        match schema.get_field(field_name) {
            Some(field) => {
                let mut snippet_generator = SnippetGenerator::create(&searcher, &*query, field)?;
                snippet_generator.set_max_num_chars(snippet_size.unwrap());
                snippet_generators.push((field_name, snippet_generator));
            }
            None => {}
        }
    }
    //ms += start.elapsed().as_millis();
    for (ratio, ref_address) in top_docs {
        let retrieved_doc = searcher.doc(ref_address)?;
        let start = Instant::now();
        let mut snippets = HashMap::new();
        for (name, generator) in &snippet_generators {
            let snippet = generator.snippet_from_doc(&retrieved_doc);
            let html = snippet.to_html();
            snippets.insert(*name, html);
        }
        ms += start.elapsed().as_millis();

        let mut doc = schema.to_named_doc(&retrieved_doc);
        let id = doc.0
            .remove("_id")
            .ok_or_else(|| Error::msg("document has been missed labeled (missing primary key '_id'), the dataset is invalid"))?;

        if let Value::U64(doc_id) = id[0] {
            let mut hit = DocumentHit::from_tantivy_document(ctx, doc_id, doc, ratio.as_score());
            for (field, snippet) in &snippets {
                if let Some(value) = hit.doc.get_mut(field.as_str()) {
                    *value = Some(CompliantDocumentValue::Single(Value::Str(snippet.to_owned())));
                }
            }
            hits.push(hit);
        } else {
            return Err(Error::msg("document has been missed labeled (missing identifier tag), the dataset is invalid"));
        }
    }
    println!("Time for snippet calculation: {} ms", ms);

    Ok(hits)
}

/// Orders the search results by the given field with a given sort (ASC, DESC)
///
/// This function is super messy just because of all the type inference
/// so any contributions to clean this up would be very appreciated.
#[allow(clippy::too_many_arguments)]
fn order_and_sort(
    sort: Sort,
    field: Field,
    query: Box<dyn Query>,
    ctx: &SchemaContext,
    schema: &Schema,
    searcher: &Searcher,
    limit: usize,
    offset: usize,
    executor: &Executor,
    snippet_size: Option<usize>,
    snippet_fields: Vec<String>,
) -> Result<(Vec<DocumentHit>, usize)> {
    let is_multi_value = ctx
        .multi_value_fields()
        .contains(schema.get_field_name(field));

    if is_multi_value {
        return Err(anyhow!(
            "multi-value fields cannot be used to sort results see issue #70"
        ));
    }

    let field_type = schema.get_field_entry(field).field_type();
    if let Sort::Desc = sort {
        return match field_type {
            FieldType::I64(_) => {
                let out: (Vec<(i64, DocAddress)>, usize) =
                    order_and_search(searcher, field, &query, limit, offset, executor)?;
                Ok((process_search(ctx, searcher, schema, out.0, &query, snippet_size, snippet_fields)?, out.1))
            },
            FieldType::U64(_) => {
                let out: (Vec<(u64, DocAddress)>, usize) =
                    order_and_search(searcher, field, &query, limit, offset, executor)?;
                Ok((process_search(ctx, searcher, schema, out.0, &query, snippet_size, snippet_fields)?, out.1))
            },
            FieldType::F64(_) => {
                let out: (Vec<(f64, DocAddress)>, usize) =
                    order_and_search(searcher, field, &query, limit, offset, executor)?;
                Ok((process_search(ctx, searcher, schema, out.0, &query, snippet_size, snippet_fields)?, out.1))
            },
            FieldType::Date(_) => {
                let out: (Vec<(DateTime, DocAddress)>, usize) =
                    order_and_search(searcher, field, &query, limit, offset, executor)?;
                Ok((process_search(ctx, searcher, schema, out.0, &query, snippet_size, snippet_fields)?, out.1))
            },
            _ => Err(Error::msg("field is not a fast field")),
        };
    }

    let collector = TopDocs::with_limit(limit + offset);
    let out = match field_type {
        FieldType::I64(_) => {
            let collector =
                collector.custom_score(move |segment_reader: &SegmentReader| {
                    let reader = segment_reader
                        .fast_fields()
                        .i64(field)
                        .expect("field exists");

                    move |doc: DocId| {
                        let value: i64 = reader.get_val(doc);
                        std::cmp::Reverse(value)
                    }
                });

            let out: (Vec<(Reverse<i64>, DocAddress)>, usize) = execute_staged_search!(
                query, searcher, collector, executor, limit, offset
            )?;
            (process_search(ctx, searcher, schema, out.0, &query, snippet_size, snippet_fields)?, out.1)
        },
        FieldType::U64(_) => {
            let collector =
                collector.custom_score(move |segment_reader: &SegmentReader| {
                    let reader = segment_reader
                        .fast_fields()
                        .u64(field)
                        .expect("field exists");

                    move |doc: DocId| {
                        let value: u64 = reader.get_val(doc);
                        std::cmp::Reverse(value)
                    }
                });

            let out: (Vec<(Reverse<u64>, DocAddress)>, usize) = execute_staged_search!(
                query, searcher, collector, executor, limit, offset
            )?;
            (process_search(ctx, searcher, schema, out.0, &query, snippet_size, snippet_fields)?, out.1)
        },
        FieldType::F64(_) => {
            let collector =
                collector.custom_score(move |segment_reader: &SegmentReader| {
                    let reader = segment_reader
                        .fast_fields()
                        .f64(field)
                        .expect("field exists");

                    move |doc: DocId| {
                        let value: f64 = reader.get_val(doc);
                        std::cmp::Reverse(value)
                    }
                });

            let out: (Vec<(Reverse<f64>, DocAddress)>, usize) = execute_staged_search!(
                query, searcher, collector, executor, limit, offset
            )?;
            (process_search(ctx, searcher, schema, out.0, &query, snippet_size, snippet_fields)?, out.1)
        },
        FieldType::Date(_) => {
            let collector =
                collector.custom_score(move |segment_reader: &SegmentReader| {
                    let reader = segment_reader
                        .fast_fields()
                        .date(field)
                        .expect("field exists");

                    move |doc: DocId| {
                        let value: DateTime = reader.get_val(doc);
                        std::cmp::Reverse(value)
                    }
                });

            let out: (Vec<(Reverse<DateTime>, DocAddress)>, usize) = execute_staged_search!(
                query, searcher, collector, executor, limit, offset
            )?;
            (process_search(ctx, searcher, schema, out.0, &query, snippet_size, snippet_fields)?, out.1)
        },
        _ => return Err(Error::msg("field is not a fast field")),
    };

    Ok(out)
}

/// The reader of the given index.
///
/// This manages all searches on the index which encompasses the concurrency
/// limiters and thread pool execution.
///
/// Each index should only have on `Reader` instance.
#[derive(Clone)]
pub(crate) struct Reader {
    index_name: Cow<'static, str>,

    schema_ctx: Cow<'static, SchemaContext>,

    /// The executor pool.
    pool: crate::ReaderExecutor,

    /// The query factory system.
    query_handler: Arc<QueryBuilder>,
}

impl Reader {
    /// Creates a new reader from the given index context.
    #[instrument(name = "index-reader", skip_all)]
    pub(crate) async fn create(ctx: &IndexContext) -> Result<Self> {
        let reader: IndexReader = ctx
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommit)
            .num_warming_threads(ctx.reader_ctx.max_concurrency)
            .try_into()?;
        info!(
            "index reader created with reload policy=OnCommit, num_searchers={}",
            ctx.reader_ctx.max_concurrency,
        );

        let pool = {
            let pool = SearcherExecutorPool::create(
                reader,
                ctx.reader_ctx.reader_threads,
                ctx.reader_ctx.max_concurrency,
            )
            .await?;
            Arc::new(pool)
        };
        info!(
            "executor pool has successfully started! max_concurrency={}, total_threads={}",
            ctx.reader_ctx.max_concurrency,
            ctx.reader_ctx.max_concurrency * ctx.reader_ctx.reader_threads
        );

        let query_ctx = ctx.query_ctx.clone();
        let query_handler = QueryBuilder::new(
            query_ctx,
            ctx.stop_words.clone(),
            ctx.correction_manager.clone(),
            ctx.synonyms.clone(),
            &ctx.index,
            pool.clone(),
        );
        info!(
            "query builder constructed with config: fast-fuzzy={} strip-stop-words={}.",
            ctx.query_ctx.use_fast_fuzzy, ctx.query_ctx.strip_stop_words,
        );

        Ok(Self {
            index_name: Cow::Owned(ctx.name()),
            schema_ctx: Cow::Owned(ctx.schema_ctx.clone()),
            pool,
            query_handler: Arc::new(query_handler),
        })
    }

    /// Gets a list of suggested corrections based off of the index corpus.
    pub fn get_corrected_query_hint(&self, query: &str) -> String {
        self.query_handler.get_corrected_query_hint(query)
    }

    /// Gets a singular document from the given id.
    ///
    /// If no document is found an error is raised without context.
    #[instrument(name = "document-fetcher", skip(self), fields(index = %self.index_name))]
    pub(crate) async fn get_document(&self, id: DocumentId) -> Result<DocumentHit> {
        let id_field = self.query_handler.id_field();
        let document = self
            .pool
            .spawn(move |searcher, executor| {
                let qry = TermQuery::new(
                    Term::from_field_u64(id_field, id),
                    IndexRecordOption::Basic,
                );

                let mut results = searcher.search_with_executor(
                    &qry,
                    &TopDocs::with_limit(1),
                    executor,
                )?;
                if results.is_empty() {
                    return Err(Error::msg(format!(
                        "no document exists with id: '{}'",
                        id
                    )));
                }

                let (_, addr) = results.remove(0);
                let doc = searcher.doc(addr)?;
                let schema = searcher.schema();

                Ok(schema.to_named_doc(&doc))
            })
            .await??;

        Ok(DocumentHit::from_tantivy_document(
            self.schema_ctx.as_ref(),
            id,
            document,
            Some(1.0),
        ))
    }

    /// Searches the index reader with the given query payload.
    ///
    /// The payload determines the behaviour of the query results.
    /// The actual behaviour of how a query is built is upto the query handler
    /// which will parse and interpret the given data.
    #[instrument(name = "document-searcher", skip_all, fields(index = %self.index_name))]
    pub(crate) async fn search(&self, qry: QueryPayload) -> Result<QueryResults> {
        let start = std::time::Instant::now();

        let limit = qry.limit;
        let sort = qry.sort;
        let order_by = qry.order_by;
        let offset = qry.offset;
        let queries = self.query_handler.build_query(qry.query).await?;
        let ctx = self.schema_ctx.clone();
        let snippet_size = qry.snippet_size;
        let snippet_field = qry.snippet_fields;

        let (hits, count) = self
            .pool
            .spawn(move |searcher, executor| {
                let schema = searcher.schema();
                let order_by = order_by.map(|v| schema.get_field(&v));

                let (hits, count) = if let Some(Some(field)) = order_by {
                    order_and_sort(
                        sort,
                        field,
                        queries,
                        ctx.as_ref(),
                        schema,
                        &searcher,
                        limit,
                        offset,
                        executor,
                        snippet_size,
                        snippet_field
                    )?
                } else {
                    let collector = TopDocs::with_limit(limit + offset);
                    let out: (Vec<(Score, DocAddress)>, usize) = execute_staged_search!(
                        queries, searcher, collector, executor, limit, offset
                    )?;

                    (
                        process_search(ctx.as_ref(), &searcher, schema, out.0, &queries, snippet_size, snippet_field)?,
                        out.1,
                    )
                };

                Ok::<_, Error>((hits, count))
            })
            .await??;

        let elapsed = start.elapsed();

        Ok(QueryResults {
            time_taken: elapsed.as_secs_f32(), // filled in by handler later
            hits,
            count,
        })
    }

    pub(crate) fn get_synonyms(&self) -> HashMap<String, Box<[String]>> {
        self.query_handler.synonyms()
    }

    pub(crate) fn get_stop_words(&self) -> Vec<String> {
        self.query_handler.stop_words()
    }

    /// This forces the reader to reload after a commit.
    pub(crate) fn force_reload(&self) -> Result<()> {
        self.pool.reload()
    }

    /// This should not be used for general things.
    ///
    /// This is an internal export to allow the writer
    /// to have access to the segment reader information.
    pub(crate) fn get_searcher(&self) -> Searcher {
        self.pool.searcher()
    }
}
