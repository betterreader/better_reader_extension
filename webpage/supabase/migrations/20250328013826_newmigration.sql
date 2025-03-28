create extension if not exists "vector" with schema "public" version '0.8.0';

create table "public"."article_segments" (
    "id" uuid not null,
    "article_id" uuid not null,
    "segment_text" text not null,
    "embedding" vector(1536),
    "segment_index" integer not null,
    "created_at" timestamp with time zone default now(),
    "keywords" text[],
    "importance_score" double precision default 1.0
);


create table "public"."articles" (
    "id" uuid not null,
    "url" text not null,
    "title" text not null,
    "user_id" uuid,
    "created_at" timestamp with time zone default now(),
    "topics" text[],
    "summary" text
);


CREATE INDEX article_segments_embedding_idx ON public.article_segments USING ivfflat (embedding vector_cosine_ops) WITH (lists='100');

CREATE UNIQUE INDEX article_segments_pkey ON public.article_segments USING btree (id);

CREATE UNIQUE INDEX articles_pkey ON public.articles USING btree (id);

CREATE INDEX idx_article_segments_article_id ON public.article_segments USING btree (article_id);

CREATE INDEX idx_article_segments_keywords ON public.article_segments USING gin (keywords);

CREATE INDEX idx_articles_summary ON public.articles USING gin (to_tsvector('english'::regconfig, summary));

CREATE INDEX idx_articles_topics ON public.articles USING gin (topics);

CREATE INDEX idx_articles_url ON public.articles USING btree (url);

CREATE INDEX idx_articles_user_id ON public.articles USING btree (user_id);

CREATE UNIQUE INDEX unique_article_url ON public.articles USING btree (url);

CREATE UNIQUE INDEX unique_segment_per_article ON public.article_segments USING btree (article_id, segment_index);

alter table "public"."article_segments" add constraint "article_segments_pkey" PRIMARY KEY using index "article_segments_pkey";

alter table "public"."articles" add constraint "articles_pkey" PRIMARY KEY using index "articles_pkey";

alter table "public"."article_segments" add constraint "article_segments_article_id_fkey" FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE not valid;

alter table "public"."article_segments" validate constraint "article_segments_article_id_fkey";

alter table "public"."article_segments" add constraint "unique_segment_per_article" UNIQUE using index "unique_segment_per_article";

alter table "public"."articles" add constraint "unique_article_url" UNIQUE using index "unique_article_url";

set check_function_bodies = off;

CREATE OR REPLACE FUNCTION public.find_similar_articles(article_embedding vector, current_article_id uuid, match_limit integer)
 RETURNS TABLE(similarity double precision, articles jsonb)
 LANGUAGE plpgsql
AS $function$
BEGIN
    RETURN QUERY
    SELECT
        1 - (article_segments.embedding <=> article_embedding) AS similarity,
        jsonb_build_object(
            'id', a.id,
            'url', a.url,
            'title', a.title,
            'topics', a.topics
        ) AS articles
    FROM
        article_segments
    JOIN
        articles a ON article_segments.article_id = a.id
    WHERE
        article_segments.article_id != current_article_id
    ORDER BY
        article_segments.embedding <=> article_embedding
    LIMIT match_limit;
END;
$function$
;

CREATE OR REPLACE FUNCTION public.match_article_segments(query_embedding vector, article_filter uuid, match_threshold double precision, match_limit integer)
 RETURNS TABLE(id uuid, article_id uuid, segment_text text, segment_index integer, similarity double precision, keywords text[], importance_score double precision)
 LANGUAGE plpgsql
AS $function$
BEGIN
    RETURN QUERY
    SELECT
        article_segments.id,
        article_segments.article_id,
        article_segments.segment_text,
        article_segments.segment_index,
        1 - (article_segments.embedding <=> query_embedding) AS similarity,
        article_segments.keywords,
        article_segments.importance_score
    FROM
        article_segments
    WHERE
        article_segments.article_id = article_filter
        AND 1 - (article_segments.embedding <=> query_embedding) > match_threshold
    ORDER BY
        article_segments.embedding <=> query_embedding
    LIMIT match_limit;
END;
$function$
;

CREATE OR REPLACE FUNCTION public.match_segments(query_embedding vector, match_threshold double precision, match_limit integer)
 RETURNS TABLE(id uuid, article_id uuid, segment_text text, segment_index integer, similarity double precision, keywords text[], importance_score double precision)
 LANGUAGE plpgsql
AS $function$
BEGIN
    RETURN QUERY
    SELECT
        article_segments.id,
        article_segments.article_id,
        article_segments.segment_text,
        article_segments.segment_index,
        1 - (article_segments.embedding <=> query_embedding) AS similarity,
        article_segments.keywords,
        article_segments.importance_score
    FROM
        article_segments
    WHERE
        1 - (article_segments.embedding <=> query_embedding) > match_threshold
    ORDER BY
        article_segments.embedding <=> query_embedding
    LIMIT match_limit;
END;
$function$
;

CREATE OR REPLACE FUNCTION public.search_segments(query_embedding vector, match_limit integer)
 RETURNS TABLE(id uuid, article_id uuid, segment_text text, segment_index integer, created_at timestamp with time zone, similarity double precision)
 LANGUAGE plpgsql
AS $function$
BEGIN
  RETURN QUERY
  SELECT
    article_segments.id,
    article_segments.article_id,
    article_segments.segment_text,
    article_segments.segment_index,
    article_segments.created_at,
    1 - (article_segments.embedding <=> query_embedding) as similarity
  FROM
    article_segments
  ORDER BY
    article_segments.embedding <=> query_embedding
  LIMIT match_limit;
END;
$function$
;

CREATE OR REPLACE FUNCTION public.search_segments_by_article(query_embedding vector, match_limit integer, article_filter uuid)
 RETURNS TABLE(id uuid, article_id uuid, segment_text text, segment_index integer, created_at timestamp with time zone, similarity double precision)
 LANGUAGE plpgsql
AS $function$
BEGIN
  RETURN QUERY
  SELECT
    article_segments.id,
    article_segments.article_id,
    article_segments.segment_text,
    article_segments.segment_index,
    article_segments.created_at,
    1 - (article_segments.embedding <=> query_embedding) as similarity
  FROM
    article_segments
  WHERE
    article_segments.article_id = article_filter
  ORDER BY
    article_segments.embedding <=> query_embedding
  LIMIT match_limit;
END;
$function$
;

grant delete on table "public"."article_segments" to "anon";

grant insert on table "public"."article_segments" to "anon";

grant references on table "public"."article_segments" to "anon";

grant select on table "public"."article_segments" to "anon";

grant trigger on table "public"."article_segments" to "anon";

grant truncate on table "public"."article_segments" to "anon";

grant update on table "public"."article_segments" to "anon";

grant delete on table "public"."article_segments" to "authenticated";

grant insert on table "public"."article_segments" to "authenticated";

grant references on table "public"."article_segments" to "authenticated";

grant select on table "public"."article_segments" to "authenticated";

grant trigger on table "public"."article_segments" to "authenticated";

grant truncate on table "public"."article_segments" to "authenticated";

grant update on table "public"."article_segments" to "authenticated";

grant delete on table "public"."article_segments" to "service_role";

grant insert on table "public"."article_segments" to "service_role";

grant references on table "public"."article_segments" to "service_role";

grant select on table "public"."article_segments" to "service_role";

grant trigger on table "public"."article_segments" to "service_role";

grant truncate on table "public"."article_segments" to "service_role";

grant update on table "public"."article_segments" to "service_role";

grant delete on table "public"."articles" to "anon";

grant insert on table "public"."articles" to "anon";

grant references on table "public"."articles" to "anon";

grant select on table "public"."articles" to "anon";

grant trigger on table "public"."articles" to "anon";

grant truncate on table "public"."articles" to "anon";

grant update on table "public"."articles" to "anon";

grant delete on table "public"."articles" to "authenticated";

grant insert on table "public"."articles" to "authenticated";

grant references on table "public"."articles" to "authenticated";

grant select on table "public"."articles" to "authenticated";

grant trigger on table "public"."articles" to "authenticated";

grant truncate on table "public"."articles" to "authenticated";

grant update on table "public"."articles" to "authenticated";

grant delete on table "public"."articles" to "service_role";

grant insert on table "public"."articles" to "service_role";

grant references on table "public"."articles" to "service_role";

grant select on table "public"."articles" to "service_role";

grant trigger on table "public"."articles" to "service_role";

grant truncate on table "public"."articles" to "service_role";

grant update on table "public"."articles" to "service_role";