alter table "public"."highlight" add column "article_title" text default ''::text not null;

alter table "public"."highlight" add column "local_id" text default ''::text not null;

alter table "public"."highlight" add column "text" text default ''::text not null;
