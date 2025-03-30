create extension if not exists "vector" with schema "public" version '0.8.0';

alter table "public"."bookmark" add column "tags" text[] not null;