create table "public"."analysis" (
    "id" bigint generated always as identity not null,
    "user_id" uuid not null,
    "url" text,
    "article_content" text not null,
    "summary" text not null,
    "context" text not null,
    "key_terms" jsonb not null,
    "questions" jsonb not null,
    "short_response_questions" jsonb not null,
    "future_research" jsonb not null,
    "created_at" timestamp with time zone default CURRENT_TIMESTAMP
);


alter table "public"."analysis" enable row level security;

CREATE UNIQUE INDEX analysis_pkey ON public.analysis USING btree (id);

alter table "public"."analysis" add constraint "analysis_pkey" PRIMARY KEY using index "analysis_pkey";

alter table "public"."analysis" add constraint "analysis_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) not valid;

alter table "public"."analysis" validate constraint "analysis_user_id_fkey";

grant delete on table "public"."analysis" to "anon";

grant insert on table "public"."analysis" to "anon";

grant references on table "public"."analysis" to "anon";

grant select on table "public"."analysis" to "anon";

grant trigger on table "public"."analysis" to "anon";

grant truncate on table "public"."analysis" to "anon";

grant update on table "public"."analysis" to "anon";

grant delete on table "public"."analysis" to "authenticated";

grant insert on table "public"."analysis" to "authenticated";

grant references on table "public"."analysis" to "authenticated";

grant select on table "public"."analysis" to "authenticated";

grant trigger on table "public"."analysis" to "authenticated";

grant truncate on table "public"."analysis" to "authenticated";

grant update on table "public"."analysis" to "authenticated";

grant delete on table "public"."analysis" to "service_role";

grant insert on table "public"."analysis" to "service_role";

grant references on table "public"."analysis" to "service_role";

grant select on table "public"."analysis" to "service_role";

grant trigger on table "public"."analysis" to "service_role";

grant truncate on table "public"."analysis" to "service_role";

grant update on table "public"."analysis" to "service_role";

create policy "Enable insert for users based on user_id"
on "public"."analysis"
as permissive
for all
to public
using ((( SELECT auth.uid() AS uid) = user_id))
with check ((( SELECT auth.uid() AS uid) = user_id));



