drop policy "Enable insert for users based on user_id" on "public"."analysis";

alter table "public"."analysis" add column "title" text not null;

alter table "public"."analysis" alter column "created_at" set not null;

alter table "public"."analysis" alter column "url" set not null;

create policy "Enable delete for users based on user_id"
on "public"."analysis"
as permissive
for delete
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable insert for authenticated users only"
on "public"."analysis"
as permissive
for insert
to authenticated
with check (true);


create policy "Enable select for users based on user_id"
on "public"."analysis"
as permissive
for select
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable update for users based on user_id"
on "public"."analysis"
as permissive
for update
to public
using ((( SELECT auth.uid() AS uid) = user_id))
with check ((( SELECT auth.uid() AS uid) = user_id));



