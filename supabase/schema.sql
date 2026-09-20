-- BovineInsight field-collection schema
-- =====================================
-- Run this once in the Supabase SQL editor (Dashboard -> SQL Editor -> New query).
--
-- Security model, in one sentence: anonymous visitors may ADD a screening and
-- upload one photo, and may read nothing; only a signed-in account can read.
--
-- That asymmetry is the whole design. The app is shared by a public WhatsApp
-- link, so the anon key is effectively public. Anyone who has the link can
-- contribute data. Nobody who has the link can enumerate other people's
-- photographs of their animals.

-- ---------------------------------------------------------------- the table
create table if not exists public.screenings (
  id                 uuid primary key,
  created_at         timestamptz  not null default now(),   -- server time
  captured_at        timestamptz,                           -- client time; the
                                                            -- gap reveals offline use
  device_id          uuid,        -- anonymous, per-device; groups a farm's photos

  -- image
  image_path         text         not null,
  image_width        int,
  image_height       int,

  -- provenance: which model produced this verdict
  model_version      text         not null,
  thresholds_version text,
  inference_location text         not null default 'on-device',

  -- model output
  probability        real         not null check (probability >= 0 and probability <= 1),
  verdict            text         not null check (verdict in
                       ('possible_condition', 'unclear', 'no_obvious_lesion')),
  concern_band       text         check (concern_band in ('low', 'moderate', 'high')),
  abstain_reason     text         check (abstain_reason in
                       ('blurry', 'dark', 'bright', 'borderline_score')),
  inference_ms       int,

  -- image quality gate, stored so thresholds can be retuned from real data
  blur_variance      real,
  dark_fraction      real,
  bright_fraction    real,
  mean_luma          real,

  -- THE LABELS. Everything above exists to give these context.
  species            text         not null check (species in ('cattle', 'buffalo')),
  reported_lumps     text         not null check (reported_lumps in ('yes', 'no', 'not_sure')),
  species_validated  boolean      not null default false,

  -- filled in later, by you, from the review page
  vet_verdict        text,
  reviewer_note      text,
  reviewed_at        timestamptz,

  user_agent         text
);

-- The queries this table actually gets asked: "show me the newest", and
-- "show me every diseased buffalo", which is the gap the whole project exists
-- to close.
create index if not exists screenings_created_at_idx
  on public.screenings (created_at desc);

create index if not exists screenings_species_lumps_idx
  on public.screenings (species, reported_lumps);

create index if not exists screenings_device_idx
  on public.screenings (device_id, created_at desc);

-- ------------------------------------------------------------------- RLS
alter table public.screenings enable row level security;

-- Anonymous visitors may contribute.
drop policy if exists "anon can insert screenings" on public.screenings;
create policy "anon can insert screenings"
  on public.screenings for insert
  to anon
  with check (true);

-- ...and may read nothing. There is deliberately no SELECT policy for anon;
-- with RLS on, no policy means no rows.

-- Signed-in accounts (that is: you) can read and annotate.
drop policy if exists "authenticated can read screenings" on public.screenings;
create policy "authenticated can read screenings"
  on public.screenings for select
  to authenticated
  using (true);

drop policy if exists "authenticated can annotate screenings" on public.screenings;
create policy "authenticated can annotate screenings"
  on public.screenings for update
  to authenticated
  using (true)
  with check (true);

-- --------------------------------------------------------------- storage
-- Private bucket, 3 MB per object. The client downscales to 1024 px / q0.85,
-- which lands around 150-300 kB, so 3 MB is headroom rather than a target --
-- and it is the ceiling that stops one visitor filling the free tier.
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('screenings', 'screenings', false, 3145728, array['image/jpeg', 'image/png', 'image/webp'])
on conflict (id) do update
  set file_size_limit   = excluded.file_size_limit,
      allowed_mime_types = excluded.allowed_mime_types,
      public             = false;

drop policy if exists "anon can upload screening photos" on storage.objects;
create policy "anon can upload screening photos"
  on storage.objects for insert
  to anon
  with check (bucket_id = 'screenings');

-- No anon select, update or delete: a contributor cannot read back, overwrite
-- or erase what anyone else has sent, including themselves.

drop policy if exists "authenticated can read screening photos" on storage.objects;
create policy "authenticated can read screening photos"
  on storage.objects for select
  to authenticated
  using (bucket_id = 'screenings');

-- ------------------------------------------------------- convenience view
-- What the collection is actually worth, at a glance. The bottom-right cell --
-- buffalo with reported lumps -- is the number this project lives or dies by.
--
-- SECURITY: a view runs as its OWNER, not its caller, so by default this one
-- would read straight past the RLS policy above and hand anonymous visitors
-- aggregate counts, timestamps and mean scores. Measured, not theorised: as
-- first written it returned real numbers to the anon key. `security_invoker`
-- makes it run as the caller, so anon gets nothing and a signed-in account
-- gets everything. The revoke is belt and braces.
create or replace view public.collection_summary
with (security_invoker = true) as
select
  species,
  reported_lumps,
  count(*)                                         as n,
  count(*) filter (where verdict = 'possible_condition') as model_flagged,
  count(*) filter (where verdict = 'unclear')            as model_unclear,
  round(avg(probability)::numeric, 4)              as mean_probability,
  min(created_at)                                  as first_seen,
  max(created_at)                                  as last_seen
from public.screenings
group by species, reported_lumps
order by species, reported_lumps;

revoke all on public.collection_summary from anon;
grant select on public.collection_summary to authenticated;
