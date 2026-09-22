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
                       ('no_animal', 'blurry', 'dark', 'bright', 'borderline_score')),
  inference_ms       int,

  -- image quality gate, stored so thresholds can be retuned from real data
  blur_variance      real,
  dark_fraction      real,
  bright_fraction    real,
  mean_luma          real,

  -- Gate 1 / species detector output (migration 002). Raw probabilities are
  -- stored so the gate threshold can be retuned from field data later.
  animal_present     boolean,
  detected_species   text         check (detected_species in ('cattle', 'buffalo', 'other')),
  p_cattle           real         check (p_cattle  >= 0 and p_cattle  <= 1),
  p_buffalo          real         check (p_buffalo >= 0 and p_buffalo <= 1),
  p_other            real         check (p_other   >= 0 and p_other   <= 1),
  ood_distance       real         check (ood_distance >= 0),
  app_version        text,

  -- Legacy label columns. Since migration 002 the person's answers live in
  -- screening_labels (the row is written before they answer); these stay
  -- nullable for rows written by the first release.
  species            text         check (species in ('cattle', 'buffalo')),
  reported_lumps     text         check (reported_lumps in ('yes', 'no', 'not_sure')),
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

-- ------------------------------------------------------------ screening_labels
create table if not exists public.screening_labels (
  id              uuid primary key default gen_random_uuid(),
  screening_id    uuid not null references public.screenings (id) on delete cascade,
  created_at      timestamptz not null default now(),
  species         text not null check (species in ('cattle', 'buffalo')),
  reported_lumps  text not null check (reported_lumps in ('yes', 'no', 'not_sure')),
  -- Did the person overrule the detector? A high rate here is the detector
  -- being wrong, or the question being unclear; either way worth knowing.
  species_changed boolean not null default false
);

create index if not exists screening_labels_screening_idx
  on public.screening_labels (screening_id, created_at desc);

alter table public.screening_labels enable row level security;

drop policy if exists "anon can insert labels" on public.screening_labels;
create policy "anon can insert labels"
  on public.screening_labels for insert
  to anon
  with check (true);

drop policy if exists "authenticated can read labels" on public.screening_labels;
create policy "authenticated can read labels"
  on public.screening_labels for select
  to authenticated
  using (true);

-- ------------------------------------------------ screenings_with_labels view
-- What /review reads: each screening joined to its newest human answer.
-- security_invoker so it runs as the caller -- anon gets nothing.
create or replace view public.screenings_with_labels
with (security_invoker = true) as
select
  s.*,
  l.species         as label_species,
  l.reported_lumps  as label_lumps,
  l.species_changed as label_species_changed,
  l.created_at      as labelled_at,
  -- the species to analyse by: the person's answer if given, else the detector's
  coalesce(l.species, case when s.detected_species in ('cattle', 'buffalo')
                           then s.detected_species end) as species_final
from public.screenings s
left join lateral (
  select species, reported_lumps, species_changed, created_at
  from public.screening_labels
  where screening_id = s.id
  order by created_at desc
  limit 1
) l on true;

revoke all on public.screenings_with_labels from anon;
grant select on public.screenings_with_labels to authenticated;

-- --------------------------------------------------------- collection_summary
-- Rebuilt on the new view, and with security_invoker -- as first written this
-- view ran as its owner and handed aggregate counts to the anon key.
drop view if exists public.collection_summary;
create view public.collection_summary
with (security_invoker = true) as
select
  species_final                                            as species,
  label_lumps                                              as reported_lumps,
  count(*)                                                 as n,
  count(*) filter (where verdict = 'possible_condition')   as model_flagged,
  count(*) filter (where verdict = 'unclear')              as model_unclear,
  count(*) filter (where animal_present = false)           as gate_rejected,
  round(avg(probability)::numeric, 4)                      as mean_probability,
  min(created_at)                                          as first_seen,
  max(created_at)                                          as last_seen
from public.screenings_with_labels
group by species_final, label_lumps
order by species_final, label_lumps;

revoke all on public.collection_summary from anon;
grant select on public.collection_summary to authenticated;
