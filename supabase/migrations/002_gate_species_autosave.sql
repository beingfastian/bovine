-- Migration 002 — Gate 1, species detection, auto-save
-- ======================================================
-- Run once in the SQL editor, on top of schema.sql. Idempotent.
--
-- What changed in the app and why the schema follows:
--
--   1. A photo is now saved the moment it is scored, BEFORE the person answers
--      anything. So `species` and `reported_lumps` on screenings become
--      nullable, and the human answers move to their own table.
--
--   2. Anonymous visitors still have no UPDATE right anywhere -- that is the
--      security model -- so a later answer cannot edit the row. It is an
--      INSERT into screening_labels instead. Several answers per screening are
--      allowed; the newest wins. The id is a client-generated random UUID, so
--      labelling someone else's screening would mean guessing 122 random bits.
--
--   3. The model now also emits P(cattle), P(buffalo), P(other). Those are
--      stored raw so the gate threshold can be retuned from field data later.

-- ---------------------------------------------------------------- screenings
alter table public.screenings
  add column if not exists animal_present   boolean,
  add column if not exists detected_species text
    check (detected_species in ('cattle', 'buffalo', 'other')),
  add column if not exists p_cattle         real check (p_cattle  >= 0 and p_cattle  <= 1),
  add column if not exists p_buffalo        real check (p_buffalo >= 0 and p_buffalo <= 1),
  add column if not exists p_other          real check (p_other   >= 0 and p_other   <= 1),
  -- Mahalanobis d^2 of the photo's features from the bovine training
  -- distribution (the open-set half of Gate 1). Stored raw so oodMax can be
  -- retuned from field data.
  add column if not exists ood_distance     real check (ood_distance >= 0),
  add column if not exists app_version      text;

alter table public.screenings alter column species        drop not null;
alter table public.screenings alter column reported_lumps drop not null;

-- 'no_animal' is a new reason the model can abstain.
alter table public.screenings drop constraint if exists screenings_abstain_reason_check;
alter table public.screenings add constraint screenings_abstain_reason_check
  check (abstain_reason in ('no_animal', 'blurry', 'dark', 'bright', 'borderline_score'));

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

-- ------------------------------------------------------------------ cleanup
-- Rows and photos created by the automated end-to-end test on 2026-09-20.
delete from public.screenings where user_agent ilike '%HeadlessChrome%';
delete from storage.objects
where bucket_id = 'screenings'
  and (name like '_setup_check/%' or name like '2026-09-20/%');

-- Photos whose row insert failed (the client uploads first, then inserts; a
-- schema mismatch between app and database leaves the photo with no row).
-- Safe to re-run at any time: a photo with no screening row is unreachable.
delete from storage.objects
where bucket_id = 'screenings'
  and name not in (select image_path from public.screenings where image_path is not null);
