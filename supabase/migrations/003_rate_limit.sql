-- Migration 003 — rate limiting for anonymous inserts
-- ====================================================
-- Run once in the SQL editor, on top of 002. Idempotent.
--
-- ORDER MATTERS: deploy the client that INSERTS THE ROW BEFORE UPLOADING THE
-- PHOTO first (web commit "rate limit" or later), then run this. The new
-- storage policy refuses any upload whose path has no screening row yet; the
-- older client uploads first and would be refused every time.
--
-- Threat model, plainly: the app saves automatically and the anon key is
-- public, so anyone with the link can insert without pressing anything. One
-- bored person, or one stuck retry loop, could fill the free tier. Three
-- ceilings, all generous for real use:
--
--   per device   60 screenings / hour     (a real farm visit is a few dozen)
--   globally     1,000 screenings / hour  (protects against spoofed device ids)
--   per screening 20 labels total         (a person changing their mind)
--
-- And the photo itself can only be uploaded for a path that already has a
-- row, so storage cannot grow faster than the row ceiling allows. The row
-- trigger runs as the table owner, so it can count rows anon cannot read.

-- ------------------------------------------------------------ screenings
create or replace function public.screenings_rate_limit()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  per_device int;
  global_cnt int;
begin
  if new.device_id is not null then
    select count(*) into per_device
      from public.screenings
     where device_id = new.device_id
       and created_at > now() - interval '1 hour';
    if per_device >= 60 then
      raise exception 'rate limit: this device has saved % screenings in the last hour', per_device
        using errcode = 'P0001', hint = 'Try again later.';
    end if;
  end if;

  select count(*) into global_cnt
    from public.screenings
   where created_at > now() - interval '1 hour';
  if global_cnt >= 1000 then
    raise exception 'rate limit: the service is receiving too many screenings right now'
      using errcode = 'P0001', hint = 'Try again later.';
  end if;

  return new;
end;
$$;

drop trigger if exists screenings_rate_limit on public.screenings;
create trigger screenings_rate_limit
  before insert on public.screenings
  for each row execute function public.screenings_rate_limit();

-- The trigger's per-hour counts need this to stay cheap at any table size.
create index if not exists screenings_device_recent_idx
  on public.screenings (device_id, created_at desc);

-- ------------------------------------------------------- screening_labels
create or replace function public.labels_rate_limit()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  n int;
begin
  select count(*) into n
    from public.screening_labels
   where screening_id = new.screening_id;
  if n >= 20 then
    raise exception 'rate limit: too many answers for one screening'
      using errcode = 'P0001';
  end if;
  return new;
end;
$$;

drop trigger if exists labels_rate_limit on public.screening_labels;
create trigger labels_rate_limit
  before insert on public.screening_labels
  for each row execute function public.labels_rate_limit();

-- ----------------------------------------------------------------- storage
-- Anon cannot read screenings, so a plain subquery in the storage policy would
-- always be false. This SECURITY DEFINER function answers exactly one question
-- -- "does a row claim this path?" -- and nothing else leaks.
create or replace function public.screening_path_exists(p text)
returns boolean
language sql
security definer
stable
set search_path = public
as $$
  select exists (select 1 from public.screenings where image_path = p);
$$;

revoke all on function public.screening_path_exists(text) from public;
grant execute on function public.screening_path_exists(text) to anon, authenticated;

drop policy if exists "anon can upload screening photos" on storage.objects;
create policy "anon can upload screening photos"
  on storage.objects for insert
  to anon
  with check (
    bucket_id = 'screenings'
    and public.screening_path_exists(name)
  );
