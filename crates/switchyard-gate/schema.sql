-- SPDX-License-Identifier: Apache-2.0
--
-- Minimal Postgres schema for switchyard-gate.
--
-- The gate connects as the switchyard_gate role and calls exactly two functions:
-- gate.lookup_key(hash) and gate.lookup_user_by_email(email). Each returns zero or one
-- row with the columns the gate reads. Everything else here (tables, columns you add,
-- how users sign up) is yours to change, as long as those two functions keep their
-- column names and types.
--
-- The program that reads the usage:events stream from Valkey and stores it is not part
-- of this file.
--
-- To insert a key, store the lowercase SHA-256 hex of the plain key:
--   insert into gate.api_keys (user_id, key_hash)
--   values ('<user uuid>', encode(sha256(convert_to('plain-key', 'UTF8')), 'hex'));

create schema gate;

create table gate.users (
    id                uuid primary key default gen_random_uuid(),
    email             text not null unique,
    role              text not null default 'user',    -- 'user', 'admin' or 'pending'
    status            text not null default 'active',  -- anything else is refused
    trusted_forwarder boolean not null default false,
    rate_limit_rpm    integer,          -- NULL means unlimited
    rate_limit_tpm    bigint,
    hourly_limit      bigint,
    daily_limit       bigint,
    weekly_limit      bigint,
    monthly_limit     bigint,
    w_input           double precision not null default 1.0,
    w_cache_read      double precision not null default 0.1,
    w_cache_write     double precision not null default 1.0,
    w_output          double precision not null default 1.0,
    w_reasoning       double precision not null default 1.0
);

create table gate.api_keys (
    id       uuid primary key default gen_random_uuid(),
    user_id  uuid not null references gate.users (id) on delete cascade,
    key_hash text not null unique,  -- lowercase SHA-256 hex of the plain key
    revoked  boolean not null default false
);

-- The row both functions return. The gate casts key_id and user_id to text in its
-- query, so uuid is fine here.
create type gate.identity as (
    key_id uuid, user_id uuid, role text, status text, trusted_forwarder boolean,
    rate_limit_rpm integer, rate_limit_tpm bigint, hourly_limit bigint, daily_limit bigint,
    weekly_limit bigint, monthly_limit bigint, w_input double precision,
    w_cache_read double precision, w_cache_write double precision,
    w_output double precision, w_reasoning double precision
);

create or replace function gate.lookup_key(p_hash text)
returns setof gate.identity language sql stable security definer as $$
    select k.id, u.id, u.role, u.status, u.trusted_forwarder,
           u.rate_limit_rpm, u.rate_limit_tpm, u.hourly_limit, u.daily_limit,
           u.weekly_limit, u.monthly_limit, u.w_input, u.w_cache_read, u.w_cache_write,
           u.w_output, u.w_reasoning
    from gate.api_keys k join gate.users u on u.id = k.user_id
    where k.key_hash = p_hash and not k.revoked
$$;

-- key_id is NULL: a forwarded user is billed to the user, not to a key.
create or replace function gate.lookup_user_by_email(p_email text)
returns setof gate.identity language sql stable security definer as $$
    select null::uuid, u.id, u.role, u.status, u.trusted_forwarder,
           u.rate_limit_rpm, u.rate_limit_tpm, u.hourly_limit, u.daily_limit,
           u.weekly_limit, u.monthly_limit, u.w_input, u.w_cache_read, u.w_cache_write,
           u.w_output, u.w_reasoning
    from gate.users u
    where lower(u.email) = p_email
$$;

-- The gate's role may run the two functions and nothing else. Set a password after
-- creating it, or use another auth method.
do $$ begin
    if not exists (select from pg_roles where rolname = 'switchyard_gate') then
        create role switchyard_gate login;
    end if;
end $$;
grant usage on schema gate to switchyard_gate;
grant execute on function gate.lookup_key(text) to switchyard_gate;
grant execute on function gate.lookup_user_by_email(text) to switchyard_gate;
