\echo Use "ALTER EXTENSION evoke UPDATE TO '0.2.6'" to load this file. \quit

-- The published 0.2.5 and 0.2.6 install catalogs are byte-identical.
-- This binary-only privilege fix changes no SQL objects or index formats.
-- PostgreSQL advances extversion; existing objects, ACLs and indexes remain.
