-- Pass-through during the migration: the silver table is already at the
-- grain Power BI reads. Materialised as Parquet in gold/star so the report
-- reads one folder only.
select * from {{ source('silver', 'dim_firecause') }}
