-- Rollup of fact_fireevent at country x month x source. One row per source:
-- EFFIS perimeters and BDIFF declarations are never summed together
-- (docs/data-model.md, Dim_Source as a guardrail).
-- Mirrors src/transform/rollups.py::build_countrymonth, proven equal by
-- tests/test_dbt_gold.py.
select
    country_key,
    cast(date_trunc('month', firedate_key) as timestamp) as month_key,
    source_key,
    sum(area_ha)                                        as burnt_area_ha,
    count(fire_event_key)                               as number_of_fires
from {{ source('silver', 'fact_fireevent') }}
group by 1, 2, 3
