-- Country x year x dominant land cover, EFFIS perimeters only: BDIFF carries
-- no land cover, and a perimeter without a dominant class is left out rather
-- than filed under "other". Mirrors src/transform/rollups.py::build_landcover.
select
    country_key,
    cast(date_trunc('year', firedate_key) as timestamp) as year_key,
    landcover_dominant_key                              as landcover_key,
    source_key,
    sum(area_ha)                                        as burnt_area_ha
from {{ source('silver', 'fact_fireevent') }}
where source_key = 'EFFIS_RDA'
  and landcover_dominant_key is not null
group by 1, 2, 3, 4
