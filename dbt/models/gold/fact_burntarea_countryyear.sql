-- Country x year x source. Burnt area and fire counts roll up from
-- fact_fireevent; GWIS/GFAS emissions get their OWN rows (source_key
-- GWIS_GFAS) with burnt_area_ha and number_of_fires null: GFAS measures the
-- fire independently of EFFIS, so the two never share a record.
-- Mirrors src/transform/rollups.py::build_countryyear.
with fires as (
    select
        country_key,
        cast(date_trunc('year', firedate_key) as timestamp) as year_key,
        source_key,
        sum(area_ha)                                       as burnt_area_ha,
        cast(count(fire_event_key) as double)              as number_of_fires,
        cast(null as double)                               as co2_emissions_t
    from {{ source('silver', 'fact_fireevent') }}
    group by 1, 2, 3
),

emissions as (
    select
        country_key,
        cast(date_trunc('year', cast(month_key as timestamp)) as timestamp) as year_key,
        'GWIS_GFAS'                                        as source_key,
        cast(null as double)                               as burnt_area_ha,
        cast(null as double)                               as number_of_fires,
        sum(co2_t)                                         as co2_emissions_t
    from {{ source('interim', 'gwis_gfas_emissions') }}
    where year(cast(month_key as timestamp)) between {{ var('base_year_min') }} and {{ var('base_year_max') }}
    group by 1, 2, 3
)

select * from fires
union all
select * from emissions
