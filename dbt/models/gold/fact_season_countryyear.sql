-- Fire season per country, year and source, from the fire events themselves:
-- onset = first fire date at which the cumulative burnt area of the year
-- reaches 5 percent of the year's total, end = 95 percent, length in days.
-- No external threshold, no calendar assumption: a year with one fire has a
-- one-day season. Years below 10 fires are kept but flagged (low_count).
with events as (
    select
        country_key,
        source_key,
        cast(date_trunc('year', firedate_key) as timestamp) as year_key,
        cast(firedate_key as date)                          as fire_date,
        area_ha
    from {{ source('silver', 'fact_fireevent') }}
    where area_ha is not null and firedate_key is not null
),

daily as (
    select country_key, source_key, year_key, fire_date,
           sum(area_ha) as area_ha, count(*) as n_fires
    from events
    group by 1, 2, 3, 4
),

cumul as (
    select *,
           sum(area_ha)  over (partition by country_key, source_key, year_key order by fire_date) as cum_area,
           sum(area_ha)  over (partition by country_key, source_key, year_key)                    as total_area,
           sum(n_fires)  over (partition by country_key, source_key, year_key)                    as total_fires
    from daily
),

marks as (
    select country_key, source_key, year_key, total_area, total_fires,
           min(case when cum_area >= 0.05 * total_area then fire_date end) as onset_date,
           min(case when cum_area >= 0.95 * total_area then fire_date end) as end_date
    from cumul
    group by 1, 2, 3, 4, 5
)

select
    country_key,
    year_key,
    source_key,
    total_area                                           as burnt_area_ha,
    total_fires                                          as number_of_fires,
    cast(onset_date as timestamp)                        as onset_date,
    cast(end_date as timestamp)                          as end_date,
    dayofyear(onset_date)                                as onset_doy,
    dayofyear(end_date)                                  as end_doy,
    date_diff('day', onset_date, end_date) + 1           as season_length_days,
    total_fires < 10                                     as low_count
from marks
