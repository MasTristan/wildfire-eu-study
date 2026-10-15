{% test no_source_mixing(model) %}
-- GWIS rows carry emissions only; every other source carries no emissions.
select *
from {{ model }}
where (source_key = 'GWIS_GFAS' and (burnt_area_ha is not null or number_of_fires is not null or co2_emissions_t is null))
   or (source_key <> 'GWIS_GFAS' and co2_emissions_t is not null)
{% endtest %}
