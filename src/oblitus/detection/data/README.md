# Reference lists

- `municipalities.txt`: the names of the Brazilian municipalities, from the IBGE localities API
  (servicodados.ibge.gov.br, `/api/v1/localidades/municipios`).
- `first_names.txt`: first names from the IBGE 2010 Census names ranking
  (`/api/v2/censos/nomes/ranking`), collected for every state, every decade and the municipalities
  returned by the API (20 names per query).
- `surnames.txt`: common Brazilian surnames from the Faker `pt_BR` provider (MIT licence), because
  IBGE publishes no surname ranking.

IBGE data is public data published by the Brazilian government. Rebuild the files with
`uv run python benchmarks/build_reference_lists.py`.
