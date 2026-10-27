# Sharing business data without revealing who earns what

A shopping-mall operator wants an outside provider to model how its tenants behave:
seasonality, how sales respond to promotions, which segments grow. The provider needs
monthly sales, rent and floor area per store. It must not learn how much any one
tenant sells or pays.

Replacing the store name is not enough. In most malls the biggest fashion store is a
well-known anchor tenant. If the release shows a store with the highest revenue in
the "Fashion" segment, anyone can see which store it is, and its revenue is still
in reais. The name was hidden, but the number was not.

This guide shows the plan the library proposes for such a table, what each step
removes, and what still remains.

## What gives a tenant away

| Clue in the release | What an outsider learns |
|---|---|
| Revenue or rent in reais | The tenant's size, once the tenant is recognised |
| The segment label ("Fashion") | Which few stores the row can belong to |
| A segment with one dominant store | That store's figure, read straight from the segment total |
| Floor area in square metres | Anchor stores have unusual areas, so the area points to them |
| A small segment ("Jewellery", two stores) | Each store, and from it the other's figure |
| Exact dates | Opening dates or one-off events that identify a store |

## The default plan

`anonymize propose` recognises these columns by name, in Portuguese and English:

| Category | Name hints |
|---|---|
| Company | `razao_social`, `nome_fantasia`, `company_name`, `empresa`, `tenant`, `lojista`, `store_name`, `nome_loja`, `brand`, `marca`, `merchant` |
| Revenue | `revenue`, `sales`, `faturamento`, `vendas`, `receita`, `gmv`, `turnover` |
| Rent | `rent`, `aluguel`, `condominio`, `occupancy_cost`, `custo_ocupacao` |
| Floor area | `abl`, `gla`, `sqm`, `m2`, `metragem`, `area_m2`, `floor_area` |
| Segment | `segmento`, `segment`, `ramo`, `setor`, `store_category`, `categoria_loja` |

A name such as `sales_per_sqm` counts as revenue, not floor area. A bare `loja` is too
ambiguous to count, since it is often a store number; mark it in the plan by hand. Under
the default `anonymization` target the proposal is:

| Column | Technique | Result |
|---|---|---|
| Store name | `scramble_categories` (`ORG-…`) | An opaque code per store. All rows of one store keep the same code, so its behaviour over time can still be modelled |
| Segment | `scramble_categories` (`SEG-…`) | Opaque codes in random order. Segments with fewer than 3 rows (5 with `--fidelity privacy`) are merged into `SEG-OTHER` |
| Floor area | `bin_numeric`, quartiles | Size bands instead of square metres |
| Revenue, rent | `normalize`, `robust` | (value − median) / interquartile range. No reais are left; the ranking, the seasonality and the shape of the distribution are kept |
| Dates | `date_shift` by store | Every row of one store moves by the same number of days, so its monthly pattern stays intact |

The anchor tenant can still be the biggest store in the release. That is the point:
its relative size is part of the behaviour to model. What it no longer shows is how
much it sells.

## The magnitude check

After a plan runs, the protection check looks for any numeric sensitive column that
is still in its original units. A column counts as still in original units when its
spread and centre are within a factor of two of the original. For each such
column, it tests every discrete quasi-identifier with two rules that statistical
agencies use for business data:

- **Dominance**: one contributor holds more than half of a segment's total.
- **p% rule**: the second-largest contributor could estimate the largest one to
  within 10% by subtracting its own figure from the total.

A store with 24 monthly rows counts once, with its full total. The plan's
`entity_column` (the store column) says which rows belong together.

In a test mall of 40 stores over 24 months, hiding only the store name leaves the
release at risk:

```text
level: at_risk
- 960 rows are unique across their discrete columns (segmento, month, aluguel), so they
  can be singled out
- faturamento is released unchanged and 100% of its values are distinct, so it can
  single out rows
- faturamento keeps its original units, and in segmento = 'Moda' one contributor holds
  90% of the total (dominance); normalize it or merge segments
```

The default plan for the same table reaches `anonymized`. Its caveats include the
rescaling limit described below.

## What remains, and the stronger options

`robust`, `zscore` and `minmax` are shifts and scales. They keep the distribution's
shape, which is why they are the default. Their known limit: anyone who knows the
true value of **two** records can solve for the shift and scale and convert the
whole column back. A tenant knows its own figures, so two tenants together can
undo the rescaling. Each column gets its own shift and scale, so undoing revenue
does not undo rent. The protection report always lists this caveat.

To remove the risk entirely, rescale each store against itself. Every method that
does this changes the distribution's shape, so it needs `allow_shape_change`:

```json
{
  "column": "faturamento",
  "technique": "normalize",
  "params": {"method": "index", "by": "store_name", "allow_shape_change": true}
}
```

| Method | Each value becomes | Keeps | Loses |
|---|---|---|---|
| `index` with `by` | value / that store's mean × 100 | Each store's seasonality and spikes | Every store averages 100, so sizes disappear |
| `growth` with `by` and `order_by` | change from the store's previous period | Growth rates | Levels; the first period of each store is blank |
| `rank` | percentile rank, 0 to 1 | Order | Distances between stores |
| `log_ratio` | log(value / geometric mean) | Ratios between stores | Additive structure; values must be positive |

Choose `index` by store when the provider models behaviour over time. Keep the
default `robust` when the provider also needs to compare stores of different sizes.

## Checklist before sending the file

1. Run `anonymize apply` and read the protection level and problems.
2. Confirm no segment, area band or category has fewer than three stores.
3. Read the caveats. If two tenants could pool their own figures, use `index` by
   store.
4. Remove columns the provider does not need, such as contract numbers and
   contact names.
