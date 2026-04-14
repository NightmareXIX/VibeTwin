# Preprocessing Audit

## Totals
| metric | count |
| --- | --- |
| healthy train windows | 1143 |
| healthy val windows | 241 |
| healthy test windows | 241 |
| fault test windows | 1412 |

## Per-file Audit
| filename | condition | class | load_hp | signal_length | signal_key | rpm | train | val | test | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| normal_0.mat | healthy | normal | 0 | 243938 | X097_DE_time | present (1796) | 162 | 34 | 34 | 230 |
| normal_1.mat | healthy | normal | 1 | 483903 | X098_DE_time | missing | 326 | 69 | 69 | 464 |
| normal_2.mat | healthy | normal | 2 | 485063 | X099_DE_time | missing | 327 | 69 | 69 | 465 |
| normal_3.mat | healthy | normal | 3 | 485643 | X100_DE_time | present (1725) | 328 | 69 | 69 | 466 |
| ir007_0.mat | fault | inner_race | 0 | 121265 | X105_DE_time | present (1797) | 0 | 0 | 117 | 117 |
| ir007_1.mat | fault | inner_race | 1 | 121991 | X106_DE_time | present (1772) | 0 | 0 | 118 | 118 |
| ir007_2.mat | fault | inner_race | 2 | 122136 | X107_DE_time | present (1748) | 0 | 0 | 118 | 118 |
| ir007_3.mat | fault | inner_race | 3 | 122917 | X108_DE_time | present (1721) | 0 | 0 | 119 | 119 |
| ball007_0.mat | fault | ball | 0 | 122571 | X118_DE_time | present (1796) | 0 | 0 | 118 | 118 |
| ball007_1.mat | fault | ball | 1 | 121410 | X119_DE_time | present (1772) | 0 | 0 | 117 | 117 |
| ball007_2.mat | fault | ball | 2 | 121556 | X120_DE_time | present (1748) | 0 | 0 | 117 | 117 |
| ball007_3.mat | fault | ball | 3 | 121556 | X121_DE_time | present (1722) | 0 | 0 | 117 | 117 |
| or007_6_0.mat | fault | outer_race_6 | 0 | 121991 | X130_DE_time | present (1796) | 0 | 0 | 118 | 118 |
| or007_6_1.mat | fault | outer_race_6 | 1 | 122426 | X131_DE_time | present (1773) | 0 | 0 | 118 | 118 |
| or007_6_2.mat | fault | outer_race_6 | 2 | 121410 | X132_DE_time | present (1750) | 0 | 0 | 117 | 117 |
| or007_6_3.mat | fault | outer_race_6 | 3 | 122571 | X133_DE_time | present (1725) | 0 | 0 | 118 | 118 |

## Counts by Class
| class | train | val | test | total |
| --- | --- | --- | --- | --- |
| normal | 1143 | 241 | 241 | 1625 |
| ball | 0 | 0 | 469 | 469 |
| inner_race | 0 | 0 | 472 | 472 |
| outer_race_6 | 0 | 0 | 471 | 471 |

## Counts by Load HP
| load_hp | train | val | test | total |
| --- | --- | --- | --- | --- |
| 0 | 162 | 34 | 387 | 583 |
| 1 | 326 | 69 | 422 | 817 |
| 2 | 327 | 69 | 421 | 817 |
| 3 | 328 | 69 | 423 | 820 |

## Healthy Split Check
- Confirmed no cross-split healthy window overlap for 4 healthy files.
- Healthy splits were generated from contiguous regions with a 2048-sample guard gap.

## Fault Label Map
- 0=ball, 1=inner_race, 2=outer_race_6
