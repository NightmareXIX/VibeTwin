# Preprocessing Audit

## Totals
| metric | count |
| --- | --- |
| healthy train windows | 1143 |
| healthy val windows | 241 |
| healthy test windows | 241 |
| fault test windows | 4977 |

## Per-file Audit
| filename | condition | class | load_hp | signal_length | signal_key | rpm | train | val | test | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| normal_0.mat | healthy | normal | 0 | 243938 | X097_DE_time | present (1796) | 162 | 34 | 34 | 230 |
| normal_1.mat | healthy | normal | 1 | 483903 | X098_DE_time | missing | 326 | 69 | 69 | 464 |
| normal_2.mat | healthy | normal | 2 | 485063 | X099_DE_time | missing | 327 | 69 | 69 | 465 |
| normal_3.mat | healthy | normal | 3 | 485643 | X100_DE_time | present (1725) | 328 | 69 | 69 | 466 |
| ir007_0.mat | fault | inner_race | 0 | 243938 | X109_DE_time | present (1796) | 0 | 0 | 237 | 237 |
| ir007_1.mat | fault | inner_race | 1 | 486224 | X110_DE_time | present (1772) | 0 | 0 | 473 | 473 |
| ir007_2.mat | fault | inner_race | 2 | 485643 | X111_DE_time | present (1748) | 0 | 0 | 473 | 473 |
| ir007_3.mat | fault | inner_race | 3 | 485643 | X112_DE_time | present (1721) | 0 | 0 | 473 | 473 |
| ball007_0.mat | fault | ball | 0 | 244739 | X122_DE_time | present (1796) | 0 | 0 | 238 | 238 |
| ball007_1.mat | fault | ball | 1 | 487384 | X123_DE_time | present (1772) | 0 | 0 | 474 | 474 |
| ball007_2.mat | fault | ball | 2 | 486804 | X124_DE_time | present (1747) | 0 | 0 | 474 | 474 |
| ball007_3.mat | fault | ball | 3 | 488545 | X125_DE_time | present (1721) | 0 | 0 | 476 | 476 |
| or007_6_0.mat | fault | outer_race_6 | 0 | 243538 | X135_DE_time | present (1796) | 0 | 0 | 236 | 236 |
| or007_6_1.mat | fault | outer_race_6 | 1 | 486804 | X136_DE_time | present (1774) | 0 | 0 | 474 | 474 |
| or007_6_2.mat | fault | outer_race_6 | 2 | 486804 | X137_DE_time | present (1750) | 0 | 0 | 474 | 474 |
| or007_6_3.mat | fault | outer_race_6 | 3 | 487964 | X138_DE_time | present (1725) | 0 | 0 | 475 | 475 |

## Counts by Class
| class | train | val | test | total |
| --- | --- | --- | --- | --- |
| normal | 1143 | 241 | 241 | 1625 |
| ball | 0 | 0 | 1662 | 1662 |
| inner_race | 0 | 0 | 1656 | 1656 |
| outer_race_6 | 0 | 0 | 1659 | 1659 |

## Counts by Load HP
| load_hp | train | val | test | total |
| --- | --- | --- | --- | --- |
| 0 | 162 | 34 | 745 | 941 |
| 1 | 326 | 69 | 1490 | 1885 |
| 2 | 327 | 69 | 1490 | 1886 |
| 3 | 328 | 69 | 1493 | 1890 |

## Healthy Split Check
- Confirmed no cross-split healthy window overlap for 4 healthy files.
- Healthy splits were generated from contiguous regions with a 2048-sample guard gap.

## Fault Label Map
- 0=ball, 1=inner_race, 2=outer_race_6
