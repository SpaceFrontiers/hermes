# Paired throughput results

QPS, higher is better. Median of six repetitions across two rounds;
CPU change is server CPU time per completed request. See
[protocol and limitations](README.md) and [all samples](results.json).

## PLAIN

| Family        | Operation |  Before |   After | QPS change | CPU/request change |
| ------------- | --------- | ------: | ------: | ---------: | -----------------: |
| and_high_high | COUNT     |   2,290 |   2,289 |      -0.1% |              -0.1% |
| and_high_high | TOP_10    |   1,162 |   1,322 |     +13.7% |             -12.7% |
| and_high_high | TOP_100   |   1,153 |   1,155 |      +0.2% |              -0.6% |
| and_high_low  | COUNT     |  13,907 |  13,698 |      -1.5% |              -0.9% |
| and_high_low  | TOP_10    |   3,157 |   3,049 |      -3.4% |              +3.5% |
| and_high_low  | TOP_100   |   3,117 |   3,050 |      -2.1% |              +2.6% |
| and_high_med  | COUNT     |   4,716 |   4,710 |      -0.1% |              +0.1% |
| and_high_med  | TOP_10    |   2,908 |   2,989 |      +2.8% |              -3.3% |
| and_high_med  | TOP_100   |   2,867 |   2,719 |      -5.1% |              +5.3% |
| high_term     | COUNT     | 111,834 | 111,369 |      -0.4% |              +0.5% |
| high_term     | TOP_10    |     844 |   5,681 |    +573.2% |             -86.0% |
| high_term     | TOP_100   |     842 |   3,183 |    +278.2% |             -74.6% |
| low_term      | COUNT     | 111,293 | 110,716 |      -0.5% |              +0.8% |
| low_term      | TOP_10    |   8,380 |  35,076 |    +318.6% |             -77.4% |
| low_term      | TOP_100   |   8,047 |  21,742 |    +170.2% |             -64.3% |
| med_phrase    | COUNT     |     174 |     174 |      +0.2% |              -0.3% |
| med_phrase    | TOP_10    |   2,014 |   2,022 |      +0.4% |              -0.5% |
| med_phrase    | TOP_100   |     758 |     759 |      +0.2% |              -0.2% |
| med_term      | COUNT     | 112,628 | 111,407 |      -1.1% |              +0.5% |
| med_term      | TOP_10    |   2,569 |  21,176 |    +724.2% |             -89.1% |
| med_term      | TOP_100   |   2,540 |  10,760 |    +323.7% |             -78.2% |

## RGB

| Family        | Operation |  Before |   After | QPS change | CPU/request change |
| ------------- | --------- | ------: | ------: | ---------: | -----------------: |
| and_high_high | COUNT     |   2,334 |   2,328 |      -0.3% |              +0.2% |
| and_high_high | TOP_10    |   1,230 |   2,391 |     +94.3% |             -49.3% |
| and_high_high | TOP_100   |   1,220 |   1,609 |     +31.8% |             -24.5% |
| and_high_low  | COUNT     |  18,061 |  17,895 |      -0.9% |              -0.1% |
| and_high_low  | TOP_10    |   3,938 |   3,846 |      -2.4% |              +2.1% |
| and_high_low  | TOP_100   |   3,890 |   3,814 |      -1.9% |              +1.7% |
| and_high_med  | COUNT     |   5,668 |   5,637 |      -0.5% |              -0.1% |
| and_high_med  | TOP_10    |   3,696 |   4,651 |     +25.8% |             -22.5% |
| and_high_med  | TOP_100   |   3,598 |   3,762 |      +4.6% |              -4.9% |
| high_term     | COUNT     | 113,024 | 111,428 |      -1.4% |              +1.4% |
| high_term     | TOP_10    |  10,014 |   9,829 |      -1.8% |              -0.2% |
| high_term     | TOP_100   |   6,105 |   6,052 |      -0.9% |              +0.1% |
| low_term      | COUNT     | 112,255 | 111,283 |      -0.9% |              +1.1% |
| low_term      | TOP_10    |  47,087 |  46,501 |      -1.2% |              +0.4% |
| low_term      | TOP_100   |  28,028 |  27,855 |      -0.6% |              -0.0% |
| med_phrase    | COUNT     |     193 |     195 |      +1.2% |              -1.3% |
| med_phrase    | TOP_10    |   1,951 |   1,941 |      -0.5% |              +0.5% |
| med_phrase    | TOP_100   |     755 |     749 |      -0.7% |              +0.1% |
| med_term      | COUNT     | 113,683 | 112,617 |      -0.9% |              +1.3% |
| med_term      | TOP_10    |  25,288 |  25,010 |      -1.1% |              +0.6% |
| med_term      | TOP_100   |  14,361 |  14,224 |      -1.0% |              +0.8% |
