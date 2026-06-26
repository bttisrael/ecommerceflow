# CommerceFlow AI — Data Quality Report


Rows: **100,000**  
Columns: **42**  
Date range: **2025-06-01 00:03:38+00:00 → 2025-06-19 06:52:27+00:00**  

Target: `delay_risk_label`  
Delay rate: **27.29%**


## Leakage Columns to Exclude
None detected.


## Column Audit
                       column               dtype  nunique  null_pct
                     order_id                 str   100000       0.0
              order_timestamp datetime64[us, UTC]    94548       0.0
             delay_risk_label               int64        2       0.0
                  order_value             float64    68308       0.0
              order_weight_kg             float64     1790       0.0
      product_fragility_score             float64        8       0.0
                  distance_km             float64    52396       0.0
      carrier_base_delay_rate             float64        5       0.0
       promised_delivery_days               Int64        5       0.0
      estimated_delivery_days               Int64       12       0.0
estimated_minus_promised_days               Int64        9       0.0
      estimated_over_promised             float64       18       0.0
              log_order_value             float64    68308       0.0
              log_distance_km             float64    52396       0.0
                log_weight_kg             float64     1790       0.0
                 value_per_kg             float64    98882       0.0
                 value_per_km             float64    99988       0.0
  weight_distance_interaction             float64    98086       0.0
                  order_month               Int64        1       0.0
            order_day_of_week               Int64        7       0.0
                   order_hour               Int64       24       0.0
                   is_weekend               Int64        2       0.0
                 is_peak_hour               Int64        2       0.0
                     hour_sin             float64       22       0.0
                     hour_cos             float64       22       0.0
                      dow_sin             float64        7       0.0
                      dow_cos             float64        7       0.0
                    month_sin             float64        1       0.0
                    month_cos             float64        1       0.0
                         city                 str       20       0.0
                        state                 str       16       0.0
                       region                 str        5       0.0
             product_category                 str        7       0.0
                      carrier                 str        5       0.0
                        dc_id                 str        4       0.0
                     dc_state                 str        4       0.0
            traffic_condition                 str        4       0.0
            weather_condition                 str        5       0.0
                   route_type                 str        4       0.0
                same_state_dc               Int64        2       0.0
                distance_band                 str        5       0.0
                  weight_band                 str        5       0.0


## Numeric Summary
                       column     count         mean          std        min        25%         50%          75%           max
             delay_risk_label  100000.0      0.27289     0.445447        0.0        0.0         0.0          1.0           1.0
                  order_value  100000.0   705.651959   683.487792      47.26   245.6275     443.615       899.12       3540.12
              order_weight_kg  100000.0     2.581169     3.467414       0.09       0.52        1.05          3.0         20.65
      product_fragility_score  100000.0     0.051621     0.026711       0.02       0.03        0.04         0.07          0.12
                  distance_km  100000.0   516.950724   592.745115        0.1    36.4375      357.85     677.0275       3018.99
      carrier_base_delay_rate  100000.0     0.087251     0.025063      0.055      0.075        0.09        0.105          0.13
       promised_delivery_days  100000.0      3.24745     1.976112        1.0        1.0         4.0          4.0           8.0
      estimated_delivery_days  100000.0      2.38819     1.637371        1.0        1.0         2.0          3.0          12.0
estimated_minus_promised_days  100000.0     -0.85926      1.08175       -4.0       -2.0        -1.0          0.0           4.0
      estimated_over_promised  100000.0     0.827079     0.409397       0.25        0.5        0.75          1.0           4.0
              log_order_value  100000.0     6.143755     0.927594   3.876603   5.507879    6.097209     6.802528      8.172198
              log_distance_km  100000.0     5.322998     1.691445    0.09531   3.622673    5.882904     6.519188      8.013009
                log_weight_kg  100000.0      0.97352     0.707554   0.086178    0.41871     0.71784     1.386294      3.075005
                 value_per_kg  100000.0   1073.59167  1717.792438  29.821782  90.820441  345.425283  1119.427419  21231.272727
                 value_per_km  100000.0    17.855528    91.413312   0.020229   0.540342    1.607349      9.02785  18587.384615
  weight_distance_interaction  100000.0  1343.510754  3163.611476     0.0598   69.77625    317.0956  1151.388625    49079.5317
                  order_month  100000.0          6.0          0.0        6.0        6.0         6.0          6.0           6.0
            order_day_of_week  100000.0      2.65157     2.026398        0.0        1.0         3.0          4.0           6.0
                   order_hour  100000.0     14.81858     5.049335        0.0       11.0        15.0         19.0          23.0
                   is_weekend  100000.0      0.20404        0.403        0.0        0.0         0.0          0.0           1.0
                 is_peak_hour  100000.0      0.60458     0.488943        0.0        0.0         1.0          1.0           1.0
                     hour_sin  100000.0    -0.242696     0.660092       -1.0  -0.866025        -0.5     0.258819           1.0
                     hour_cos  100000.0    -0.235419     0.670794       -1.0  -0.866025        -0.5          0.5           1.0
                      dow_sin  100000.0     0.057273     0.644676  -0.974928  -0.433884         0.0     0.781831      0.974928
                      dow_cos  100000.0     0.053964     0.760402  -0.900969  -0.900969   -0.222521      0.62349           1.0
                    month_sin  100000.0          0.0          0.0        0.0        0.0         0.0          0.0           0.0
                    month_cos  100000.0         -1.0          0.0       -1.0       -1.0        -1.0         -1.0          -1.0
                same_state_dc  100000.0      0.34739     0.476143        0.0        0.0         0.0          1.0           1.0


## Business Reading
This dataset is suitable for a supervised delay-risk classification model. The model must use features known before delivery completion and must not use actual delivery outcome columns.