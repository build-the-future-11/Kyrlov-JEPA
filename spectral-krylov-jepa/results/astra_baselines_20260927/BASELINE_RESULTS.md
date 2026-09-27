# Baseline audit — historical smoke only

| Method | Labels | Split | Fidelity | Relative energy error | Residual at true E |
|---|---|---|---:|---:|---:|
| free_box | no_labels | test_ID | 0.994460 | 0.012983 | 1.483805 |
| sine_ritz_9 | no_labels | test_ID | 0.999898 | 0.001092 | 0.761666 |
| train_mean | n10 | test_ID | 0.993491 | 0.091419 | 1.537090 |
| linear_ridge | n10 | test_ID | 0.999525 | 0.011010 | 0.876381 |
| train_mean | n20 | test_ID | 0.994432 | 0.088873 | 1.469374 |
| linear_ridge | n20 | test_ID | 0.999904 | 0.012829 | 0.516766 |
| free_box | no_labels | test_OOD_double | 0.997242 | 0.006501 | 1.284092 |
| sine_ritz_9 | no_labels | test_OOD_double | 0.999935 | 0.000679 | 0.718784 |
| train_mean | n10 | test_OOD_double | 0.996469 | 0.051605 | 1.395321 |
| linear_ridge | n10 | test_OOD_double | 0.999385 | 0.007714 | 0.945318 |
| train_mean | n20 | test_OOD_double | 0.996667 | 0.066650 | 1.380113 |
| linear_ridge | n20 | test_OOD_double | 0.999881 | 0.010278 | 0.603358 |
| free_box | no_labels | test_OOD_narrow | 0.999739 | 0.000860 | 0.749944 |
| sine_ritz_9 | no_labels | test_OOD_narrow | 0.999973 | 0.000336 | 0.669640 |
| train_mean | n10 | test_OOD_narrow | 0.999269 | 0.113510 | 0.887413 |
| linear_ridge | n10 | test_OOD_narrow | 0.999904 | 0.005337 | 0.725391 |
| train_mean | n20 | test_OOD_narrow | 0.999361 | 0.136299 | 0.864499 |
| linear_ridge | n20 | test_OOD_narrow | 0.999964 | 0.005771 | 0.595600 |
| free_box | no_labels | test_OOD_strong | 0.964871 | 0.310383 | 4.266197 |
| sine_ritz_9 | no_labels | test_OOD_strong | 0.998979 | 0.032930 | 2.527482 |
| train_mean | n10 | test_OOD_strong | 0.965296 | 1.673079 | 4.172788 |
| linear_ridge | n10 | test_OOD_strong | 0.996309 | 0.257729 | 2.485790 |
| train_mean | n20 | test_OOD_strong | 0.963903 | 1.610789 | 4.227920 |
| linear_ridge | n20 | test_OOD_strong | 0.998982 | 0.234182 | 1.501539 |
