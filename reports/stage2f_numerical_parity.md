# Stage 2F numerical-parity investigation

Generated 2026-10-02T06:00:54.066274+00:00. Evidence: `runs/stage2f_parity`.

Original strict verdict: **FAIL**. Scientific assessment: **likely_benign**.
GPU numerical recommendation: **GO on the tested numerical evidence**. Production launch: **GO on numerical grounds; launch deferred until full-cache and production-trainer readiness are verified**.
Full-cache generation recommendation: **GO (recommendation only; not generated)**.

No production training, full-cache generation, VAL2 access or source-image changes occurred. The strict evaluator, original tolerance utility, cache representation, model, split and scientific training configuration are protected by SHA-256.

## Exact reproduction

`{"exact_tensor_files": {"cpu": {"gradients.npy": true, "weights_1.npy": true, "weights_2.npy": true, "weights_3.npy": true}, "gpu": {"gradients.npy": true, "weights_1.npy": true, "weights_2.npy": true, "weights_3.npy": true}}, "original_failures": [0, 7, 9, 10, 17, 19, 20, 22], "reproduced_failures": [0, 7, 9, 10, 17, 19, 20, 22], "same_failures": true, "strict_pass": false}`

The unchanged Stage 2E parity worker was rerun with byte-verified copies of its initialization and 1000-image cache. It uses identical first-100 inputs/labels, physical 4/effective 100, Adam, frozen BN, float32, TF32 disabled, deterministic settings and the original comparison utility. Three repeated updates match the original experiment. The diagnostic trajectory then uses ten distinct logical batches: the original first 100, followed by a fixed SHA-256 ordering of the remaining 900. No augmentation is used.

## Complete gradient table: identity and norms

| ID | Variable | Shape | Elements | CPU norm | GPU norm | Strict verdict |
|---:|---|---|---:|---:|---:|---|
| 0 | normal_A_block_17/normal_conv_1_17/kernel | [1, 1, 4032, 672] | 2709504 | 1.37446062 | 1.37446405 | **FAIL** |
| 1 | normal_A_block_17/block_1/separable_conv_block_normal_left1_17/separable_conv_1_normal_left1_17/depthwise_kernel | [5, 5, 672, 1] | 16800 | 0.196732416 | 0.196768915 | PASS |
| 2 | normal_A_block_17/block_1/separable_conv_block_normal_left1_17/separable_conv_1_normal_left1_17/pointwise_kernel | [1, 1, 672, 672] | 451584 | 0.590182639 | 0.590171813 | PASS |
| 3 | normal_A_block_17/block_1/separable_conv_block_normal_right1_17/separable_conv_1_normal_right1_17/depthwise_kernel | [3, 3, 672, 1] | 6048 | 0.0706042367 | 0.070608913 | PASS |
| 4 | normal_A_block_17/block_1/separable_conv_block_normal_right1_17/separable_conv_1_normal_right1_17/pointwise_kernel | [1, 1, 672, 672] | 451584 | 0.286822259 | 0.286834669 | PASS |
| 5 | normal_A_block_17/block_2/separable_conv_block_normal_left2_17/separable_conv_1_normal_left2_17/depthwise_kernel | [5, 5, 672, 1] | 16800 | 0.0963864364 | 0.0963791714 | PASS |
| 6 | normal_A_block_17/block_2/separable_conv_block_normal_left2_17/separable_conv_1_normal_left2_17/pointwise_kernel | [1, 1, 672, 672] | 451584 | 0.426544352 | 0.426537347 | PASS |
| 7 | normal_A_block_17/block_2/separable_conv_block_normal_right2_17/separable_conv_1_normal_right2_17/depthwise_kernel | [3, 3, 672, 1] | 6048 | 0.0707637229 | 0.0707556495 | **FAIL** |
| 8 | normal_A_block_17/block_2/separable_conv_block_normal_right2_17/separable_conv_1_normal_right2_17/pointwise_kernel | [1, 1, 672, 672] | 451584 | 0.33015527 | 0.330147569 | PASS |
| 9 | normal_A_block_17/block_5/separable_conv_block_normal_left5_17/separable_conv_1_normal_left5_17/depthwise_kernel | [3, 3, 672, 1] | 6048 | 0.153557658 | 0.153584256 | **FAIL** |
| 10 | normal_A_block_17/block_5/separable_conv_block_normal_left5_17/separable_conv_1_normal_left5_17/pointwise_kernel | [1, 1, 672, 672] | 451584 | 0.460633916 | 0.460639457 | **FAIL** |
| 11 | normal_A_block_17/block_1/separable_conv_block_normal_left1_17/separable_conv_2_normal_left1_17/depthwise_kernel | [5, 5, 672, 1] | 16800 | 0.240242458 | 0.240225088 | PASS |
| 12 | normal_A_block_17/block_1/separable_conv_block_normal_left1_17/separable_conv_2_normal_left1_17/pointwise_kernel | [1, 1, 672, 672] | 451584 | 0.563477048 | 0.563481137 | PASS |
| 13 | normal_A_block_17/block_1/separable_conv_block_normal_right1_17/separable_conv_2_normal_right1_17/depthwise_kernel | [3, 3, 672, 1] | 6048 | 0.0668149177 | 0.0668102259 | PASS |
| 14 | normal_A_block_17/block_1/separable_conv_block_normal_right1_17/separable_conv_2_normal_right1_17/pointwise_kernel | [1, 1, 672, 672] | 451584 | 0.264337354 | 0.264335301 | PASS |
| 15 | normal_A_block_17/block_2/separable_conv_block_normal_left2_17/separable_conv_2_normal_left2_17/depthwise_kernel | [5, 5, 672, 1] | 16800 | 0.103867705 | 0.103856428 | PASS |
| 16 | normal_A_block_17/block_2/separable_conv_block_normal_left2_17/separable_conv_2_normal_left2_17/pointwise_kernel | [1, 1, 672, 672] | 451584 | 0.408996493 | 0.408999313 | PASS |
| 17 | normal_A_block_17/block_2/separable_conv_block_normal_right2_17/separable_conv_2_normal_right2_17/depthwise_kernel | [3, 3, 672, 1] | 6048 | 0.0564893842 | 0.0564849198 | **FAIL** |
| 18 | normal_A_block_17/block_2/separable_conv_block_normal_right2_17/separable_conv_2_normal_right2_17/pointwise_kernel | [1, 1, 672, 672] | 451584 | 0.313011755 | 0.313010204 | PASS |
| 19 | normal_A_block_17/block_5/separable_conv_block_normal_left5_17/separable_conv_2_normal_left5_17/depthwise_kernel | [3, 3, 672, 1] | 6048 | 0.13603916 | 0.136036042 | **FAIL** |
| 20 | normal_A_block_17/block_5/separable_conv_block_normal_left5_17/separable_conv_2_normal_left5_17/pointwise_kernel | [1, 1, 672, 672] | 451584 | 0.428001654 | 0.428000121 | **FAIL** |
| 21 | normal_A_block_18/adjust_block/adjust_projection_block_18/adjust_conv_projection_18/kernel | [1, 1, 4032, 672] | 2709504 | 2.89635845 | 2.89636499 | PASS |
| 22 | normal_A_block_18/normal_conv_1_18/kernel | [1, 1, 4032, 672] | 2709504 | 2.33702608 | 2.33696368 | **FAIL** |
| 23 | normal_A_block_18/block_1/separable_conv_block_normal_left1_18/separable_conv_1_normal_left1_18/depthwise_kernel | [5, 5, 672, 1] | 16800 | 0.307451764 | 0.307451637 | PASS |
| 24 | normal_A_block_18/block_1/separable_conv_block_normal_left1_18/separable_conv_1_normal_left1_18/pointwise_kernel | [1, 1, 672, 672] | 451584 | 0.847283181 | 0.847283667 | PASS |
| 25 | normal_A_block_18/block_1/separable_conv_block_normal_right1_18/separable_conv_1_normal_right1_18/depthwise_kernel | [3, 3, 672, 1] | 6048 | 0.191330157 | 0.191334743 | PASS |
| 26 | normal_A_block_18/block_1/separable_conv_block_normal_right1_18/separable_conv_1_normal_right1_18/pointwise_kernel | [1, 1, 672, 672] | 451584 | 0.718143158 | 0.718157688 | PASS |
| 27 | normal_A_block_18/block_2/separable_conv_block_normal_left2_18/separable_conv_1_normal_left2_18/depthwise_kernel | [5, 5, 672, 1] | 16800 | 0.291263313 | 0.291262889 | PASS |
| 28 | normal_A_block_18/block_2/separable_conv_block_normal_left2_18/separable_conv_1_normal_left2_18/pointwise_kernel | [1, 1, 672, 672] | 451584 | 0.934687578 | 0.934683708 | PASS |
| 29 | normal_A_block_18/block_2/separable_conv_block_normal_right2_18/separable_conv_1_normal_right2_18/depthwise_kernel | [3, 3, 672, 1] | 6048 | 0.216777512 | 0.21677602 | PASS |
| 30 | normal_A_block_18/block_2/separable_conv_block_normal_right2_18/separable_conv_1_normal_right2_18/pointwise_kernel | [1, 1, 672, 672] | 451584 | 0.787507137 | 0.787505575 | PASS |
| 31 | normal_A_block_18/block_5/separable_conv_block_normal_left5_18/separable_conv_1_normal_left5_18/depthwise_kernel | [3, 3, 672, 1] | 6048 | 0.154874354 | 0.154877655 | PASS |
| 32 | normal_A_block_18/block_5/separable_conv_block_normal_left5_18/separable_conv_1_normal_left5_18/pointwise_kernel | [1, 1, 672, 672] | 451584 | 0.406331735 | 0.406341242 | PASS |
| 33 | normal_A_block_18/block_1/separable_conv_block_normal_left1_18/separable_conv_2_normal_left1_18/depthwise_kernel | [5, 5, 672, 1] | 16800 | 0.237148539 | 0.237148595 | PASS |
| 34 | normal_A_block_18/block_1/separable_conv_block_normal_left1_18/separable_conv_2_normal_left1_18/pointwise_kernel | [1, 1, 672, 672] | 451584 | 1.04212572 | 1.04212588 | PASS |
| 35 | normal_A_block_18/block_1/separable_conv_block_normal_right1_18/separable_conv_2_normal_right1_18/depthwise_kernel | [3, 3, 672, 1] | 6048 | 0.138332912 | 0.138332938 | PASS |
| 36 | normal_A_block_18/block_1/separable_conv_block_normal_right1_18/separable_conv_2_normal_right1_18/pointwise_kernel | [1, 1, 672, 672] | 451584 | 0.91711007 | 0.91711014 | PASS |
| 37 | normal_A_block_18/block_2/separable_conv_block_normal_left2_18/separable_conv_2_normal_left2_18/depthwise_kernel | [5, 5, 672, 1] | 16800 | 0.275503311 | 0.275499581 | PASS |
| 38 | normal_A_block_18/block_2/separable_conv_block_normal_left2_18/separable_conv_2_normal_left2_18/pointwise_kernel | [1, 1, 672, 672] | 451584 | 1.02348193 | 1.02348519 | PASS |
| 39 | normal_A_block_18/block_2/separable_conv_block_normal_right2_18/separable_conv_2_normal_right2_18/depthwise_kernel | [3, 3, 672, 1] | 6048 | 0.299953583 | 0.299946144 | PASS |
| 40 | normal_A_block_18/block_2/separable_conv_block_normal_right2_18/separable_conv_2_normal_right2_18/pointwise_kernel | [1, 1, 672, 672] | 451584 | 0.903041656 | 0.903044834 | PASS |
| 41 | normal_A_block_18/block_5/separable_conv_block_normal_left5_18/separable_conv_2_normal_left5_18/depthwise_kernel | [3, 3, 672, 1] | 6048 | 0.19429042 | 0.194315516 | PASS |
| 42 | normal_A_block_18/block_5/separable_conv_block_normal_left5_18/separable_conv_2_normal_left5_18/pointwise_kernel | [1, 1, 672, 672] | 451584 | 0.519956639 | 0.519944071 | PASS |
| 43 | paper_dense_256/kernel | [487872, 256] | 124895232 | 55.7146741 | 55.7146866 | PASS |
| 44 | paper_dense_256/bias | [256] | 256 | 0.105917856 | 0.105917892 | PASS |
| 45 | class_probabilities/kernel | [256, 2] | 512 | 1.26955503 | 1.26955528 | PASS |
| 46 | class_probabilities/bias | [2] | 2 | 0.0895902124 | 0.0895902704 | PASS |

### Complete gradient table: errors

| ID | Max abs | Mean abs | RMS | Max relative | Mean relative | Relative L2 | Cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0001572998008 | 1.018022886e-07 | 1.061355523e-06 | 57.8041687 | 0.001345538449 | 0.001271081741 | 0.9999991922 |
| 1 | 1.861015335e-05 | 3.623424886e-07 | 1.13674011e-06 | 3.638149118 | 0.003403665594 | 0.0007489277123 | 0.9999997368 |
| 2 | 4.357192665e-05 | 1.294284612e-07 | 5.086534541e-07 | 66.96174273 | 0.002660930413 | 0.0005791683772 | 0.9999998324 |
| 3 | 9.068986401e-06 | 1.58773446e-07 | 5.059509718e-07 | 16.47603973 | 0.00504554112 | 0.0005572929682 | 0.9999998469 |
| 4 | 5.494442303e-05 | 5.272023858e-08 | 3.314675712e-07 | 54.90835119 | 0.002127739154 | 0.0007766001435 | 0.9999996994 |
| 5 | 1.450011041e-05 | 1.282951094e-07 | 5.687081288e-07 | 1.550674634 | 0.001155166212 | 0.0007647652623 | 0.9999997104 |
| 6 | 6.457138807e-05 | 6.781929668e-08 | 4.972521158e-07 | 37.3311341 | 0.001443405744 | 0.0007833966634 | 0.9999996933 |
| 7 | 2.712756395e-05 | 1.837789994e-07 | 9.628200759e-07 | 1.095476878 | 0.00229630326 | 0.001058133235 | 0.9999994466 |
| 8 | 3.913789988e-05 | 6.741289699e-08 | 4.507518129e-07 | 30.6790468 | 0.001561589418 | 0.0009174629205 | 0.9999995794 |
| 9 | 7.043220103e-05 | 6.153180743e-07 | 2.489238899e-06 | 2.478821035 | 0.003695412706 | 0.00126066876 | 0.9999992205 |
| 10 | 5.999486893e-05 | 1.549631575e-07 | 8.50490883e-07 | 70.97330013 | 0.003389123179 | 0.001240746402 | 0.9999992304 |
| 11 | 2.857483923e-05 | 3.254659956e-07 | 1.001533126e-06 | 1.237629051 | 0.00164770123 | 0.0005403438298 | 0.9999998566 |
| 12 | 9.476952255e-05 | 7.952286014e-08 | 4.898507114e-07 | 117.8638727 | 0.001666046201 | 0.0005841935876 | 0.9999998294 |
| 13 | 7.312512025e-06 | 1.857794166e-07 | 5.359862498e-07 | 0.4975357598 | 0.001572820027 | 0.0006238585079 | 0.9999998079 |
| 14 | 1.574587077e-05 | 4.975031269e-08 | 3.458297962e-07 | 54.71574097 | 0.001589934172 | 0.000879170572 | 0.9999996136 |
| 15 | 1.978105865e-05 | 1.554792561e-07 | 6.999945291e-07 | 1.680166775 | 0.001180842617 | 0.0008735117475 | 0.9999996243 |
| 16 | 4.789792001e-05 | 4.180866191e-08 | 4.89450347e-07 | 17.73071668 | 0.0004356026715 | 0.0008041893723 | 0.9999996767 |
| 17 | 1.02603808e-05 | 2.38688933e-07 | 7.362594623e-07 | 0.7463777284 | 0.001908259707 | 0.001013607791 | 0.9999994894 |
| 18 | 2.81179673e-05 | 4.860204237e-08 | 4.585251684e-07 | 42.18434375 | 0.0007212924868 | 0.0009844004525 | 0.9999995155 |
| 19 | 4.033157893e-05 | 6.7652601e-07 | 2.025340226e-06 | 1.689434727 | 0.002967453414 | 0.001157817041 | 0.99999933 |
| 20 | 3.515416756e-05 | 1.583182902e-07 | 8.889826919e-07 | 1274.559014 | 0.005583115984 | 0.001395780513 | 0.9999990259 |
| 21 | 8.27293843e-06 | 1.466549247e-08 | 8.089246421e-08 | 7.033348304 | 9.286574538e-05 | 4.597276822e-05 | 0.9999999989 |
| 22 | 0.0005150362849 | 2.857610476e-08 | 1.199503196e-06 | 185.2762863 | 0.0002923207635 | 0.0008448561083 | 0.9999996435 |
| 23 | 5.874753697e-06 | 1.484318472e-08 | 1.04687686e-07 | 0.6610228366 | 0.0001097696866 | 4.41339961e-05 | 0.999999999 |
| 24 | 2.440426033e-05 | 2.022884424e-09 | 6.868350114e-08 | 0.3842681826 | 1.787114323e-05 | 5.447448242e-05 | 0.9999999985 |
| 25 | 1.021428034e-05 | 2.389327159e-07 | 8.214746436e-07 | 0.1061987866 | 0.0004375851791 | 0.0003339001593 | 0.9999999445 |
| 26 | 5.269492976e-05 | 9.328874521e-09 | 4.117160822e-07 | 2.722736036 | 3.800081139e-05 | 0.0003852619137 | 0.999999926 |
| 27 | 2.818414941e-06 | 7.991393653e-08 | 1.894189503e-07 | 0.9422961361 | 0.0003834185661 | 8.429314952e-05 | 0.9999999964 |
| 28 | 2.959743142e-06 | 3.292556934e-08 | 1.016925661e-07 | 11.63174319 | 0.0003659704701 | 7.311256296e-05 | 0.9999999973 |
| 29 | 2.159737051e-06 | 1.208121685e-07 | 2.607066882e-07 | 0.6687228787 | 0.0005226993507 | 9.352847147e-05 | 0.9999999956 |
| 30 | 4.207453458e-06 | 3.480460677e-08 | 1.181654471e-07 | 9.473167743 | 0.0003604249194 | 0.000100833601 | 0.9999999949 |
| 31 | 1.282349695e-05 | 6.532817648e-07 | 1.327408389e-06 | 6.467967228 | 0.003954978033 | 0.0006665472494 | 0.9999997781 |
| 32 | 1.640594564e-05 | 1.317388827e-07 | 4.776134311e-07 | 74.61509703 | 0.002829896187 | 0.0007898871732 | 0.9999996883 |
| 33 | 3.213062882e-08 | 1.386951159e-09 | 2.019514354e-09 | 0.01420395852 | 9.071477631e-06 | 1.103776471e-06 | 1 |
| 34 | 4.097819328e-08 | 9.029488806e-10 | 1.532934855e-09 | 0.4768371253 | 1.114915996e-05 | 9.88491311e-07 | 1 |
| 35 | 2.793967724e-08 | 1.593230735e-09 | 2.337900014e-09 | 0.007347487905 | 7.689175075e-06 | 1.314335701e-06 | 1 |
| 36 | 3.911554813e-08 | 7.917982123e-10 | 1.31372162e-09 | 0.3296881879 | 1.006659172e-05 | 9.626117494e-07 | 1 |
| 37 | 3.043562174e-06 | 5.613200757e-08 | 1.496614938e-07 | 0.3661087033 | 0.0003035535099 | 7.041057557e-05 | 0.9999999976 |
| 38 | 1.740921289e-05 | 3.091700408e-09 | 1.131880875e-07 | 0.3618162686 | 1.677102338e-05 | 7.431728192e-05 | 0.9999999972 |
| 39 | 3.233551979e-06 | 7.318360291e-08 | 2.172521447e-07 | 0.1686118325 | 0.0002008426145 | 5.632690778e-05 | 0.9999999987 |
| 40 | 1.918945054e-05 | 3.238574152e-09 | 1.236054591e-07 | 0.2596934889 | 1.510521551e-05 | 9.198121486e-05 | 0.9999999958 |
| 41 | 1.389719546e-05 | 3.718401652e-07 | 1.083364514e-06 | 1.318259736 | 0.002439781625 | 0.0004336397745 | 0.9999999143 |
| 42 | 0.0001058210619 | 1.335380492e-08 | 6.010942466e-07 | 39.35479233 | 0.0002222241461 | 0.0007768634988 | 0.9999996985 |
| 43 | 4.172325134e-07 | 2.701432092e-09 | 6.476398718e-09 | 2.27987762 | 1.480126806e-05 | 1.299083041e-06 | 1 |
| 44 | 1.490116119e-08 | 2.608135041e-09 | 4.009892382e-09 | 8.790153777e-05 | 2.527516407e-06 | 6.057361868e-07 | 1 |
| 45 | 2.682209015e-07 | 6.203427816e-08 | 7.999549451e-08 | 0.00248937292 | 1.844043054e-05 | 1.425768373e-06 | 1 |
| 46 | 4.470348358e-08 | 4.097819328e-08 | 4.114717624e-08 | 7.056605265e-07 | 6.46855448e-07 | 6.49522901e-07 | 1 |

### Complete gradient table: elementwise diagnostics

Absolute exceedance means |CPU-GPU| > 1e-4; relative means |CPU-GPU|/max(|CPU|,1e-8) > 1e-3. Either is the union. Near zero means magnitude <= 1e-8. These elementwise diagnostics do not replace the original tensor-level acceptance rules. Exact counts and fractions are all stored in JSON.

| ID | Absolute fraction | Relative fraction | Either fraction | CPU near-zero count/fraction | GPU near-zero count/fraction |
|---:|---:|---:|---:|---|---|
| 0 | 3.32164116e-06 | 0.065634522 | 0.065634522 | 171672/0.0633591978 | 171669/0.0633580906 |
| 1 | 0 | 0.246607143 | 0.246607143 | 0/0 | 1/5.95238095e-05 |
| 2 | 0 | 0.175318435 | 0.175318435 | 609/0.00134858631 | 611/0.00135301516 |
| 3 | 0 | 0.198082011 | 0.198082011 | 0/0 | 0/0 |
| 4 | 0 | 0.130843874 | 0.130843874 | 24944/0.055236678 | 24950/0.0552499646 |
| 5 | 0 | 0.129642857 | 0.129642857 | 0/0 | 0/0 |
| 6 | 0 | 0.0905789399 | 0.0905789399 | 1666/0.00368923611 | 1663/0.00368259283 |
| 7 | 0 | 0.147652116 | 0.147652116 | 2/0.000330687831 | 2/0.000330687831 |
| 8 | 0 | 0.10740416 | 0.10740416 | 19251/0.0426299426 | 19259/0.042647658 |
| 9 | 0 | 0.22718254 | 0.22718254 | 0/0 | 0/0 |
| 10 | 0 | 0.158524217 | 0.158524217 | 18073/0.0400213471 | 18077/0.0400302048 |
| 11 | 0 | 0.169761905 | 0.169761905 | 7/0.000416666667 | 7/0.000416666667 |
| 12 | 0 | 0.0827597966 | 0.0827597966 | 3658/0.00810037557 | 3655/0.00809373228 |
| 13 | 0 | 0.156580688 | 0.156580688 | 248/0.041005291 | 248/0.041005291 |
| 14 | 0 | 0.0768561331 | 0.0768561331 | 32300/0.0715260062 | 32300/0.0715260062 |
| 15 | 0 | 0.107619048 | 0.107619048 | 28/0.00166666667 | 28/0.00166666667 |
| 16 | 0 | 0.0272042411 | 0.0272042411 | 51402/0.113825999 | 51402/0.113825999 |
| 17 | 0 | 0.167824074 | 0.167824074 | 213/0.035218254 | 212/0.0350529101 |
| 18 | 0 | 0.0340202487 | 0.0340202487 | 64845/0.143594547 | 64849/0.143603405 |
| 19 | 0 | 0.212797619 | 0.212797619 | 217/0.0358796296 | 218/0.0360449735 |
| 20 | 0 | 0.0992329223 | 0.0992329223 | 18550/0.041077629 | 18549/0.0410754145 |
| 21 | 0 | 0.00818415474 | 0.00818415474 | 172264/0.063577688 | 172258/0.0635754736 |
| 22 | 2.7680343e-05 | 0.0139944433 | 0.0139944433 | 81186/0.0299634177 | 81181/0.0299615723 |
| 23 | 0 | 0.00738095238 | 0.00738095238 | 0/0 | 0/0 |
| 24 | 0 | 0.00168960813 | 0.00168960813 | 10/2.21442744e-05 | 10/2.21442744e-05 |
| 25 | 0 | 0.0633267196 | 0.0633267196 | 0/0 | 0/0 |
| 26 | 0 | 0.00206384637 | 0.00206384637 | 2700/0.00597895408 | 2698/0.00597452523 |
| 27 | 0 | 0.0389285714 | 0.0389285714 | 1/5.95238095e-05 | 1/5.95238095e-05 |
| 28 | 0 | 0.0289270656 | 0.0289270656 | 3383/0.00749140802 | 3379/0.00748255031 |
| 29 | 0 | 0.0439814815 | 0.0439814815 | 0/0 | 0/0 |
| 30 | 0 | 0.0300829967 | 0.0300829967 | 12896/0.0285572562 | 12898/0.0285616851 |
| 31 | 0 | 0.248181217 | 0.248181217 | 0/0 | 0/0 |
| 32 | 0 | 0.166799532 | 0.166799532 | 1156/0.00255987812 | 1161/0.00257095026 |
| 33 | 0 | 0.000773809524 | 0.000773809524 | 0/0 | 0/0 |
| 34 | 0 | 0.00089019983 | 0.00089019983 | 1373/0.00304040887 | 1373/0.00304040887 |
| 35 | 0 | 0.000496031746 | 0.000496031746 | 36/0.00595238095 | 36/0.00595238095 |
| 36 | 0 | 0.000861412273 | 0.000861412273 | 4038/0.00894185799 | 4038/0.00894185799 |
| 37 | 0 | 0.0317261905 | 0.0317261905 | 125/0.00744047619 | 125/0.00744047619 |
| 38 | 0 | 0.00134858631 | 0.00134858631 | 3430/0.00759548611 | 3429/0.00759327168 |
| 39 | 0 | 0.0309193122 | 0.0309193122 | 174/0.0287698413 | 174/0.0287698413 |
| 40 | 0 | 0.00130429776 | 0.00130429776 | 12658/0.0280302225 | 12658/0.0280302225 |
| 41 | 0 | 0.189814815 | 0.189814815 | 9/0.00148809524 | 9/0.00148809524 |
| 42 | 2.21442744e-06 | 0.00380881519 | 0.00380881519 | 736/0.00162981859 | 737/0.00163203302 |
| 43 | 0 | 0.00127224232 | 0.00127224232 | 36501036/0.292253238 | 36501055/0.29225339 |
| 44 | 0 | 0 | 0 | 0/0 | 0/0 |
| 45 | 0 | 0.00390625 | 0.00390625 | 0/0 | 0/0 |
| 46 | 0 | 0 | 0 | 0/0 | 0/0 |

## Failure pattern and cause analysis

Failing operation counts: `{"convolution": 2, "depthwise convolution": 4, "pointwise convolution": 2}`. Seven failures are in normal block 17 and one is in block 18: late backbone layers, with block 17 earliest in this trainable suffix. Dense kernels and biases pass; normalization parameters are frozen, so no trainable BN gradient is tested.

- **normal_A_block_17/normal_conv_1_17/kernel** (convolution): relative L2 bound; max error / CPU peak = 0.00264045; L2-relative error 0.00127108; cosine 0.9999991922; resulting first-Adam update-relative difference 0.0068761.
- **normal_A_block_17/block_2/separable_conv_block_normal_right2_17/separable_conv_1_normal_right2_17/depthwise_kernel** (depthwise convolution): relative L2 bound; max error / CPU peak = 0.0019725; L2-relative error 0.00105813; cosine 0.9999994466; resulting first-Adam update-relative difference 0.00866778.
- **normal_A_block_17/block_5/separable_conv_block_normal_left5_17/separable_conv_1_normal_left5_17/depthwise_kernel** (depthwise convolution): relative L2 bound; max error / CPU peak = 0.00269387; L2-relative error 0.00126067; cosine 0.9999992205; resulting first-Adam update-relative difference 0.00854931.
- **normal_A_block_17/block_5/separable_conv_block_normal_left5_17/separable_conv_1_normal_left5_17/pointwise_kernel** (pointwise convolution): relative L2 bound; max error / CPU peak = 0.00193753; L2-relative error 0.00124075; cosine 0.9999992304; resulting first-Adam update-relative difference 0.00631699.
- **normal_A_block_17/block_2/separable_conv_block_normal_right2_17/separable_conv_2_normal_right2_17/depthwise_kernel** (depthwise convolution): relative L2 bound; max error / CPU peak = 0.0015024; L2-relative error 0.00101361; cosine 0.9999994894; resulting first-Adam update-relative difference 0.00555611.
- **normal_A_block_17/block_5/separable_conv_block_normal_left5_17/separable_conv_2_normal_left5_17/depthwise_kernel** (depthwise convolution): relative L2 bound; max error / CPU peak = 0.00304762; L2-relative error 0.00115782; cosine 0.9999993300; resulting first-Adam update-relative difference 0.0046284.
- **normal_A_block_17/block_5/separable_conv_block_normal_left5_17/separable_conv_2_normal_left5_17/pointwise_kernel** (pointwise convolution): relative L2 bound; max error / CPU peak = 0.00182339; L2-relative error 0.00139578; cosine 0.9999990259; resulting first-Adam update-relative difference 0.00996893.
- **normal_A_block_18/normal_conv_1_18/kernel** (convolution): maximum absolute/tensor-scale bound; max error / CPU peak = 0.00432813; L2-relative error 0.000844856; cosine 0.9999996435; resulting first-Adam update-relative difference 0.00361009.

Failures are identified by the unchanged Stage 2E rule. A relative-L2 failure concerns the whole gradient norm and is not simply an unstable elementwise division near zero. The maximum-absolute failure is a localized discrepancy whose size, frequency and update effect are reported rather than assumed benign. Operation classes are recorded for all 47 variables; normalization parameters are frozen and excluded from this trainable set.

## Microbatch, accumulation and optimizer

Single physical-microbatch mean gradients failing the original tensor rule: [].
Logical averaged-gradient failures: [0, 7, 9, 10, 17, 19, 20, 22].
Instrumented first-gradient linkage to exact reproduction: `{"cpu": {"exact": true, "difference": {"elements": 142263170, "max_absolute": 0.0, "mean_absolute": 0.0, "max_relative": 0.0, "rms": 0.0, "relative_l2": 0.0, "reference_max_absolute": 0.33770859241485596, "reference_mean_absolute": 0.0014911317192919827, "reference_l2": 55.95498512799648}, "average_is_sum_divided_by_100": true}, "gpu": {"exact": true, "difference": {"elements": 142263170, "max_absolute": 0.0, "mean_absolute": 0.0, "max_relative": 0.0, "rms": 0.0, "relative_l2": 0.0, "reference_max_absolute": 0.3377090096473694, "reference_mean_absolute": 0.001491131827938342, "reference_l2": 55.95499557677683}, "average_is_sum_divided_by_100": true}}`.

The reference pass resets the unchanged accumulator after each physical microbatch, exports its summed gradient, and independently reduces those 25 increments in NumPy float32 and float64. It records 4 samples per isolated microgradient; the normal pass records 4,8,...,100 and averages by 100. Adam runs once per logical batch. No optimizer is applied in the isolated reference pass.

CPU: 47/47 independent float32 sums are bitwise equal to the actual accumulator. Largest float64-reference relative-L2 summation error: 9.92952621e-08. Per-tensor sums, differences and sample counts are retained in JSON.

GPU: 47/47 independent float32 sums are bitwise equal to the actual accumulator. Largest float64-reference relative-L2 summation error: 1.02476057e-07. Per-tensor sums, differences and sample counts are retained in JSON.

## Same-device controls

TensorFlow documents deterministic execution for the same inputs on the same hardware and software configuration; it does not promise CPU/GPU bitwise identity. These controls test the observed same-device behavior directly. [TensorFlow determinism API](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism).

- CPU A vs B: exact equality **True** across ten updates, including losses, gradient-sum hashes, model-weight hashes, prediction bytes, sample counts and optimizer iterations.
- GPU A vs B: exact equality **True** across ten updates, including losses, gradient-sum hashes, model-weight hashes, prediction bytes, sample counts and optimizer iterations.

## Ten-update trajectory

| Update | CPU loss | GPU loss | Loss difference | Class agreement (64) | Max probability difference | Weight L2 divergence | Weight relative L2 | Update relative L2 difference |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.846221945 | 0.846222136 | 1.90734863e-07 | 1.000000 | 2.08616257e-06 | 0.000155340164 | 2.04822884e-07 | 0.00157727891 |
| 2 | 1.8178488 | 1.81784796 | 8.39829445e-07 | 1.000000 | 2.57790089e-06 | 0.000186211338 | 2.45527888e-07 | 0.0011524189 |
| 3 | 0.419656652 | 0.419656298 | 3.54647636e-07 | 1.000000 | 8.37445259e-06 | 0.000217067949 | 2.86213689e-07 | 0.00100192743 |
| 4 | 0.874276655 | 0.874276607 | 4.81680035e-08 | 1.000000 | 9.11951065e-06 | 0.000245233264 | 3.23350895e-07 | 0.00115751907 |
| 5 | 0.721203182 | 0.721202981 | 2.01202929e-07 | 1.000000 | 6.85453415e-06 | 0.000274009789 | 3.61294004e-07 | 0.00124212028 |
| 6 | 0.744322344 | 0.744321505 | 8.38935375e-07 | 1.000000 | 1.65700912e-05 | 0.000300830593 | 3.96658405e-07 | 0.00124211938 |
| 7 | 0.417624006 | 0.41762304 | 9.65744257e-07 | 1.000000 | 7.80820847e-06 | 0.000327395281 | 4.31685104e-07 | 0.00121646827 |
| 8 | 0.62119775 | 0.621196114 | 1.63584948e-06 | 1.000000 | 8.01682472e-06 | 0.00035356831 | 4.66195384e-07 | 0.00144499036 |
| 9 | 0.376010557 | 0.376009968 | 5.88744879e-07 | 1.000000 | 1.03712082e-05 | 0.00038011944 | 5.01204203e-07 | 0.00156381145 |
| 10 | 0.607516943 | 0.607516318 | 6.25252724e-07 | 1.000000 | 1.06990337e-05 | 0.000406611841 | 5.36135587e-07 | 0.0016515477 |

Weight divergence compares the complete model states. Update-relative difference compares (GPU current - GPU previous) with (CPU current - CPU previous), normalized by the CPU update norm; this avoids hiding differences behind the much larger pretrained-weight norm.

Final 1000-image comparison: `{"probabilities": {"elements": 2000, "max_absolute": 1.7404556274414062e-05, "mean_absolute": 1.568258982686588e-06, "max_relative": 0.00010481620013610815, "rms": 2.8533402038758593e-06, "relative_l2": 4.4146508671130484e-06, "reference_max_absolute": 1.0, "reference_mean_absolute": 0.49999999959586283, "reference_l2": 28.904947869510185}, "loss_difference": 4.470348358154297e-07, "predicted_class_agreement": 1.0, "pass": true}`.

Evidence checks: `{"reproduction": true, "same_device_exact": true, "independent_float32_accumulation_exact": true, "instrumented_first_gradient_exact": true, "all_post_prediction_checks_pass": true, "final_1000_pass": true, "frozen_weights_exact": true, "averages_exact": true, "original_tensor_files_exact": true, "cross_device_inputs_and_initialization_exact": true, "all_weight_checks_pass": true, "sample_counts_correct": true}`.

## Interpretation and limits

**Material assessment: likely benign float32 backend differences, with no material short-trajectory effect observed.** No accumulator or sample-count inconsistency was found. This is a numerical GO recommendation for GPU production training, conditional on completing cache and trainer readiness checks before launching; it does not authorize or start that work.

The largest global update-relative difference is approximately 0.165%; individual failing tensors have first-update differences of approximately 0.36–1.00% of their own update norms. Thus the differences are measurable and Adam can amplify gradient perturbations, but their observed effects on loss and predictions remain small. Parameter divergence grows over ten updates without a corresponding material prediction divergence in this fixture. This supports a likely-benign judgment, not proof of identical full training.

Minimum gradient cosine: 0.999999025901; largest tensor-relative L2 error: 0.00139578051; global gradient-relative L2 error: 5.33499633e-05.

Across ten updates: largest loss difference 1.63584948e-06; largest probability difference 1.65700912e-05; loss-trajectory correlation 0.9999999999992099; largest model-weight relative divergence 5.36135587e-07; largest update-relative divergence 0.0016515477.

Model-weight L2 divergence changes from 0.000155340164 after update 1 to 0.000406611841 after update 10. Growth must be interpreted with the reported update scale and prediction effects; nonzero growth alone is not evidence of material instability.

- Microbatch discrepancies precede logical accumulation; exact independent sums rule out a summation implementation mismatch if controls pass.
- Frozen BN state is checked exactly. This rules out state mutation, not differences in float32 BN inference arithmetic.
- Convolution, nonlinear activation boundaries and reduction kernels are plausible sources; this experiment does not isolate the first divergent primitive.
- Large per-element relative errors near zero do not by themselves explain an L2-relative or maximum-absolute tensor failure.
- Float64 reference sums use already-computed float32 microgradients; they do not constitute float64 full-network gradients.
- Ten updates at the first unfreezing stage and 1000 training patches cannot prove identical 14-epoch trajectories or unchanged external generalization.
- Original strict thresholds and Stage 2E FAIL are retained. Material assessment is a separate evidence-based judgment, not a redefinition of PASS.

## Artifacts and validation

New code: `modern_pca/diagnose_parity.py`, `tools/report_stage2f.py`, `tests/test_diagnose_parity.py`. Raw runs, tensor vectors, sequence, configuration and progress logs are under the run directory. Tests and final Git checks are recorded in `reports/stage2f_checks.json`.

No commit or push was performed.
