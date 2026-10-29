#!/usr/bin/env bash
# Tempo mediano (5 testes) e pico de RSS por adaptador, um processo por (n, d). 4 venvs em paralelo.
B=$(cd "$(dirname "$0")/../.." && pwd); M=$B/benchmark/batch_multivariado/mv.py; O=$B/benchmark/batch_multivariado/results/cost.csv
run(){ v=$1; shift; for a in "$@"; do for nd in "50 10" "200 10" "1000 10" "200 100" "1000 100"; do
  timeout 900 "$B/$v/bin/python" -I "$M" "$a" --cost $nd 2>>"$O.err" | tail -1 | grep . || echo "$a,${nd/ /,},FAIL"; done; done; }
echo "adapter,n,d,sec_median,rss_base_MB,rss_peak_MB,rss_delta_MB" > "$O"; : > "$O.err"
run .venv ref_mmd ref_c2st ref_ksb ref_chi2 evidently nannyml_pca nannyml_dc > "$O.a" &
run .venv-alibi-torch alibi_mmd alibi_lsdd alibi_clf alibi_ks alibi_cvm alibi_chi2 > "$O.b" &
run .venv-frouros-torchdrift frouros_mmd frouros_mmd_med torchdrift_mmd > "$O.c" &
run .venv-deepchecks-whylogs deepchecks_mv > "$O.d" &
wait; cat "$O".[abcd] >> "$O"; rm "$O".[abcd]
