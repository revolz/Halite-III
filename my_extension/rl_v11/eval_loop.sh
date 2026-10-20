#!/bin/bash
# Every 20 iterations: real-engine games of the newest rl_v11 snapshot vs rl_v10 (24), V71 (8) and rl_v9 (8).
cd "$(dirname "$0")/.."
CK=rl_v11/checkpoints
done_list=$CK/evaluated.txt
touch $done_list
while true; do
  for s in $(ls $CK/snap_*.pt 2>/dev/null | sort); do
    it=$(basename $s .pt | sed 's/snap_0*//')
    if [ $((it % 20)) -ne 0 ] || grep -qx "$s" $done_list; then continue; fi
    for spec in "rl_v10 24" "v71 8" "rl_v9 8"; do
      set -- $spec
      python3.13 rl_v10/evaluate.py --opponent $1 --games $2 --workers 2 --name rl_v11 --seed0 $((1700000 + it * 10)) \
        --model $s --csv $CK/eval_history.csv 2>&1 | grep RESULT | sed "s#^#it $it #" >> $CK/eval_summary.txt
    done
    echo "$s" >> $done_list
  done
  sleep 120
done
