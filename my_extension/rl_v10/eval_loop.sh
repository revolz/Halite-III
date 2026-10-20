#!/bin/bash
# Periodically evaluate the newest snapshot against V71 (and rl_v9) through the real engine.
cd "$(dirname "$0")/.."
done_list=rl_v10/checkpoints/evaluated.txt
touch $done_list
while true; do
  for s in $(ls rl_v10/checkpoints/snap_*.pt 2>/dev/null | sort); do
    it=$(basename $s .pt | sed 's/snap_0*//')
    if [ $((it % 20)) -ne 0 ] || grep -qx "$s" $done_list; then continue; fi
    for opp in v71 rl_v9; do
      python3.13 rl_v10/evaluate.py --opponent $opp --games 8 --workers 2 --seed0 $((700000 + it * 10)) \
        --model $s --csv rl_v10/checkpoints/eval_history.csv 2>&1 | grep RESULT | sed "s#^#it $it #" >> rl_v10/checkpoints/eval_summary.txt
    done
    echo "$s" >> $done_list
  done
  sleep 120
done
