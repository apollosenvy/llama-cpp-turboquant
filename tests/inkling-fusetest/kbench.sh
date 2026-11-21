#!/usr/bin/env bash
# Usage: kbench.sh <case-substring>   Per-kernel busy time for one fusetest case, from a rocprofv3 kernel trace.
cd "$(dirname "$0")"; rm -rf kb; mkdir kb
FUSEBENCH=300 /opt/rocm/bin/rocprofv3 --kernel-trace -f csv -d kb -o run -- ./fusetest "$1" > kb/out.txt 2>&1
python3 - <<'P'
import csv,collections,glob
f=glob.glob('kb/**/run_kernel_trace.csv',recursive=True)[0]
a=collections.defaultdict(list)
for r in csv.DictReader(open(f)):
    a[r['Kernel_Name'].split('(')[0][:60]].append(int(r['End_Timestamp'])-int(r['Start_Timestamp']))
for k,v in sorted(a.items(), key=lambda x:-sum(x[1])):
    v.sort(); print(f"{len(v):6d} calls  median {v[len(v)//2]/1000:6.2f} us  {k}")
P
