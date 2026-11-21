#!/usr/bin/env bash
# Usage: kgrid.sh <case-substring> <kernel-substring>   Grid, block size and register use of matching kernels in one case.
cd "$(dirname "$0")"; rm -rf kb; mkdir kb
FUSEBENCH=100 /opt/rocm/bin/rocprofv3 --kernel-trace -f csv -d kb -o run -- ./fusetest "$1" > kb/out.txt 2>&1
python3 - "$2" <<'P'
import csv,collections,glob,sys
f=glob.glob('kb/**/run_kernel_trace.csv',recursive=True)[0]
a=collections.defaultdict(list)
for r in csv.DictReader(open(f)):
    if sys.argv[1] in r['Kernel_Name']:
        a[(r['Kernel_Name'].split('(')[0][:34], r['Grid_Size_X'],r['Grid_Size_Y'],r['Grid_Size_Z'],r['Workgroup_Size_X'],r['Workgroup_Size_Y'],r['VGPR_Count'],r['LDS_Block_Size'])].append(int(r['End_Timestamp'])-int(r['Start_Timestamp']))
for k,v in sorted(a.items(), key=lambda x:-len(x[1])):
    v.sort(); print(f"{len(v):5d} calls median {v[len(v)//2]/1000:7.2f} us  grid {k[1]}x{k[2]}x{k[3]} wg {k[4]}x{k[5]} vgpr {k[6]} lds {k[7]}  {k[0]}")
P
