The codes are based on https://github.com/zbh2047/clipping-algorithms

1. run `cd scripts`
2. change `<path-to-your-folder>`, e.g., `/projects/xyz/`
3. load modules including python, pytorch, and cuda

For Penn Treebank:
- run `/PTB_GPUrun_0.sh` on one machine
- run `./PTB_GPUrun_1.sh` on another machine

For Wikitext2:
- run `./WT_GPUrun_0.sh` on one machine
- run `./WT_GPUrun_1.sh` on another machin

Make sure these two machines can access to the same file specified in `<path-to-your-folder>/sharedfile`
