1. run `cd scripts`
2. change `<path-to-your-folder>`, e.g., `/projects/xyz/`
3. load modules including python, pytorch, and cuda
4. run `./GPUrun_0.sh` on one machine
5. run `./GPUrun_1.sh` on another machin
6. make sure these two machines can access to the same file specified in `<path-to-your-folder>/sharedfile`
