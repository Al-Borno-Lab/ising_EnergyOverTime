#!/bin/bash
# Smoke test for plot_model_quality_summary — run via Singularity (same as single_test.sh)
cd "$(dirname "$0")" || exit 1

singularity exec ~/projectDir/singularity-env/inverse-ising-arm-2.sif /entrypoint.sh python -c "
import numpy as np
import tempfile
import os
N=8
S=500
rng=np.random.default_rng(0)
X=rng.choice([-1,1],size=(S,N))
J=np.random.randn(N*(N-1)//2)*0.1
h=np.random.randn(N)*0.05
mult=np.concatenate([h,J])
from visualization import plot_model_quality_summary
d=tempfile.mkdtemp()
plot_model_quality_summary(X,X,mult,N,d)
print('ok', os.listdir(d))
"
