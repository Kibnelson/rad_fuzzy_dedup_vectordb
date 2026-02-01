# helper one‑liners
PYBIN=python                                   # now points to venv python
INCLUDES=$($PYBIN -m pybind11 --includes)
EXT_SUFFIX=$($PYBIN - <<'PY'
import sysconfig, sys
print(sysconfig.get_config_var("EXT_SUFFIX"))
PY
)

LDFLAGS=$($PYBIN - <<'PY'
import sysconfig, shlex, sys
# Collect LIBDIR, LIBPL, and the lib flags in one line
flags = []
cfg = sysconfig.get_config_vars()
if cfg.get("LIBDIR"):
    flags.append("-L" + cfg["LIBDIR"])
if cfg.get("LIBPL"):
    flags.append("-L" + cfg["LIBPL"])
for k in ("LIBS", "SYSLIBS", "LDFLAGS"):
    if cfg.get(k):
        flags.extend(shlex.split(cfg[k]))
print(" ".join(flags))
PY
)

# final compile
# g++-13 -O3 -march=native -flto -fopenmp -std=c++17 -shared -fPIC \
#   $INCLUDES \
#   simd_fuzzy_deduplicationv7_unsorted_multi7A.cpp MurmurHash3.cpp \
#   $LDFLAGS \
#   -o simd_fuzzy_deduplicationv7_unsorted_multi7A$EXT_SUFFIX

g++-13 -O3 -march=native -mtune=native -mavx512f -mfma -fopenmp -ftree-vectorize -Wall -shared -std=c++17 -fPIC \
  $INCLUDES \
  simd_fuzzy_deduplicationv7_unsorted_multi7A_V1.cpp MurmurHash3.cpp \
  $LDFLAGS \
  -o simd_fuzzy_deduplicationv7_unsorted_multi7A_V1$EXT_SUFFIX

