import re, numpy as np, sys, os

HDR = "DeeployTest/TEST_SIRACUSA/Tests/testLIF/testoutputs.h"
REF = "DeeployTest/Tests/testLIF/outputs.npz"  # adjust if needed

def parse_array(text, name):
    pat = re.compile(r'\b(?:float|float32_t)\s+'+re.escape(name)+r'\s*\[[^\]]*\]\s*=\s*\{([^}]*)\}', re.S)
    m = pat.search(text)
    if not m:
        raise KeyError(name+" not found")
    vals = m.group(1)
    vals = re.sub(r'/\*.*?\*/', '', vals, flags=re.S)
    vals = re.sub(r'[fF]\b', '', vals)
    tokens = [t for t in re.split(r'[,\s]+', vals) if t!='']
    return np.array([float(t) for t in tokens], dtype=np.float32)

txt = open(HDR,'r', encoding='utf-8').read()
a0 = parse_array(txt, "testOutputVector0")
a1 = parse_array(txt, "testOutputVector1")

if not os.path.exists(REF):
    print("Reference file not found:", REF); sys.exit(1)
ref = np.load(REF)
print("Reference keys:", ref.files)
print("Header sizes:", a0.size, a1.size)

# try reshape to (1,4,32,32) if sizes match
def try_reshape(v):
    for shape in [(1,4,32,32),(1,32,32,4)]:
        if v.size == np.prod(shape):
            return v.reshape(shape)
    return None

a0r = try_reshape(a0); a1r = try_reshape(a1)
print("a0 reshaped:", None if a0r is None else a0r.shape)
print("a1 reshaped:", None if a1r is None else a1r.shape)

# helper to report diffs
def report(name, arr, refarr):
    if arr.shape != refarr.shape:
        print(f"{name}: SHAPE MISMATCH header {arr.shape} vs ref {refarr.shape}")
        return
    diff = np.abs(arr - refarr)
    maxd = diff.max()
    print(f"{name}: max abs diff = {maxd}, equal count = {np.sum(diff==0)} / {diff.size}")
    if maxd != 0:
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(" first mismatch index:", idx, " header value:", arr[idx], " ref value:", refarr[idx])

# test all plausible mappings: header a0->spk, a1->mem_out and swapped, both NCHW interpretations
for rk in ref.files:
    print("\n--- comparing header arrays to ref key:", rk)
    r = ref[rk].astype(np.float32)
    # try both direct and transposed (NHWC<->NCHW)
    for label, arr in [("a0", a0r), ("a1", a1r)]:
        if arr is None:
            continue
        report(f"{label} as-is", arr, r)
        # transpose 0,3,1,2 if shapes allow
        try:
            t = arr.transpose(0,3,1,2)
            report(f"{label} transposed(0,3,1,2)", t, r)
        except Exception:
            pass

# quick explicit mapping checks
print("\nExplicit mapping checks (a0->spk, a1->mem_out) and swapped:")
# enforce correct mapping: header a0 -> spk, a1 -> mem_out
report("a0 vs spk", a0r, ref['spk'].astype(np.float32))
report("a1 vs mem_out", a1r, ref['mem_out'].astype(np.float32))