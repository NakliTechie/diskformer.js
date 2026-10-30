#!/usr/bin/env python3
"""gguf-cut.py: copy the first N layers of a GGUF (with the embedding, output norm and output) into a smaller GGUF,
for layer-by-layer gates on a model that does not fit in memory. Tensor bytes are copied unchanged; the only metadata
changes are <arch>.block_count = N and any per-layer array (e.g. clover_moe.attention.recurrent_layers) cut to N.

  python3 scripts/gguf-cut.py IN.gguf OUT.gguf --layers 4
"""
import re, sys
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType as VT

src, dst = sys.argv[1], sys.argv[2]
n = int(sys.argv[sys.argv.index('--layers') + 1])
r = GGUFReader(src)
arch = bytes(r.fields['general.architecture'].parts[-1]).decode()
w = GGUFWriter(dst, arch)

def value(f):
    t = f.types[0]
    if t == VT.STRING:
        return bytes(f.parts[f.data[0]]).decode('utf-8', errors='surrogateescape')
    if t == VT.ARRAY:
        sub = f.types[1]
        if sub == VT.STRING:
            return [bytes(f.parts[i]).decode('utf-8', errors='surrogateescape') for i in f.data]
        return [f.parts[i].tolist()[0] for i in f.data]
    return f.parts[f.data[0]].tolist()[0]

for key, f in r.fields.items():
    if key.startswith('GGUF.') or key == 'general.architecture':
        continue
    v, t = value(f), f.types[0]
    if key == f'{arch}.block_count':
        v = n
    if t == VT.ARRAY:
        if len(v) == r.fields[f'{arch}.block_count'].parts[-1][0] and key.startswith(arch + '.'):
            v = v[:n]
        sub = f.types[1]
        if sub == VT.STRING:
            w.add_array(key, v)
        else:
            w.add_array(key, [bool(x) for x in v] if sub == VT.BOOL else v)
    else:
        w.add_key_value(key, v, t)

keep = []
for t in r.tensors:
    m = re.match(r'blk\.(\d+)\.', t.name)
    if m and int(m.group(1)) >= n:
        continue
    keep.append(t)
for t in keep:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, raw_dtype=t.tensor_type)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for t in keep:
    w.write_tensor_data(t.data)
w.close()
print(f'{dst}: {len(keep)} tensors, {n} layers')
