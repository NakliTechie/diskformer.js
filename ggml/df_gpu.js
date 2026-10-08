// Expert uploads for the browser build: OPFS bytes land in mapped staging buffers (no copy through the wasm heap) and
// a copyBufferToBuffer per expert moves them into the backend's pool slots, on the backend's own queue.
addToLibrary({
  $dfStaging: { ring: [], size: 32 << 20, count: 4, next: 0, spare: [] },

  df_upload_batch_js__deps: ['$WebGPU', '$dfStaging'],
  df_upload_batch_js__async: true,
  df_upload_batch_js: async function (device, n, offsets, sizes, buffers, dstOffsets) {
    const dev = WebGPU.getJsObject(Number(device));
    const S = dfStaging;
    const readers = Module['ggufReaders'], handle = Module['ggufHandle'], file = Module['ggufFile'];
    const at = (p, i) => HEAPF64[Number(p) / 8 + i];
    const buf = (i) => WebGPU.getJsObject(Number(HEAPU64[Number(buffers) / 8 + i]));
    for (let i = 0; i < n;) {
      // the next staging buffer of the ring, mapped (it was unmapped and copied from last time round)
      if (S.ring.length < S.count) {
        S.ring.push({ buf: dev.createBuffer({ size: S.size, usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC, mappedAtCreation: true }), mapped: null });
      }
      const st = S.ring[S.next];
      S.next = (S.next + 1) % S.count;
      if (st.mapped) await st.mapped;
      const view = new Uint8Array(st.buf.getMappedRange());
      const enc = dev.createCommandEncoder();
      let pos = 0;
      const pending = [];
      while (i < n && pos + at(sizes, i) <= S.size) {
        const o = at(offsets, i), s = at(sizes, i), p = pos;  // p: this read's place in the mapped range
        if (readers) {
          // reads in flight on the reader workers at once; each lands in the mapped range when it returns
          pending.push(readers.read(o, s, S.spare.pop()).then((r) => {
            if (r.got !== s) throw new Error(`short OPFS read: ${r.got} of ${s}`);
            view.set(new Uint8Array(r.buf, 0, s), p);
            S.spare.push(r.buf);
          }));
        } else if (handle) {
          const got = handle.read(view.subarray(pos, pos + s), { at: o });
          if (got !== s) throw new Error(`short OPFS read: ${got} of ${s}`);
        } else {
          pending.push(file.slice(o, o + s).arrayBuffer().then((b) => view.set(new Uint8Array(b), p)));
        }
        enc.copyBufferToBuffer(st.buf, pos, buf(i), at(dstOffsets, i), s);
        pos += s;
        i++;
      }
      if (pos === 0) throw new Error('expert larger than a staging buffer');
      await Promise.all(pending);
      st.buf.unmap();
      dev.queue.submit([enc.finish()]);
      st.mapped = st.buf.mapAsync(GPUMapMode.WRITE);
    }
  },
});
