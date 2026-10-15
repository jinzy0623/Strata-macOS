"""Copy a split GGUF, isolating its lazy embedding table from Metal weights.

Tensor bytes and metadata are preserved; only split metadata and offsets change.
The original files are read-only. SHA256 is checked for every copied tensor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

from gguf_reader import GGUFFile, GGUF_META


def raw_metadata(path):
    entries = {}
    with path.open('rb') as f:
        f.seek(16)
        count = struct.unpack('<Q', f.read(8))[0]

        def string():
            return f.read(struct.unpack('<Q', f.read(8))[0]).decode('utf-8')

        def skip(kind):
            name, size = GGUF_META[kind]
            if name == 'string':
                f.seek(struct.unpack('<Q', f.read(8))[0], 1)
            elif name == 'array':
                element, length = struct.unpack('<IQ', f.read(12))
                if GGUF_META[element][1] is not None:
                    f.seek(length * GGUF_META[element][1], 1)
                else:
                    for _ in range(length):
                        skip(element)
            else:
                f.seek(size, 1)

        for _ in range(count):
            start = f.tell()
            key = string()
            skip(struct.unpack('<I', f.read(4))[0])
            end = f.tell()
            f.seek(start)
            entries[key] = f.read(end - start)
    return entries


def key_value(key, kind, value, fmt):
    encoded = key.encode()
    return struct.pack('<Q', len(encoded)) + encoded + struct.pack('<I' + fmt, kind, value)


def write_shard(path, group, metadata, index, total, alignment):
    entries = dict(metadata)
    entries['split.no'] = key_value('split.no', 2, index, 'H')
    entries['split.count'] = key_value('split.count', 2, 2, 'H')
    entries['split.tensors.count'] = key_value('split.tensors.count', 5, total, 'i')
    header = bytearray(struct.pack('<4sIQQ', b'GGUF', 3, len(group), len(entries)))
    header.extend(b''.join(entries.values()))
    offsets, offset = [], 0
    for source, tensor in group:
        size = tensor.expected_bytes()
        if size is None:
            raise ValueError(f'Unsupported tensor geometry: {tensor.name}')
        name = tensor.name.encode()
        header.extend(struct.pack('<Q', len(name)) + name)
        header.extend(struct.pack('<I', len(tensor.shape)))
        header.extend(struct.pack(f'<{len(tensor.shape)}Q', *tensor.shape))
        header.extend(struct.pack('<IQ', tensor.type_id, offset))
        offsets.append(offset)
        offset += size
        offset += -offset % alignment
    header.extend(b'\0' * (-len(header) % alignment))
    hashes = {}
    temporary = path.with_suffix('.gguf.partial')
    with temporary.open('xb') as out:
        out.write(header)
        for i, (source, tensor) in enumerate(group):
            digest = hashlib.sha256()
            remaining = tensor.expected_bytes()
            with source.path.open('rb') as inp:
                inp.seek(source.data_start + tensor.offset)
                while remaining:
                    chunk = inp.read(min(16 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(f'Truncated tensor: {tensor.name}')
                    out.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            hashes[tensor.name] = digest.hexdigest()
            out.write(b'\0' * (-out.tell() % alignment))
            if i % 100 == 0 or i == len(group) - 1:
                print(f'{path.name}: copied {i + 1}/{len(group)} tensors', flush=True)
    copied = GGUFFile(temporary)
    assert len(copied.tensors) == len(group)
    for (_, original), new, offset in zip(group, copied.tensors, offsets):
        assert (original.name, original.shape, original.type_id, offset) == (
            new.name, new.shape, new.type_id, new.offset)
    with temporary.open('rb') as inp:
        for i, tensor in enumerate(copied.tensors):
            inp.seek(copied.data_start + tensor.offset)
            remaining, digest = tensor.expected_bytes(), hashlib.sha256()
            while remaining:
                chunk = inp.read(min(16 * 1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(f'Truncated output: {tensor.name}')
                digest.update(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != hashes[tensor.name]:
                raise ValueError(f'Copy checksum mismatch: {tensor.name}')
    temporary.rename(path)
    print(f'{path.name}: all tensor checksums verified', flush=True)
    return hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('first_shard', type=Path)
    ap.add_argument('output_prefix', type=Path)
    args = ap.parse_args()
    match = re.fullmatch(r'(.+)-00001-of-(\d{5})\.gguf', args.first_shard.name)
    if not match:
        ap.error('Pass the first shard of a split GGUF')
    sources = [GGUFFile(args.first_shard.with_name(
        f'{match[1]}-{part:05}-of-{int(match[2]):05}.gguf'))
        for part in range(1, int(match[2]) + 1)]
    tensors = [(s, t) for s in sources for t in s.tensors]
    if len({t.name for _, t in tensors}) != len(tensors):
        raise ValueError('Duplicate tensor names')
    groups = [[(s, t) for s, t in tensors if t.name != 'per_layer_token_embd.weight'],
              [(s, t) for s, t in tensors if t.name == 'per_layer_token_embd.weight']]
    if len(groups[1]) != 1:
        raise ValueError('Expected exactly one lazy embedding tensor')
    metadata = {k: v for k, v in raw_metadata(sources[0].path).items() if not k.startswith('split.')}
    alignment = sources[0].alignment
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [Path(f'{args.output_prefix}-{part:05}-of-00002.gguf') for part in (1, 2)]
    if any(p.exists() or p.with_suffix('.gguf.partial').exists() for p in outputs):
        raise ValueError('Output exists; refusing to overwrite')
    hashes = {}
    for i, (path, group) in enumerate(zip(outputs, groups)):
        hashes.update(write_shard(path, group, metadata, i, len(tensors), alignment))
    report = {'sources': [str(s.path) for s in sources], 'outputs': [str(p) for p in outputs],
              'tensor_count': len(tensors), 'tensor_sha256': hashes}
    Path(f'{args.output_prefix}.verification.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
