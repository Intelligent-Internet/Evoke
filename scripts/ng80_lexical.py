"""Explicit native lexical binding and TRAIN-only cross-platform parity."""

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import time


MAC_LIBRARY_SHA = '49b9cfbbb6174c65031ca5e9ddd5d875d0aa79e2c8e609949fad136878d67ae5'
PREFLIGHT_SHA = '253de2432f9d4f2b05970eaf744505dc7b1fa9fe3f8dc897db1aae94a77c47cc'
ICU_LOCALE = 'en_US_POSIX'
OPTIONS = dict(lowercase=True, stem_english=False,
               fold_diacritics=False, stopwords=[])


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


class Options(C.Structure):
    _fields_ = [('lowercase', C.c_bool), ('stem_english', C.c_bool),
               ('fold_diacritics', C.c_bool),
               ('stopwords', C.POINTER(C.c_char_p)),
               ('num_stopwords', C.c_size_t)]


def pin_locale(library):
    # The frozen C tokenizer uses ICU's default break locale. Research workers
    # must not inherit different token boundaries from tmux or SSH locales.
    setter = library.uloc_setDefault_78
    getter = library.uloc_getDefault_78
    setter.argtypes, setter.restype = [C.c_char_p, C.POINTER(C.c_int)], None
    getter.argtypes, getter.restype = [], C.c_char_p
    status = C.c_int(0)
    setter(ICU_LOCALE.encode(), C.byref(status))
    if status.value > 0 or getter() != ICU_LOCALE.encode():
        raise ValueError('failed to pin original ICU locale')


class Tokenizer:
    def __init__(self, library):
        library = Path(library).resolve(strict=True)
        with library.open('rb') as stream:
            magic = stream.read(4)
        expected = {'Linux': b'\x7fELF', 'Darwin': b'\xcf\xfa\xed\xfe'}
        if magic != expected.get(platform.system()):
            raise ValueError('native tokenizer platform mismatch: ' + str(library))
        self.lib = C.CDLL(str(library))
        pin_locale(self.lib)
        tokens = C.POINTER(C.c_char_p)
        self.lib.ii42_tokenize_text.argtypes = [
            C.c_char_p, C.POINTER(Options), C.POINTER(tokens),
            C.POINTER(C.c_size_t)]
        self.lib.ii42_tokenize_text.restype = C.c_int
        self.lib.ii42_text_tokens_free.argtypes = [tokens, C.c_size_t]
        self.lib.ii42_text_tokens_free.restype = None
        self.options = Options(True, False, False, None, 0)

    def __call__(self, text):
        if '\x00' in text:
            raise ValueError('NUL in lexical input')
        tokens, length = C.POINTER(C.c_char_p)(), C.c_size_t()
        status = self.lib.ii42_tokenize_text(
            text.encode(), C.byref(self.options), C.byref(tokens),
            C.byref(length))
        if status != 0:
            raise ValueError('native tokenize failed: ' + str(status))
        try:
            return [tokens[i].decode() for i in range(length.value)]
        finally:
            self.lib.ii42_text_tokens_free(tokens, length)


def parity(tokenizer, queries, golden):
    if len(queries) != len(golden) or not queries:
        raise ValueError('lexical parity row count')
    for query, expected in zip(queries, golden, strict=True):
        digest = hashlib.sha256(query['query'].encode()).hexdigest()
        if (query['query_id'] != expected['query_id']
                or digest != expected['text_sha256']
                or tokenizer(query['query']) != expected['tokens']):
            raise ValueError('lexical parity mismatch: ' + query['query_id'])
    return dict(queries=len(queries), exact_token_sequences=True,
                options=OPTIONS, locked_test_access=False)


def training_queries(base):
    parent = base / 'NG-0080/sparse-preflight-v1'
    if sha(parent / 'inputs.json') != PREFLIGHT_SHA:
        raise ValueError('reference preflight input changed')
    manifest = read(parent / 'inputs.json')
    paths = {'NG-0080/sparse-preflight-v1/order.json':
             manifest['source']['order.json']}
    name = 'NG-0080/readout-v1/prepare/queries.json'
    paths[name] = manifest['dependencies'][name]
    for name, digest in paths.items():
        if sha(base / name) != digest:
            raise ValueError('reference query input changed: ' + name)
    queries = read(base / 'NG-0080/readout-v1/prepare/queries.json')
    order = read(parent / 'order.json')
    if len(order) != 12288 or len(set(order)) != len(order):
        raise ValueError('reference TRAIN order changed')
    return [queries[i] for i in order], paths


def golden(base, output):
    if platform.system() != 'Darwin':
        raise ValueError('golden requires the original Mac library')
    library = base / 'NG-0008/native/libtext.dylib'
    if sha(library) != MAC_LIBRARY_SHA:
        raise ValueError('original lexical library changed')
    queries, dependencies = training_queries(base)
    source = Path(__file__).resolve().parents[1] / 'src'
    frozen = read(base / 'NG-0008/frozen.json')['files']
    frozen_by_name = {Path(name).name: digest for name, digest in frozen.items()}
    files = {}
    for stem in ('ii42_text', 'ii42_stem', 'ii42_query', 'ii42_core'):
        for suffix in ('.c', '.h'):
            path = source / (stem + suffix)
            expected = frozen_by_name[path.name]
            if sha(path) != expected:
                raise ValueError('native source differs from frozen NG8: ' + path.name)
            files[path.name] = expected
    output.mkdir(exist_ok=False)
    (output / 'src').mkdir()
    for name in files:
        shutil.copy2(source / name, output / 'src' / name)
    shutil.copy2(Path(__file__), output / 'ng80_lexical.py')
    tokenizer = Tokenizer(library)
    records = [dict(query_id=q['query_id'],
                    text_sha256=hashlib.sha256(q['query'].encode()).hexdigest(),
                    tokens=tokenizer(q['query'])) for q in queries]
    write(output / 'golden.json', records)
    write(output / 'inputs.json', dict(
        protocol='NG80_native_lexical_parity_v1', train_queries=12288,
        dependencies=dependencies, original_library_sha256=MAC_LIBRARY_SHA,
        options=OPTIONS, locked_test_access=False,
        files={p.relative_to(output).as_posix(): sha(p)
               for p in output.rglob('*') if p.is_file()}))


def build(base, output):
    if platform.system() != 'Linux':
        raise ValueError('runtime build requires Linux')
    inputs = read(output / 'inputs.json')
    for name, digest in inputs['files'].items():
        if sha(output / name) != digest:
            raise ValueError('changed lexical source: ' + name)
    queries, dependencies = training_queries(base)
    if dependencies != inputs['dependencies']:
        raise ValueError('lexical TRAIN dependencies changed')
    library = output / 'libtext.so'
    if library.exists() or (output / 'complete.json').exists():
        raise ValueError('never overwrite a native build attempt')
    started = time.monotonic()
    toolchain = output / 'toolchain'
    if not toolchain.is_dir():
        raise ValueError('explicit isolated ICU toolchain required')
    env = dict(os.environ, PKG_CONFIG_SYSROOT_DIR=str(toolchain),
               PKG_CONFIG_LIBDIR=str(toolchain / 'usr/lib/x86_64-linux-gnu/pkgconfig'))
    flags = subprocess.check_output(
        ['pkg-config', '--cflags', '--libs', 'icu-uc', 'icu-i18n'], text=True, env=env)
    command = ['cc', '-std=c11', '-O2', '-shared', '-fPIC', '-Wl,-z,defs',
               '-Wl,--disable-new-dtags',
               '-Wl,-rpath,$ORIGIN/toolchain/usr/lib/x86_64-linux-gnu',
               '-o', str(library)]
    command += [str(output / 'src' / (stem + '.c')) for stem in
                ('ii42_text', 'ii42_stem', 'ii42_query', 'ii42_core')]
    command += shlex.split(flags) + ['-lm']
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    write(output / 'build.json', dict(command=command, exit_code=result.returncode,
        stdout=result.stdout, stderr=result.stderr,
        compiler=subprocess.check_output(['cc', '--version'], text=True),
        icu=subprocess.check_output(['pkg-config', '--modversion', 'icu-uc'], text=True, env=env)))
    result.check_returncode()
    checked = parity(Tokenizer(library), queries, read(output / 'golden.json'))
    write(output / 'parity.json', dict(checked, seconds=time.monotonic() - started,
                                      library_sha256=sha(library)))
    write(output / 'complete.json', dict(passed=True,
        files={p.relative_to(output).as_posix(): sha(p)
               for p in output.rglob('*') if p.is_file()}))
    print('NG80_LEXICAL_RUNTIME_PASSED', sha(output / 'complete.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('golden', 'build'))
    parser.add_argument('--research-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    {'golden': golden, 'build': build}[args.mode](args.research_root, args.output)
