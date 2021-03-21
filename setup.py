from __future__ import print_function

import os
import sys
import shutil
import contextlib
import subprocess
import glob

from setuptools import setup, find_packages
from setuptools import Extension


HERE = os.path.dirname(os.path.abspath(__file__))

# import ``__version__` from code base
exec(open(os.path.join(HERE, 'drforest', 'version.py')).read())
VERSION = __version__

# armadillo header-files included with the package
ARMADILLO_INC = os.path.join(HERE, 'third-party')

MOD_NAMES = [
    'drforest.armadillo',
    'drforest.tree._tree',
    'drforest.ensemble._forest'
]


try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)


@contextlib.contextmanager
def chdir(new_dir):
    old_dir = os.getcwd()
    try:
        sys.path.insert(0, new_dir)
        yield
    finally:
        del sys.path[0]
        os.chdir(old_dir)


def find_cython(dir, files=None):
    if files is None:
        files = []

    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            find_cython(path, files)
    return files


def clean(path):
    for name in MOD_NAMES:
        name = name.replace('.', os.path.sep)
        for ext in ['*.cpp', '*.so', '*.o', '*.html']:
            file_path = glob.glob(os.path.join(path, name + ext))
            if file_path and os.path.exists(file_path[0]):
                os.unlink(file_path[0])


def get_include():
    source_path = os.path.join(HERE, 'core/src')
    return source_path if os.path.exists(source_path) else ''


def get_sources():
    files = []
    source_path = get_include()
    if source_path:
        for root, dirs, src_files in os.walk(source_path):
            for name in src_files:
                path = os.path.join(root, name)
                if os.path.isfile(path) and path.endswith(".cpp"):
                    print(os.path.relpath(path))
                    files.append(os.path.relpath(path))

    return files


def generate_cython(cython_cov=False):
    print("Cythonizing sources")
    for source in MOD_NAMES:
        source = source.replace('.', os.path.sep) + '.pyx'
        cythonize_source(source, cython_cov)


def cythonize_source(source, cython_cov=False):
    print("Processing %s" % source)

    flags = ['--fast-fail', '--cplus']
    if cython_cov:
        flags.extend(['--directive', 'linetrace=True'])

    try:
        p = subprocess.call(['cython'] + flags + [source])
        if p != 0:
            raise Exception('Cython failed')
    except OSError:
        raise OSError('Cython needs to be installed')


def make_extension(ext_name, macros=[]):
    ext_path = ext_name.replace('.', os.path.sep) + '.cpp'
    include_dirs = [numpy.get_include(), ARMADILLO_INC, ".", "./src"]
    library_dirs = ['/usr/lib']
    if get_include():
        include_dirs = [get_include()] + include_dirs

    return Extension(
        ext_name,
        sources=[ext_path] + get_sources(),
        include_dirs=include_dirs,
        extra_compile_args=["-O3", "-fPIC", "-std=c++1z", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        define_macros=macros,
        libraries=['blas', 'lapack', 'stdc++'],
        library_dirs=library_dirs,
        language='c++')


def generate_extensions(macros=[]):
    ext_modules = []
    for mod_name in MOD_NAMES:
        ext_modules.append(make_extension(mod_name, macros=macros))

    return ext_modules


def setup_package():

    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        package_src = os.path.join(HERE, "src")
        if os.path.exists(package_src):
            shutil.rmtree(package_src)

        return clean(HERE)

    cython_cov = 'CYTHON_COV' in os.environ

    macros = []
    if cython_cov:
        print("Adding coverage information to cythonized files.")
        macros =  [('CYTHON_TRACE_NOGIL', 1)]

    with chdir(HERE):
        generate_cython(cython_cov)
        ext_modules = generate_extensions(macros=macros)
        setup(
            name="drforest",
            version=VERSION,
            packages=find_packages(),
            ext_modules=ext_modules,
            package_data={"": ["*.pyx", "*.pxd"]},
        )

if __name__ == '__main__':
    setup_package()
