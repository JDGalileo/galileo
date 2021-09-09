#!/bin/bash

pre=$(ls -d $PIP_CACHE_DIR/../_h_env*)
src_dir=$RECIPE_DIR/..

echo "build $src_dir $PKG_VERSION to $pre"

cd $src_dir
pip install --no-deps --prefix $pre dist/jdgalileo-${PKG_VERSION}-cp38-cp38-linux_x86_64.whl

