#!/bin/bash
python -c "import galileo;print(galileo.libs_dir)" > /etc/ld.so.conf.d/galileo.conf && ldconfig
