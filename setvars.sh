#!/bin/bash

_sourcecmd="${BASH_SOURCE:-$0}"

_dir="$( cd -P "$( dirname "$_sourcecmd" )" && pwd )"

export KORE_HOME=$_dir
export PYTHONPATH=$PYTHONPATH:$KORE_HOME
