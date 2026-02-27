# Setup the environment to use Kore - Polytropic branch

# Load pyenv
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

# Load petsc, slepc
export PETSC_DIR=/home/ucl/mema/dlaariar/.local/src/kore/petsc
export PETSC_ARCH=arch-linux-c-opt
export PATH=$PETSC_DIR/$PETSC_ARCH/bin:$PATH
export SLEPC_DIR=/home/ucl/mema/dlaariar/.local/src/kore/slepc
export PYTHONPATH=$PETSC_DIR/$PETSC_ARCH/lib

# Load modules
# module purge
module load foss

# Kore directory
#export KORE_HOME="/home/ucl/mema/dlaariar/.local/src/kore/kore"
export KORE_HOME="/home/ucl/mema/dlaariar/.local/src/kore/polytropic_dr/kore"
