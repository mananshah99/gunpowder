#!/usr/bin/env bash

rm snapshots/*

NAME=$(basename $(pwd)-training)

docker rm -f $NAME

USER_ID=${UID}

echo "Starting as user ${USER_ID}"

NV_GPU=0 nvidia-docker run --rm \
    -u ${USER_ID} \
    -v $HOME/data:$HOME/data \
    -v ${PWD}:${PWD} \
    -w ${PWD} \
    --name ${NAME} \
    funkey/gunpowder:v0.2-prerelease \
    /bin/bash -c "PYTHONPATH=$HOME/gunpowder:\$PYTHONPATH && python make_net.py && python -u train.py"
#    /bin/bash -c "PYTHONPATH=$HOME/src/gunpowder:\$PYTHONPATH && python make_net.py && python -u train.py"

#    /bin/bash -c "PYTHONPATH=$HOME/src/gunpowder:\$PYTHONPATH bash"
#    -e HOME=${USER_HOME} \
