#!/usr/bin/env bash

rm snapshots/*

NAME=$(basename $(pwd)-training)

sudo mount --make-shared /nrs/turaga

docker rm -f $NAME

USER_ID=${UID}

echo "Starting as user ${USER_ID}"

NV_GPU=1 nvidia-docker run --rm \
    -u ${USER_ID} \
    -v /groups/turaga/home:/groups/turaga/home \
    -v /nrs/turaga:/nrs/turaga:shared \
    -w ${PWD} \
    --name ${NAME} \
    funkey/gunpowder:latest \
    /bin/bash -c "PYTHONPATH=$HOME/src/gunpowder:\$PYTHONPATH && python make_net.py && python -u train.py"

#    /bin/bash -c "PYTHONPATH=$HOME/src/gunpowder:\$PYTHONPATH bash"
#    -e HOME=${USER_HOME} \
