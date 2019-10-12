#!/bin/bash
export TF_RUN_DIR="~/tf"

if [ -z $2 ]; then
    echo "Usage: start_cluster <python script> <cluster mode>"
    echo "Here, <python script> contains the cluster spec that assigns an ID to all server."
else
    echo "Create $TF_RUN_DIR on remote hosts if they do not exist."
    echo "Copying the script to all the remote hosts."
    for i in `seq 0 2`; do
        ssh node$i "mkdir -p $TF_RUN_DIR"
        scp $1 node$i:$TF_RUN_DIR
    done
    echo "Starting tensorflow servers on all hosts based on the spec in $1"
    echo "The server output is logged to serverlog-i.out, where i = 0, ..., 3 are the VM numbers."
    if [ "$2" = "single" ]; then
        nohup ssh node0 "cd ~/tf ; python3 $1 --deploy_mode=single" > serverlog-0.out 2>&1&
    elif [ "$2" = "cluster1" ]; then
        nohup ssh node0 "cd ~/tf ; python3 $1 --deploy_mode=cluster1  --task_index=0" > serverlog-0.out 2>&1&
        nohup ssh node1 "cd ~/tf ; python3 $1 --deploy_mode=cluster1  --task_index=1" > serverlog-1.out 2>&1&
    else
        nohup ssh node0 "cd ~/tf ; python3 $1 --deploy_mode=cluster2 --task_index=0" > serverlog-0.out 2>&1&
        nohup ssh node1 "cd ~/tf ; python3 $1 --deploy_mode=cluster2 --task_index=1" > serverlog-1.out 2>&1&
        nohup ssh node2 "cd ~/tf ; python3 $1 --deploy_mode=cluster2 --task_index=2" > serverlog-2.out 2>&1&
    fi
fi