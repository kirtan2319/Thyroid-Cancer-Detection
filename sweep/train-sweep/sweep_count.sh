OPTIONS=d:
LONGOPTS=sweep_id:
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
eval set -- "$PARSED"

sweep_id="$2"
echo "$sweep_id"

wandb agent --count 1 $sweep_id

echo "done"