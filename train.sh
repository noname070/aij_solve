#!/bin/bash
REPO_URL="https://github.com/noname070/aij_solve.git"
REPO_DIR="/workspace/AIJ"

if [ -d "$REPO_DIR" ]; then
  echo "updating repo..."
  cd $REPO_DIR
  git pull origin main
else
  echo "clone repo..."
  git clone $REPO_URL $REPO_DIR
  cd $REPO_DIR
fi

echo "running training..."
python3 ./team_code/dev_trainer.py --epochs $EPOCHS --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE
tensorboard --logdir=./runs --host 0.0.0.0