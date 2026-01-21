#!/bin/bash
# run_ml_pipeline.sh
#
# Run all scripts to conduct ML for classification and hedge portfolio creation
#

# Exit immediately if any variable is unset
set -u  


# Print a timestamp for logging clarity
echo "[$(date)] Starting full ML pipeline"

# Create log directory if it doesn’t exist
mkdir -p logs/review/temporal

# Run sequentially, continue even if one step fails
#time python -m fintextsim.create_test_train_datasets_2016_2023_temporal_masked >> logs/review/temporal/1_datasets.log 2>&1 ;  echo "Step 1 done: Datasets"
#time python -m fintextsim.run_fintextsim_training_2016_2023_triplet_temporal >> logs/review/temporal/2_fts_training.log 2>&1 ;  echo "Step 2 done: FTS trained"
#time python -m bertopic_models.generate_embeddings_temporal >> logs/review/temporal/3_embeddings.log 2>&1 ;  echo "Step 3 done: Embeddings generated"
#time python -m bertopic_models.fit_bertopic_models_temporal >> logs/review/temporal/4_bertopic.log 2>&1 ;  echo "Step 4 done: BERTopic fitted"
time python -m bertopic_models.approximate_topic_distributions_temporal >> logs/review/temporal/5_topic_dists.log 2>&1 ;  echo "Step 5 done: Topic distributions approximated"
time python -m evaluation.run_bertopic_model_evaluation_temporal >> logs/review/temporal/6_bertopic_eval.log 2>&1 ;  echo "Step 6 done: BERTopic evaluated"
time python -m feature_creation.create_text_features_temporal >> logs/review/temporal/7_text_features.log 2>&1 ;  echo "Step 7 done: text features created"
time python -m ml.Financial.run_lr_classification_temporal >> logs/review/temporal/8_lr.log 2>&1 ;  echo "Step 8 done: LR"
time python -m ml.Financial.run_xgb_classification_temporal >> logs/review/temporal/10_xgb.log 2>&1 ;  echo "Step 9 done: XGB"

echo "[$(date)] All scripts finished."
