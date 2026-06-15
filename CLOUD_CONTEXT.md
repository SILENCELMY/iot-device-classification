cd ~/iot-device-classification/code
cat > CLOUD_CONTEXT.md <<'EOF'
# Cloud Context

Project root:
~/iot-device-classification

Code:
~/iot-device-classification/code

Dataset:
~/iot-device-classification/dataset

Results:
~/iot-device-classification/results

Figures:
~/iot-device-classification/figures

Reports:
~/iot-device-classification/reports

Conda environment:
iotcls

Before running experiments:
conda activate iotcls

Main script:
scripts/robust_iot_research.py

Default dataset argument:
--dataset-root ../dataset

Default output root:
--output-root ../results/<run_name>

Research topic:
面向复杂场景的 IoT 设备鲁棒识别方法研究

Main principle:
Use FULL features by default. RSSI is only one feature family and should not be treated as the main research line.

Core smoke test:
python scripts/robust_iot_research.py \
  --dataset-root ../dataset \
  --output-root ../results/cloud_smoke \
  --tasks single_round_R2 \
  --models rf,xgboost,lightgbm,stacking \
  --feature-mode selected \
  --max-rows 300 \
  --n-jobs 16
EOF