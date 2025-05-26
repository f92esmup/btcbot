# Refactored BTCBot Data Pipeline & Model Management

This document describes the changes made to the BTCBot system to improve the data pipeline and model management strategy.

## Key Improvements

1. **Integrated Data Acquisition & Preprocessing**
   - Consolidated data handling into a single, streamlined process
   - Implemented chunk-based processing (quarterly) to manage memory with large datasets
   - Created new `src/data/data_pipeline.py` module to handle all data processing
   - New command: `scripts/run_data_pipeline.py` replaces separate download & preprocess scripts

2. **Smart Checkpointing & Resuming**
   - Training now automatically loads and starts from the existing "best model" if available
   - Data pipeline can resume from where it left off by checking existing chunks
   - Added ability to load and concatenate multiple processed sequence files

3. **Optimized Model Management - "Best Model Only" Strategy**
   - Simplified model management by maintaining only a single "best model" file in GCS
   - Best model is automatically updated during training when improvements are found
   - New path: `gs://YOUR_BUCKET/models/best_sac_model.zip`

4. **Live Trading - Automatic Best Model Selection**
   - Live trading bot automatically uses the latest "best model" without manual updates
   - Single source of truth for model selection with clear upgrade path

## New Components

### Integrated Data Pipeline
- **Core module**: `src/data/data_pipeline.py`
- **Runner script**: `scripts/run_data_pipeline.py`
- **Key features**:
  - Quarterly chunking for memory management
  - Smart resuming (only processes missing chunks)
  - Final processed sequences saved to `gs://YOUR_BUCKET/data/processed_sequences/`

### Enhanced Data Loading
- **New module**: `src/environments/chunked_data_loader.py`
- **Key features**:
  - Loads and concatenates multiple sequence chunks
  - Fallback to legacy formats for backward compatibility
  - Optimized memory usage for large datasets

### Best Model Strategy
- **New module**: `src/agent/best_model_strategy.py`
- **New training script**: `scripts/train_with_best_model.py`
- **Key features**:
  - Single "best model" management
  - Automatic model evaluation and updates
  - Smart continuation from existing best model

## Configuration Changes

The following changes were made to `src/config.yaml`:

1. **New Data Paths**:
   ```yaml
   data_paths:
     gcs_processed_sequences: "data/processed_sequences"
     gcs_best_model: "models/best_sac_model.zip"
   ```

2. **Data Pipeline Configuration**:
   ```yaml
   preprocessing:
     chunk_duration_months: 3
     required_buffer_periods: 200
   ```

3. **Best Model Strategy Configuration**:
   ```yaml
   agent:
     best_model_path: "models/best_sac_model.zip"
     enable_best_model_only: true
     best_model_evaluation_frequency: 10000
   ```

## Usage Instructions

### Running the Data Pipeline

```bash
# Process data for a specific date range
python scripts/run_data_pipeline.py --start-date 2020-01-01 --end-date 2023-12-31

# Check pipeline status without processing
python scripts/run_data_pipeline.py --status --start-date 2020-01-01

# List all available data chunks
python scripts/run_data_pipeline.py --list-chunks

# Force reprocessing of all chunks
python scripts/run_data_pipeline.py --start-date 2020-01-01 --force-reprocess
```

### Training with Best Model Strategy

```bash
# Train using best model strategy (loads existing best model if available)
python scripts/train_with_best_model.py --steps 100000
```

### Live Trading with Best Model

```bash
# Run live trader (will automatically use best model)
python scripts/run_live_trader.py
```

## Legacy Scripts (Deprecated)

The following scripts are now deprecated and should be replaced by the new integrated pipeline:
- `scripts/download_data.py`
- `scripts/preprocess_data.py`

Traditional training can still be done with `scripts/train_rl_agent.py`, but `scripts/train_with_best_model.py` 
is now the recommended approach for improved model management.
